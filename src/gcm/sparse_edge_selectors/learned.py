import torch
from typing import List, Any, Tuple, Union

from torchtyping import TensorType, patch_typeguard  # type: ignore
from typeguard import typechecked  # type: ignore
from gcm import util

#patch_typeguard()


class LearnedEdge(torch.nn.Module):
    """Add temporal edges to the edge list"""

    def __init__(
        self, 
        # Feature size of a graph node
        input_size: int = 0,
        # Custom model, if None, one will be created for you
        model: Union[None, torch.nn.Module] = None,
        # Number of edges to sample per node (upper bounds the
        # number of edges for each node)
        num_edge_samples: int = 5,
        # Whether to randomly sample using gumbel softmax
        # or use sparsemax
        deterministic: bool = False,
        # Only consider edges to vertices in a fixed-size window
        # this reduces memory usage but prohibits edges to nodes outside
        # the window. Use None for no window (all possible edges)
        window: Union[int, None] = None
    ):
        super().__init__()
        assert model or input_size, "Must specify either input_size or model"
        self.deterministic = deterministic
        self.num_edge_samples = num_edge_samples
        # This MUST be done here
        # if initialized in forward model does not learn...
        self.edge_network = self.build_edge_network(input_size) if model is None else model
        if deterministic:
            self.sm = util.Spardmax()
        self.ste = util.StraightThroughEstimator()
        self.window = window

    def build_edge_network(self, input_size: int) -> torch.nn.Sequential:
        """Builds a network to predict edges.
        Network input: (i || j)
        Network output: logits(edge(i,j))
        """
        return torch.nn.Sequential(
            torch.nn.Linear(2 * input_size, input_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(input_size),
            torch.nn.Linear(input_size, input_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(input_size),
            torch.nn.Linear(input_size, 1),
        )

    @typechecked
    def forward(
        self,
        nodes: TensorType["B", "N", "feat", float],  # type: ignore # noqa: F821
        T: TensorType["B", int],  # type: ignore # noqa: F821
        taus: TensorType["B", int],  # type: ignore # noqa: F821
        B: int,
        ) -> TensorType["B", "N", "N", float, torch.sparse_coo]:  # type: ignore # noqa: F821

        # No edges to create
        if (T + taus).max() <= 1:
            return torch.sparse_coo_tensor(
                indices=torch.zeros(3,0, dtype=torch.long, device=nodes.device),
                values=torch.zeros(0, device=nodes.device),
                size=(B, nodes.shape[1], nodes.shape[1])
            )
                
        if list(self.edge_network.parameters())[0].device != nodes.device:
            self.edge_network = self.edge_network.to(nodes.device)

        # TODO: use window

        # Do for all batches at once
        #
        # Construct indices denoting all edges, which we sample from
        # Note that we only want to sample incoming edges from nodes T to T + tau
        edge_idx = []
        for b in range(B):
            # Use windows to reduce size, in case the graph is too big
            if self.window is not None:
                window_min_idx = max(0, T[b] - self.window)
            else:
                window_min_idx = 0
            edge = torch.tril_indices(
                T[b] + taus[b], T[b] + taus[b], offset=-1, 
                dtype=torch.long,
                device=nodes.device,
            )
            window_mask = edge[1] >= window_min_idx
            # Remove edges outside of window
            edge = edge[:, window_mask]
            # Filter edges -- we only want incoming edges to tau nodes
            # we should have no sinks < T
            edge = edge[:, edge[0] >= T[b]]


            batch = b * torch.ones(edge[-1].shape[-1], device=nodes.device, dtype=torch.long)
            edge_idx.append(torch.cat((batch.unsqueeze(0), edge), dim=0))


        # Shape [3, N] denoting batch, sink, source
        # these indices denote nodes pairs being fed to network
        edge_idx = torch.cat(edge_idx, dim=-1)
        batch_idx, sink_idx, source_idx = edge_idx.unbind()
        # Feed node pairs to network
        sink_nodes = nodes[batch_idx, sink_idx]
        source_nodes = nodes[batch_idx, source_idx]
        network_input = torch.cat((sink_nodes, source_nodes), dim=-1)
        # Logits is of shape [N]
        logits = self.edge_network(network_input).squeeze()
        # TODO rather than sparse to dense conversion, implement
        # a sparse gumbel softmax
        # TODO: This will be a dense NxN matrix at some point
        # we should offset max() - min()
        # MAKE SURE TO RETRANSFORM INDICES BELOW
        sink_shift = sink_idx.min()
        source_shift = source_idx.min()
        gs_input = torch.empty(
            (
                batch_idx.max() + 1, 
                sink_idx.max() - sink_shift + 1, 
                source_idx.max() - source_shift + 1
            ),
            device=nodes.device, dtype=torch.float
        ).fill_(torch.finfo(torch.float).min)
        gs_input[batch_idx, sink_idx - sink_shift, source_idx - source_shift] = logits
        # Draw num_samples from gs distribution
        # TODO mismatch between adj_idx and edge_idx
        # e.g. 0,0,0 is not in edge_idx but is in adj_idx
        #
        # it is performing softmax even on rows of all zeros
        # and selecting an entry with 1e-38 prob
        gs_input = gs_input.repeat(self.num_edge_samples, 1, 1, 1)
        soft = torch.nn.functional.gumbel_softmax(gs_input, hard=True, dim=3)
        # Clamp adj to 1
        edges = self.ste(soft.sum(dim=0))
        # Only extract valid edges
        # as we min-padded the gs_input matrix to make it dense.
        # Rows < T should be all zero (not have any incoming edges)
        # but gs will have made these rows nonzero
        # so let's ignore the padded rows and only extract the valid sampled edges
        valid_edges = edges[batch_idx, sink_idx - sink_shift, source_idx - source_shift] 
        # Of 
        valid_edge_mask = valid_edges > 0

        # Sampled edge indices
        sampled_idx = edge_idx[:, valid_edge_mask]
        # and the sampled results, to propagate grad thru adj_vals
        sampled_vals = valid_edges[valid_edge_mask]


        adj = torch.sparse_coo_tensor(
            indices=sampled_idx,
            values=sampled_vals,
            size=(B, nodes.shape[1], nodes.shape[1])
        )
        return adj

