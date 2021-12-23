import torch
import functools
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
        window: Union[int, None] = None,
        # Stores useful information in instance variables
        log_stats: bool = True
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
        self.log_stats = log_stats
        self.stats = {}
        self.tau_param = torch.nn.Parameter(torch.tensor([1.]))

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear): 
            torch.nn.init.orthogonal_(m.weight)

    def build_edge_network(self, input_size: int) -> torch.nn.Sequential:
        """Builds a network to predict edges.
        Network input: (i || j)
        Network output: logits(edge(i,j))
        """
        m = torch.nn.Sequential(
            torch.nn.Linear(2 * input_size, input_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(input_size),
            torch.nn.Linear(input_size, input_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(input_size),
            torch.nn.Linear(input_size, 1),
        )
        m.apply(self.init_weights)
        return m

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
                
        if list(self.parameters())[0].device != nodes.device:
            self = self.to(nodes.device)

        # Do for all batches at once
        #
        # Construct indices denoting all edges, which we sample from
        # Note that we only want to sample incoming edges from nodes T to T + tau
        edge_idx = []
        tril_inputs = T + taus
        for b in range(B):
            edge = torch.tril_indices(
                tril_inputs[b], tril_inputs[b], offset=-1, 
                dtype=torch.long,
                device=nodes.device,
            )
            # Use windows to reduce size, in case the graph is too big.
            # Remove indices outside of the window
            if self.window is not None:
                window_min_idx = max(0, T[b] - self.window)
                window_mask = edge[1] >= window_min_idx
                # Remove edges outside of window
                edge = edge[:, window_mask]

            # Filter edges -- we only want incoming edges to tau nodes
            # we should have no sinks < T
            edge = edge[:, edge[0] >= T[b]]


            batch = b * torch.ones(1, device=nodes.device, dtype=torch.long)
            batch = batch.expand(edge[-1].shape[-1])
            
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
        sparse_gs = True
        fast_gs = True
        if fast_gs:
            cutoff = 1 / (1 + self.num_edge_samples)
            gs_input = torch.sparse_coo_tensor(
                indices=edge_idx,
                values=logits,
                size=(B, nodes.shape[1], nodes.shape[1])
            )
            soft = util.sparse_gumbel_softmax(
                gs_input, dim=2, hard=False, tau=1.#self.tau_param.abs()
            )
            activation_mask = soft._values() > cutoff
            adj = torch.sparse_coo_tensor(
                indices=soft.indices()[:,activation_mask],
                values=(
                    soft.values()[activation_mask] 
                    / soft.values()[activation_mask].detach()
                ),
                size=(B, nodes.shape[1], nodes.shape[1])
            )

        if self.training:
            self.var = logits.var()
        if self.log_stats and self.training:
            self.stats["edges_per_node"] = (
                adj._values().numel() / taus.sum().detach()
            ).item()
            self.stats["edge_density"] = adj._values().numel() / edge_idx[0].numel()
            self.stats["logits_mean"] = logits.detach().mean().item()
            self.stats["logits_var"] = logits.detach().var().item()
            self.stats["temperature"] = self.tau_param.detach().item()
        return adj
