import torch
from typing import List, Any, Tuple, Union

from torchtyping import TensorType, patch_typeguard  # type: ignore
from typeguard import typechecked  # type: ignore
from gcm import util

patch_typeguard()


class LearnedEdge(torch.nn.Module):
    """Add temporal edges to the edge list"""

    def __init__(
        self, 
        input_size: int = 0,
        num_edge_samples: int = 5,
        deterministic: bool = False,
        window: int = 0
    ):
        super().__init__()
        self.deterministic = deterministic
        self.num_edge_samples = num_edge_samples
        # This MUST be done here
        # if initialized in forward model does not learn...
        self.edge_network = self.build_edge_network(input_size)
        if deterministic:
            self.sm = util.Spardmax()
        self.ste = util.StraightThroughEstimator()

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
            return torch.zeros((2, 0), dtype=torch.long, device=nodes.device), torch.zeros((0), dtype=torch.long, device=nodes.device)

        if self.edge_network[0].weight.device != nodes.device:
            self.edge_network = self.edge_network.to(nodes.device)

        # TODO: use window

        # Get new nodes
        sink_B_idxs, sink_tau_idxs = util.get_new_node_idxs(T, taus, B)
        # Get past nodes
        source_B_idxs, source_tau_idxs = util.get_valid_node_idxs(T, taus, B)

        # Each batch should produce |v1| * |v2| total edges
        # where v1 are sink nodes and v2 are source nodes for batch b
        # where |v1| == taus[b] + T[b]
        # and |v2| == taus[b]

        v1_mag = taus + T
        v2_mag = taus
        num_nodes = torch.sum(v1_mag * v2_mag)



        net_in = torch.zeros(
            (num_nodes, 2 * nodes.shape[-1]),
            device=nodes.device
        )
        
        stop_idx = (taus * (T + taus)).cumsum(dim=0)
        start_idx = stop_idx.roll(1)
        start_idx[0] = 0


        adj_idxs = []
        adj_vals = []
        for b in range(B):
            # Construct all valid edge combinations
            edge_idx = torch.tril_indices(
                T[b] + taus[b], T[b] + taus[b], offset=-1
            )
            # Don't evaluate incoming edges for the T entries, as we are only
            # interested in incoming data for T + tau
            sink_idx, source_idx = edge_idx[:, edge_idx[0] > T[b]] 

            sink_nodes = nodes[b, sink_idx]
            source_nodes = nodes[b, source_idx]
            """
            sink_idx_idx = torch.where(sink_B_idxs == b)
            sink_idx = sink_B_idxs[sink_idx_idx], sink_tau_idxs[sink_idx_idx]
            sink_nodes = nodes[sink_idx]

            source_idx_idx = torch.where(source_B_idxs == b)
            source_idx = source_B_idxs[source_idx_idx], source_tau_idxs[source_idx_idx]
            source_nodes = nodes[source_idx]

            left_nodes = sink_nodes.repeat(T[b] + taus[b], 1)
            left_idx = sink_idx_idx[0].repeat(T[b] + taus[b])
            right_nodes = source_nodes.repeat_interleave(taus[b], 0)
            right_idx = source_idx_idx[0].repeat_interleave(T[b] + taus[b])
            """

            #net_in[start_idx[b]:stop_idx[b]] = torch.cat((left_nodes, right_nodes), dim=-1)
            # Thru network for logits
            network_input = torch.cat((sink_nodes, source_nodes), dim=-1)
            logits = self.edge_network(network_input).squeeze()

            # Logits to probabilities via gumbel softmax
            gs_in = logits.repeat(self.num_edge_samples, 1, 1)
            soft = torch.nn.functional.gumbel_softmax(gs_in, hard=True)
            # Store gumbel softmax output so we can propagate their gradients
            # thru weights/values in the adj
            sampled_edge_grad_path = self.ste(soft.sum(dim=0)).squeeze(0)
            sampled_edge_idx = sampled_edge_grad_path.nonzero().squeeze(1)

            # Add [B, source, sink] edge indices
            adj_idx = torch.stack((
                b * torch.ones(sampled_edge_idx.shape[-1]),
                sink_idx[sampled_edge_idx],
                source_idx[sampled_edge_idx],
            ))
            adj_idxs.append(adj_idx)

            # Gradients are stored here
            adj_vals.append(sampled_edge_grad_path[sampled_edge_idx])


        indices = torch.cat(adj_idxs, dim=-1)
        values = torch.cat(adj_vals)

        adj = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(B, nodes.shape[1], nodes.shape[1])
        )
        return adj



        #logits = self.edge_network(net_in).squeeze() 
        # Logits are indexed as [B, T*tau
        

            
