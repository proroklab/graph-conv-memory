import torch
import ray
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
        deterministic: bool = False
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
    ) -> Tuple[TensorType[2, "NE", int], TensorType["NE", float]]:  # type: ignore # noqa: F821
        # Connect each [t in T to T + tau] to [t - h for h in hops]

        # No edges to create
        if T.max() <= 1:
            return torch.zeros((2, 0), dtype=torch.long, device=nodes.device), torch.zeros((0), dtype=torch.long, device=nodes.device)

        if self.edge_network[0].weight.device != nodes.device:
            self.edge_network = self.edge_network.to(nodes.device)

        # Get new nodes
        sink_B_idxs, sink_tau_idxs = util.get_new_node_idxs(T, taus, b)
        # Get past nodes
        source_B_idxs, source_tau_idxs = util.get_valid_node_idxs(T, taus, b)

        # Each batch should produce |v1| * |v2| total edges
        # where v1 are sink nodes and v2 are source nodes for batch b
        # sum( taus[b] * T[b] )

        num_nodes = (taus * T).sum()
        net_in = torch.zeros(
            (num_nodes, 2 * nodes.shape[-1]),
            device=nodes.device
        )
        
        stop_idx = (taus * T).cumsum()
        start_idx = stop_idx.roll(1)
        start_idx[0] = 0
        for b in range(B):
            sink_idx_idx = torch.where(sink_B_idxs == b)
            sink_idx = sink_B_idxs[sink_idx_idx], sink_tau_idxs[sink_idx_idx]
            sink_nodes = nodes[*sink_idx]

            source_idx_idx = torch.where(source_B_idxs == b)
            source_idx = source_B_idxs[source_idx_idx], source_tau_idxs[source_idx_idx]
            source_nodes = nodes[*source_idx]

            left_nodes = sink_nodes.repeat(T[b], -1)
            right_nodes = source_nodes.repeat_interleave(taus[b], -1)

            net_in[start_idx:stop_idx] = torch.cat((left_nodes, right_nodes), dim=-1)

        logits = self.edge_network(net_in).squeeze() 
        # Logits are indexed as [B, T*tau
        

            
