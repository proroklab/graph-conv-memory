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
        ) -> TensorType["B", "MAX_EDGES", "MAX_EDGES", float, torch.sparse_coo]:  # type: ignore # noqa: F821

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

        # Compute a single, large sparse adj matrix of shape [B, max_nodes, max_nodes]
        import pdb; pdb.set_trace()







        source_edges = []
        sink_edges = []
        weights = []
        for b in range(B):
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

            #net_in[start_idx[b]:stop_idx[b]] = torch.cat((left_nodes, right_nodes), dim=-1)
            network_input = torch.cat((left_nodes, right_nodes), dim=-1)
            logits = self.edge_network(network_input).squeeze()

            gs_in = logits.repeat(self.num_edge_samples, 1, 1)
            soft = torch.nn.functional.gumbel_softmax(gs_in, hard=True, tau=0.01)
            edge_grad_path = self.ste(soft.sum(dim=0))
            edge_idx = edge_grad_path.nonzero()[:,1]

            sink_edges.append(left_idx[edge_idx])
            source_edges.append(right_idx[edge_idx])
            import pdb; pdb.set_trace()
            weights.append(edge_grad_path[edge_idx])

            # TODO gumbel softmax

        #logits = self.edge_network(net_in).squeeze() 
        # Logits are indexed as [B, T*tau
        

            
