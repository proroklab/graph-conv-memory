import torch
import itertools
from typing import Dict, Tuple, List
import gcm.util


@torch.jit.script
def up_to_num_nodes_idxs(
    adj: torch.Tensor, num_nodes: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Given num_nodes, returns idxs from adj
    up to but not including num_nodes. I.e.
    [batches, 0:num_nodes, num_nodes]. Note the order is
    sorted by (batches, num_nodes, 0:num_nodes) in ascending order"""
    seq_lens = num_nodes.unsqueeze(-1)
    N = adj.shape[-1]
    N_idx = torch.arange(N, device=adj.device).unsqueeze(0)
    N_idx = N_idx.expand(seq_lens.shape[0], N_idx.shape[1])
    # Do not include the current node
    N_idx = torch.nonzero(N_idx < num_nodes.unsqueeze(1))
    assert N_idx.shape[-1] == 2
    batch_idxs = N_idx[:, 0]
    past_idxs = N_idx[:, 1]
    curr_idx = num_nodes[batch_idxs]

    return batch_idxs, past_idxs, curr_idx

class BernoulliEdge(torch.nn.Module):
    """Add temporal bidirectional back edge, but only if we have >1 nodes
    E.g., node_{t} <-> node_{t-1}"""

    def __init__(
        self,
        input_size: int = 0,
        model: torch.nn.Sequential = None,
        backward_edges: bool = False,
        desired_num_edges: int = 5,
        deterministic: bool = False,
        # gradient_scale: float = 0.5
    ):
        super().__init__()
        self.backward_edges = backward_edges
        self.deterministic = deterministic
        self.desired_num_edges = desired_num_edges
        assert input_size or model, "Must specify either input_size or model"
        if model:
            self.edge_network = model
        else:
            # This MUST be done here
            # if done in forward model does not learn...
            self.edge_network = self.build_edge_network(input_size)
        if deterministic:
            self.sm = gcm.util.SparsegenLin(0.8)
        else:
            self.ste = gcm.util.StraightThroughEstimator()

    def build_edge_network(self, input_size: int) -> torch.nn.Sequential:
        """Builds a network to predict edges.
        Network input: (i || j)
        Network output: p(edge) in [0,1]
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

    def compute_new_adj(
        self,
        nodes: torch.Tensor,
        num_nodes: torch.Tensor,
        adj: torch.Tensor,
        B: int,
    ):
        """Computes a new adjacency matrix using the edge network.
        The edge network outputs logits for all possible edges,
        which are spardmaxed to produce edges. Edges are then
        placed into a new adjacency matrix"""
        # No edges for a single node
        if torch.max(num_nodes) < 1:
            return adj

        b_idxs, past_idxs, curr_idx = gcm.util.idxs_up_to_num_nodes(adj, num_nodes)
        # curr_idx > past_idxs
        # flows from past_idxs to j
        # so [j, past_idxs]
        curr_nodes = nodes[b_idxs, curr_idx]
        past_nodes = nodes[b_idxs, past_idxs]

        net_in = torch.cat((curr_nodes, past_nodes), dim=-1)
        logits = self.edge_network(net_in).squeeze()

        # Load logits into [B, nodes] matrix and set unfilled entries to large
        # negative value so unfilled entries don't affect spardmax
        # then spardmax per-batch (dim=-1)
        shaped_logits = torch.empty((B, torch.max(num_nodes)), device=nodes.device).fill_(-1e10)
        shaped_logits[b_idxs, past_idxs] = logits
        if self.deterministic:
            edges = self.sm(shaped_logits)
        else:
            # Multinomial straight-thru estimator
            gs_in = shaped_logits.unsqueeze(0).repeat(self.desired_num_edges, 1, 1)
            # int_edges in Z but we need in [0,1] -- straight thru estimator
            soft = torch.nn.functional.gumbel_softmax(gs_in, hard=True)
            edges = self.ste(soft.sum(dim=0))

        new_adj = adj.clone()
        # Reindexing edges in this manner ensures even if the edge network
        # went beyond -10e20 and set an invalid edge, it will not be used
        # at most, it affects the scaling for the valid edges
        new_adj[b_idxs, curr_idx, past_idxs] = edges[b_idxs, past_idxs]
        # Ensure we don't overwrite 1's in adj in case we have more than one
        # edge selector
        #new_adj = self.ste(new_adj + adj)
        #new_adj = gcm.util.diff_or([new_adj, adj])

        return new_adj

        """
        shaped_logits = torch.zeros_like(adj)
        new_adj = adj.clone()
        shaped_logits[b_idxs, curr_idx, past_idxs] = logits
        for b in range(B):
            if num_nodes[b] < 1:
                continue
            # TODO: should we multiply prev adj to care about
            # potentially other edge selectors in the queue?
            new_adj[b, num_nodes[b], :num_nodes[b]] = self.sm(
                shaped_logits[b, num_nodes[b], :num_nodes[b]].unsqueeze(0)
            )
        return new_adj
        """

    def forward(self, nodes, adj, weights, num_nodes, B):
        """A(i,j) = Ber[phi(i || j), e]

        Modify the nodes/adj_mats/state in-place by reference. Return value
        is not used.
        """

        # a(b,i,j) = gumbel_softmax(phi(n(b,i) || n(b,j))) for i, j < num_nodes
        # First run
        if self.edge_network[0].weight.device != nodes.device:
            self.edge_network = self.edge_network.to(nodes.device)

        if torch.max(num_nodes) < 1:
            return adj, weights

        new_adj = self.compute_new_adj(nodes, num_nodes, adj, B)
        return new_adj, weights

