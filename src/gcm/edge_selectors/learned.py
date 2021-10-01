import torch
import itertools
from typing import Dict, Tuple, List
import gcm.util


class LearnedEdge(torch.nn.Module):
    """An edge selector where the prior is learned from data. An MLP 
    computes logits which create edges via either sampling or sparsemax."""

    def __init__(
        self,
        input_size: int = 0,
        model: torch.nn.Sequential = None,
        num_edge_samples: int = 5,
        deterministic: bool = False,
    ):
        """
        input_size: Feature dim size of GNN, not required if model is specificed
        model: Model for logits network, if not specified one is provided
        num_edge_samples: If not deterministic, how many samples to take from dist.
        determinstic: Whether edges are randomly sampled or argmaxed
        """
        super().__init__()
        self.deterministic = deterministic
        self.num_edge_samples = num_edge_samples
        assert input_size or model, "Must specify either input_size or model"
        if model:
            self.edge_network = model
        else:
            # This MUST be done here
            # if initialized in forward model does not learn...
            self.edge_network = self.build_edge_network(input_size)
        if deterministic:
            self.sm = gcm.util.Spardmax()
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
        which are spardmaxed or sampled to produce edges. Edges are then
        placed into a new adjacency matrix and returned."""
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
            gs_in = shaped_logits.unsqueeze(0).repeat(self.num_edge_samples, 1, 1)
            # int_edges in Z but we need in [0,1] -- straight thru estimator
            soft = torch.nn.functional.gumbel_softmax(gs_in, hard=True)
            edges = self.ste(soft.sum(dim=0))

        new_adj = adj.clone()
        # Reindexing edges in this manner ensures even if the edge network
        # went beyond -10e20 and set an invalid edge, it will not be used
        # at most, it affects the scaling for the valid edges
        # 
        # Ensure we don't overwrite 1's in adj in case we have more than one
        # edge selector
        # We don't want to add the old adj to the new adj,
        # because grads from previous rows will accumulate
        # and grad will explode
        new_adj[b_idxs, curr_idx, past_idxs] = self.ste(
            edges[b_idxs, past_idxs] + adj[b_idxs, curr_idx, past_idxs]
        )

        return new_adj

    def forward(self, nodes, adj, weights, num_nodes, B):
        # First run
        if self.edge_network[0].weight.device != nodes.device:
            self.edge_network = self.edge_network.to(nodes.device)

        # No self edges allowed
        if torch.max(num_nodes) < 1:
            return adj, weights

        new_adj = self.compute_new_adj(nodes, num_nodes, adj, B)
        return new_adj, weights

