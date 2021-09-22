import torch
from typing import List
from gcm.util import diff_or, Spardmax

# VISDOM TOP HALF PROVIDES BEST PERF

# adj[0,3] = 1
# neigh = matmul(adj, nodes) = nodes[0]
# [i,j] => base[j] neighbor[i]
# propagates from i to j

# neighbor: torch.matmul(Adj[i, j], x) = x[i] = adj[i]
# self: adj[j]
# Vis: should be top half of visdom


class TemporalBackedge(torch.nn.Module):
    """Add temporal directional back edge, e.g., node_{t} -> node_{t-1}"""

    def __init__(self, hops: List[int] = [1], direction="forward", learned=False, learning_window=10, deterministic=False, num_samples=3):
        """
        Hops: number of hops in the past to connect to
        E.g. [1] is t <- t-1, [2] is t <- t-2,
        [5,8] is t <- t-5 AND t <- t-8

        Direction: Directionality of graph edges. You likely want
        'forward', which indicates information flowing from past
        to future. Backward is information from future to past,
        and both is both.
        """
        super().__init__()
        self.hops = hops
        assert direction in ["forward", "backward", "both"]
        self.direction = direction
        self.learned = learned
        if learned:
            self.window = torch.nn.Parameter(torch.ones(learning_window))
            self.num_samples = num_samples
            self.deterministic = deterministic
            if deterministic:
                self.spardmax = Spardmax()

    def learned_forward(self, nodes, adj_mats, edge_weights, num_nodes, B):
        if self.window.device != nodes.device:
            self.window = self.window.to(nodes.device)
        for b in range(B):
            if num_nodes[b] == 0:
                continue
            relative_window = self.window[:num_nodes[b]]
            if self.deterministic:
                mask = self.spardmax(
                    relative_window.reshape(1, -1) 
                ).reshape(-1)
            else:
                masks = []
                for i in range(self.num_samples):
                    masks.append(torch.nn.functional.gumbel_softmax(relative_window, hard=True))
                mask = diff_or(masks)
            adj_mats[b][num_nodes[b]][:num_nodes[b]] = adj_mats[b][num_nodes[b]][:num_nodes[b]] + mask
        return adj_mats, edge_weights

    def deterministic_forward(self, nodes, adj_mats, edge_weights, num_nodes, B):
        for hop in self.hops:
            [valid_batches] = torch.where(num_nodes >= hop)
            if self.direction in ["forward", "both"]:
                adj_mats[
                    valid_batches,
                    num_nodes[valid_batches],
                    num_nodes[valid_batches] - hop,
                ] = 1
            if self.direction in ["backward", "both"]:
                adj_mats[
                    valid_batches,
                    num_nodes[valid_batches] - hop,
                    num_nodes[valid_batches],
                ] = 1

        return adj_mats, edge_weights
        

    def forward(self, nodes, adj_mats, edge_weights, num_nodes, B):
        if self.learned:
            return self.learned_forward(nodes, adj_mats, edge_weights, num_nodes, B)
        
        return self.deterministic_forward(nodes, adj_mats, edge_weights, num_nodes, B)
