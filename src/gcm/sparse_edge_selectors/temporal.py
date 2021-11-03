import torch
import ray
from typing import List, Any, Tuple, Union

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from gcm import util
patch_typeguard()


class TemporalEdge(torch.nn.Module):
    """Add temporal edges to the edge list"""

    def __init__(self, hops: List[int] = [1]):
        super().__init__()
        self.hops = torch.tensor(hops)

    @typechecked
    def forward(self, 
            nodes: TensorType["B", "N", "feat", float],
            T: TensorType["B", int], 
            taus: TensorType["B", int],
            B: int,
    ) -> Tuple[
            TensorType[2, "NE", int], 
            TensorType["NE", float],
        ]:
        # Connect each [t in T to T + tau] to [t - h for h in hops]
        # shape [B, t + tau]
        # shape [B, ?]

        '''
        new_edges = -1 * torch.ones((B, 2, len(self.hops)), device=edges.device, dtype=torch.long)
        for b in range(B):
            edge_base = torch.arange(T[b], T[b] + taus[b])
            edge_ends = edge_base.unsqueeze(-1).repeat(1, len(self.hops))
            edge_starts = edge_ends - self.hops
            new_edges[b] = torch.cat((edge_starts, edge_ends))
        edges, weights = util.add_edges(edges, new_edges, weights)
        ray.util.pdb.set_trace()
        return edges, weights
        '''



        edge_base = [
            torch.arange(T[b], T[b] + taus[b], device=nodes.device) 
            for b in range(B) if T[b] > -1 # Initial nodes dont need edges
        ]
        # No edges to add
        if len(edge_base) < 1:
            return
        ray.util.pdb.set_trace()
        # There are edges to add
        edge_base = torch.cat(edge_base)
        # shape [B, t + tau, hops]
        edge_ends = edge_base.unsqueeze(-1).repeat(1, len(self.hops)) 
        # shape [B, t + tau, hops]
        edge_starts = edge_ends - self.hops
        # flatten

        # Shape [B, 2, t + tau * hops]
        new_edges = torch.stack(
            (edge_ends.flatten(1,-1), edge_starts.flatten(1,-1)),
            dim=1
        )
        new_weights = torch.zeros_like(new_edges[:,:1,:])
        return new_edges, new_weights



