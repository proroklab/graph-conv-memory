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



        # TODO: also get rid of invalid (-1) edges
        batch_offsets = util.get_batch_offsets(T, taus)
        edge_base = [
            torch.arange(batch_offsets[b] + T[b], batch_offsets[b] + T[b] + taus[b], device=nodes.device) 
            for b in range(B) if T[b] > -1 # Initial nodes dont need edges
        ]

        # No edges to add
        if len(edge_base) < 1:
            empty_edges = torch.empty((2,0), device=nodes.device, dtype=torch.long)
            empty_weights = torch.empty((0.0), device=nodes.device, dtype=torch.float)
            return empty_edges, empty_weights

        edge_base = torch.cat(edge_base)
        # TODO: remove
        tmp_edges = edge_base.repeat(2,1)
        tmp_weights = edge_base.float()
        
        return tmp_edges, tmp_weights

        """
        # shape [B, t + tau, hops]
        edge_ends = edge_base.unsqueeze(-1).repeat(1, len(self.hops)) 
        # shape [B, t + tau, hops]
        edge_starts = edge_ends - self.hops

        new_edges = torch.cat((edge_starts, edge_ends), dim=-1).permute(1,0)
        new_weights = torch.ones_like(new_edges[0], dtype=torch.float)

        batch_offsets = T.cumsum(dim=0).roll(1)
        batch_offsets[0] = 0

        edge_offsets = batch_offsets.unsqueeze(-1).unsqueeze(-1).expand(-1,2,edges.shape[-1])
        offset_edges = edges + edge_offsets
        offset_edges_B_idx = torch.cat(
            [
                b * torch.ones(
                    edges.shape[-1], device=edges.device, dtype=torch.long
                ) for b in range(B)
            ]
        )
    # Filter invalid edges (those that were < 0 originally)
    # Swap dims (B,2,NE) => (2,B,NE)
    mask = (offset_edges >= edge_offsets).permute(1,0,2)
        mask = new_edges >= 0
        new_edges = new_edges.masked_select(mask).reshape(2,-1)
        new_weights = new_weights.masked_select(mask[0])
        """

        '''
        # flatten
        # Shape [B, 2, t + tau * hops]
        new_edges = torch.stack(
            (edge_ends.flatten(1,-1), edge_starts.flatten(1,-1)),
            dim=1
        )
        new_weights = torch.zeros_like(new_edges[:,:1,:])
        '''
        return new_edges, new_weights



