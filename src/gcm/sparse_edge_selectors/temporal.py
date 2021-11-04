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

        # TODO: also get rid of invalid (-1, -2, ...) edges
        batch_offsets = util.get_batch_offsets(T, taus)
        
        edge_base = [
            torch.arange(
                T[b], T[b] + taus[b], 
                device=nodes.device
            ) 
            for b in range(B)  
        ]

        # No edges to add
        if len(edge_base) < 1:
            empty_edges = torch.empty((2,0), device=nodes.device, dtype=torch.long)
            empty_weights = torch.empty((0.0), device=nodes.device, dtype=torch.float)
            return empty_edges, empty_weights

        edge_base = torch.cat(edge_base)
        # shape [B, t + tau, hops]
        edge_ends = edge_base.unsqueeze(-1).repeat(1, len(self.hops)) 
        # shape [B, t + tau, hops]
        # Remove invalid edges (<0)

        edge_starts = edge_ends - self.hops.to(nodes.device)
        edge_starts = edge_starts.flatten()
        edge_ends = edge_ends.flatten()
        mask = (edge_starts >= 0)

        new_edges = torch.stack((edge_starts, edge_ends))
        new_weights = torch.ones_like(new_edges[0], dtype=torch.float)

        # Filter invalid edges
        new_edges = new_edges[:, mask]
        new_weights = new_weights[mask]

        return new_edges, new_weights
