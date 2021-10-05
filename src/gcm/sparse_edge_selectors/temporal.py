import torch
from typing import List, Any, Tuple, Union

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
patch_typeguard()


class TemporalEdge(torch.nn.Module):
    """Add temporal edges to the edge list"""

    def __init__(self, hops: List[int] = [1]):
        super().__init__()
        self.hops = torch.tensor(hops)

    @typechecked
    def forward(self, 
            nodes: TensorType["B", "N", "feat", float],
            edges: TensorType["B", 2, "E", int],
            weights: Union[None, TensorType["batch", 1, "E", float]],
            T: TensorType["B", int], 
            taus: TensorType["B", int],
    ) -> Tuple[
            TensorType["B", 2, "NE", int], 
            TensorType["B", 1, "NE", float]
        ]:
        # Connect each [t in T to T + tau] to [t - h for h in hops]
        # shape [B, t + tau]
        #edge_base = torch.stack([torch.arange(t, t + tau, device=nodes.device) for t in T]) 
        # shape [B, ?]
        edge_base = torch.cat([torch.arange(T[b], T[b] + taus[b], device=x.device) for b in range(B)])
        # shape [B, t + tau, hops]
        #edge_ends = edge_base.unsqueeze(-1).repeat(1,1, len(self.hops)) 
        import pdb; pdb.set_trace()
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
        edges = torch.cat((edges, new_edges), dim=-1)
        weights = torch.cat((weights, new_weights), dim=-1)
        return edges, weights



