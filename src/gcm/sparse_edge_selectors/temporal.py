import torch
import ray
from typing import List, Any, Tuple, Union

from torchtyping import TensorType, patch_typeguard  # type: ignore
from typeguard import typechecked  # type: ignore
from gcm import util

patch_typeguard()


class TemporalEdge(torch.nn.Module):
    """Add temporal edges to the edge list"""

    def __init__(self, hops: List[int] = [1]):
        super().__init__()
        self.hops = torch.tensor(hops)

    @typechecked
    def forward(
        self,
        nodes: TensorType["B", "N", "feat", float],  # type: ignore # noqa: F821
        T: TensorType["B", int],  # type: ignore # noqa: F821
        taus: TensorType["B", int],  # type: ignore # noqa: F821
        B: int,
    ) -> Tuple[TensorType[2, "NE", int], TensorType["NE", float]]:  # type: ignore # noqa: F821
        # Connect each [t in T to T + tau] to [t - h for h in hops]

        batch_offsets = util.get_batch_offsets(T, taus)
        edge_base: Union[List, torch.Tensor] = []
        edge_base_offsets: Union[List, torch.Tensor] = []

        # Build a base of edges (x - hop for all hops)
        # then we add the batch offsets to them
        for b in range(B):
            edge_base.append(torch.arange(T[b], T[b] + taus[b], device=nodes.device))

            edge_base_offsets.append(
                batch_offsets[b]
                * torch.ones(taus[b], device=nodes.device, dtype=torch.long)
            )

        # No edges to add
        if len(edge_base) < 1:
            empty_edges = torch.empty((2, 0), device=nodes.device, dtype=torch.long)
            empty_weights = torch.empty((0), device=nodes.device, dtype=torch.float)
            return empty_edges, empty_weights

        edge_base = torch.cat(edge_base)
        edge_base_offsets = torch.cat(edge_base_offsets)
        edge_ends = edge_base.unsqueeze(-1).repeat(1, len(self.hops))

        edge_starts = edge_ends - self.hops.to(nodes.device)
        edge_starts = edge_starts.flatten()
        edge_ends = edge_ends.flatten()
        # Remove invalid edges (<0) before we add offsets
        mask = edge_starts >= 0
        # Offset edges
        edge_starts = edge_starts + edge_base_offsets.repeat_interleave(len(self.hops))
        edge_ends = edge_ends + edge_base_offsets.repeat_interleave(len(self.hops))

        new_edges = torch.stack((edge_starts, edge_ends))
        new_weights = torch.ones_like(new_edges[0], dtype=torch.float)

        # Filter invalid edges
        new_edges = new_edges[:, mask]
        new_weights = new_weights[mask]

        assert torch.all(new_edges[0] < new_edges[1]), "Tried to add invalid edge"

        return new_edges, new_weights
