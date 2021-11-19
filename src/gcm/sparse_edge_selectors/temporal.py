import torch
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
        ) -> TensorType["B", "MAX_EDGES", "MAX_EDGES", float, torch.sparse_coo]:  # type: ignore # noqa: F821
        # Connect each [t in T to T + tau] to [t - h for h in hops]

        batch_starts, batch_ends = util.get_batch_offsets(T + taus)
        edge_base: Union[List, torch.Tensor] = []

        # Build a base of edges (x - hop for all hops)
        # then we add the batch offsets to them
        for b in range(B):
            edge_base.append(torch.arange(T[b], T[b] + taus[b], device=nodes.device))

        # No edges to add
        if len(edge_base) < 1:
            # TODO don't hardcode max edges
            return torch.zeros((B, int(1e5), int(1e5)), device=nodes.device, layout=torch.sparse_coo, dtype=torch.float)
            empty_edges = torch.empty((2, 0), device=nodes.device, dtype=torch.long)
            empty_weights = torch.empty((0), device=nodes.device, dtype=torch.float)
            return empty_edges, empty_weights

        # [B, num_edges]
        edge_base = torch.stack(edge_base)
        sink_edges  = edge_base.unsqueeze(-1).repeat(1, 1, len(self.hops))
        source_edges = sink_edges - self.hops.to(nodes.device)
        batch_idx = torch.arange(B, device=nodes.device).unsqueeze(-1).unsqueeze(-1).expand(source_edges.shape)

        sink_edges = sink_edges.flatten()
        source_edges = source_edges.flatten()
        edge_batch = batch_idx.flatten()
        edge_idx = torch.stack([edge_batch, sink_edges, source_edges])
        weights = torch.ones(source_edges.shape, device=nodes.device)

        # Need to filter negative (invalid) indices
        mask = (source_edges >= 0) * (sink_edges > 0)
        filtered_edge_idx = edge_idx[:, mask]
        weights = weights[mask]
        adj = torch.sparse_coo_tensor(indices=filtered_edge_idx, values=weights, size=(B, int(1e5), int(1e5)), device=nodes.device)
        return adj
