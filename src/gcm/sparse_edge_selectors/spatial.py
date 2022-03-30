import torch
import torch_geometric
from typing import List, Any, Tuple, Union

from torchtyping import TensorType, patch_typeguard  # type: ignore
from typeguard import typechecked  # type: ignore
from gcm import util
from torch_geometric.transforms.delaunay import Delaunay

#patch_typeguard()

class SpatialKNNEdge(torch.nn.Module):
    def __init__(self, position_slice, k, causal=True):
        # In meters
        super().__init__()
        self.k = k
        self.position_slice = position_slice
        self.causal = causal

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
            return torch.sparse_coo_tensor(
                indices=torch.zeros(3,0, dtype=torch.long, device=nodes.device),
                values=torch.zeros(0, device=nodes.device),
                size=(B, nodes.shape[1], nodes.shape[1])
            )

        pos = nodes[:, :, self.position_slice]
        if not self.causal:
            raise NotImplementedError()

        edges = []
        for b in range(B):
            sink_idx = torch.arange(T[b], T[b] + taus[b])
            source_idx = torch.arange(0, T[b] + taus[b])
            sink = pos[b, sink_idx]
            source = pos[b, source_idx]
            edge = torch_geometric.nn.knn(source, sink, self.k)
            # Filter out non-causal edges
            # TODO: This will break during backprop
            # we won't actually compute knn as we will prune
            # probably all edges
            # try topk: https://discuss.pytorch.org/t/k-nearest-neighbor-in-pytorch/59695/2
            mask = edge[0] > edge[1]
            edge = edge[:, mask]
            batch = b * torch.ones(edge.shape[-1], device=pos.device, dtype=torch.long)
            edges.append(torch.stack([batch, edge[0], edge[1]]))

        edges = torch.cat(edges, dim=-1)
        weights = torch.ones(edges.shape[-1], device=edges.device)
        adj = torch.sparse_coo_tensor(
            indices=edges, values=weights, size=(B, nodes.shape[1], nodes.shape[1])
        )
        return adj

class SpatialRadiusEdge(torch.nn.Module):
    def __init__(self, position_slice, radius=0.25, causal=True):
        # In meters
        super().__init__()
        self.radius = radius
        self.position_slice = position_slice
        self.causal = causal

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
            return torch.sparse_coo_tensor(
                indices=torch.zeros(3,0, dtype=torch.long, device=nodes.device),
                values=torch.zeros(0, device=nodes.device),
                size=(B, nodes.shape[1], nodes.shape[1])
            )

        pos = nodes[:, :, self.position_slice]
        edges = []
        for b in range(B):
            if self.causal:
                sink_idx, source_idx = util.get_causal_edges_one_batch(T[b], taus[b])
            else:
                new_idx = torch.arange(T[b], T[b] + taus[b], device=pos.device)
                old_idx = torch.arange(T[b] + taus[b], device=pos.device)
                sink_idx, source_idx = torch.cartesian_prod(old_idx, new_idx).unbind(dim=1)
            sink_nodes = pos[b, sink_idx]
            source_nodes = pos[b, source_idx]
            # For some reason, torch.cdist is really slow...
            dist = ((sink_nodes - source_nodes) ** 2).sum(dim=-1).sqrt()
            idx_idx = torch.where(dist < self.radius)
            sink_edges = sink_idx[idx_idx]
            source_edges = source_idx[idx_idx]
            batch = b * torch.ones(sink_edges.numel(), device=pos.device, dtype=torch.long)
            edges.append(
                torch.stack([batch, sink_edges, source_edges])
            )

        edges = torch.cat(edges, dim=-1)
        weights = torch.ones(edges.shape[-1], device=edges.device)
        adj = torch.sparse_coo_tensor(
            indices=edges, values=weights, size=(B, nodes.shape[1], nodes.shape[1])
        )
        return adj

'''
class SpatialDelaunayEdge(torch.nn.Module):
    """Add temporal edges to the edge list"""

    def __init__(self, position_slice):
        super().__init__()
        self.position_slice = position_slice

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

        pos = nodes[:, :, position_slice]
        for b in range(B):
            d = 


        # Build a base of edges (x - hop for all hops)
        # then we add the batch offsets to them
        # Add the -1 filler so we can stack the batches later
        edge_base = torch.empty(
            (B, taus.max()), device=nodes.device, dtype=torch.long
        ).fill_(-1)
        for b in range(B):
            edge_base[b, : taus[b]] = torch.arange(
                T[b], T[b] + taus[b], device=nodes.device
            )

        # No edges to add
        if len(edge_base) < 1:
            # TODO don't hardcode max edges
            return torch.zeros(
                (B, int(1e5), int(1e5)),
                device=nodes.device,
                layout=torch.sparse_coo,
                dtype=torch.float,
            )
            empty_edges = torch.empty((2, 0), device=nodes.device, dtype=torch.long)
            empty_weights = torch.empty((0), device=nodes.device, dtype=torch.float)
            return empty_edges, empty_weights

        sink_edges = edge_base.unsqueeze(-1).repeat(1, 1, len(self.hops))
        source_edges = sink_edges - self.hops.to(nodes.device)
        batch_idx = (
            torch.arange(B, device=nodes.device)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(source_edges.shape)
        )

        sink_edges = sink_edges.flatten()
        source_edges = source_edges.flatten()
        edge_batch = batch_idx.flatten()
        edge_idx = torch.stack([edge_batch, sink_edges, source_edges])
        weights = torch.ones(source_edges.shape, device=nodes.device)

        # Need to filter negative (invalid) indices
        mask = (source_edges >= 0) * (sink_edges > 0)
        filtered_edge_idx = edge_idx[:, mask]
        weights = weights[mask]
        adj = torch.sparse_coo_tensor(
            indices=filtered_edge_idx,
            values=weights,
            size=(B, int(1e5), int(1e5)),
            device=nodes.device,
        )
        return adj
'''
