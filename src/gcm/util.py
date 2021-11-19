import torch
import numpy as np
import torch_geometric
#import sparsemax
from typing import Tuple, List


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
        # return torch.nn.functional.hardtanh(grad_output)


class StraightThroughEstimator(torch.nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x


class Spardmax(torch.nn.Module):
    """A hard version of sparsemax"""

    def __init__(self, dim=-1, cutoff=0):
        super().__init__()
        self.dim = dim
        self.cutoff = cutoff
        self.sm = sparsemax.Sparsemax(dim)

    def forward(self, x):
        # Straight through.
        y_soft = self.sm(x)
        y_hard = (y_soft > self.cutoff).float()
        return y_hard - y_soft.detach() + y_soft


class Hardmax(torch.nn.Module):
    def __init__(self, dim=-1, cutoff=0.2):
        super().__init__()
        self.dim = dim
        self.cutoff = cutoff
        self.sm = torch.nn.Softmax(dim)

    def forward(self, x):
        # Straight through.
        y_soft = self.sm(x)
        y_hard = (y_soft > self.cutoff).float()
        return y_hard - y_soft.detach() + y_soft


@torch.jit.script
def get_nonpadded_idxs(T: torch.Tensor, taus: torch.Tensor, B: int):
    """Get the non-padded indices of a zero-padded
    batch of observations. In other words, get only valid elements and discard
    the meaningless zeros."""
    dense_B_idxs = torch.cat(
        [torch.ones(taus[b], device=T.device, dtype=torch.long) * b for b in range(B)]
    )
    # These must not be offset by T like get_new_node_idxs
    dense_tau_idxs = torch.cat(
        [torch.arange(taus[b], device=T.device) for b in range(B)]
    )
    return dense_B_idxs, dense_tau_idxs


@torch.jit.script
def get_new_node_idxs(T: torch.Tensor, taus: torch.Tensor, B: int):
    """Given T and tau tensors, return indices matching batches to taus.
    These tell us which elements in the node matrix we have just added
    during this iteration, and organize them by batch.

    E.g.
    g_idxs = torch.where(B_idxs == 0)
    zeroth_graph_new_nodes = nodes[B_idxs[g_idxs], tau_idxs[g_idxs]]
    """
    # TODO: batch this using b_idx and cumsum
    B_idxs = torch.cat(
        [torch.ones(taus[b], device=T.device, dtype=torch.long) * b for b in range(B)]
    )
    tau_idxs = torch.cat(
        [torch.arange(T[b], T[b] + taus[b], device=T.device) for b in range(B)]
    )
    return B_idxs, tau_idxs


@torch.jit.script
def get_valid_node_idxs(T: torch.Tensor, taus: torch.Tensor, B: int):
    """Given T and tau tensors, return indices matching batches to taus.
    These tell us which elements in the node matrix are valid for convolution,
    and organize them by batch.

    E.g.
    g_idxs = torch.where(B_idxs == 0)
    zeroth_graph_all_nodes = nodes[B_idxs[g_idxs], tau_idxs[g_idxs]]
    """
    # TODO: batch this using b_idx and cumsum
    B_idxs = torch.cat(
        [
            torch.ones(T[b] + taus[b], device=T.device, dtype=torch.long) * b
            for b in range(B)
        ]
    )
    tau_idxs = torch.cat(
        [torch.arange(0, T[b] + taus[b], device=T.device) for b in range(B)]
    )
    return B_idxs, tau_idxs


#@torch.jit.script
#def get_node_offsets(T: torch.Tensor):
#    """Get node offsets into flattened node tensor. Returns two tensors
#    denoting [start, end] of each batch (inclusive)"""
#    # Initial offset is zero, not T + tau, roll into place
#    batch_offsets = T.cumsum(dim=0).roll(1)
#    batch_offsets[0] = 0
#    return batch_offsets

def get_batch_offsets(T: torch.Tensor):
    """Get edge offsets into the flattened edge tensor."""
    batch_ends = T.cumsum(dim=0)
    batch_starts = batch_ends.roll(1)
    batch_starts[0] = 0
    #batch_starts[1:] += 1

    return batch_starts, batch_ends


def pack_hidden(hidden, B, max_edges, mode="dense", edge_fill=-1, weight_fill=1.0):
    assert mode in ["dense", "coo"]
    if mode == "dense":
        return _pack_hidden(*hidden, B, max_edges, edge_fill=-1, weight_fill=1.0)
    else:
        return _pack_hidden_coo(*hidden, B, max_edges)


def unpack_hidden(hidden, B, mode="dense"):
    assert mode in ["dense", "coo"]
    if mode == "dense":
        nodes, flat_edges, flat_weights, T, _ = _unpack_hidden(*hidden, B)
    else:
        nodes, flat_edges, flat_weights, T = _unpack_hidden_coo(*hidden, B)

    # Finally, remove duplicate edges and weights
    # but only if we have edges
    if flat_edges.numel() > 0:
        # The following can't be jitted, so it sits in this fn
        # Make sure idxs are removed alongside edges and weights
        flat_edges, flat_weights = torch_geometric.utils.coalesce(
            flat_edges, flat_weights, reduce="mean"
        )

    return nodes, flat_edges, flat_weights, T



# @torch.jit.script
def _pack_hidden_coo(
    nodes: torch.Tensor,
    edges: torch.Tensor,
    weights: torch.Tensor,
    T: torch.Tensor,
    B: int,
    max_edges: int=int(1e5),
):
    """Converts the hidden states to a torch.sparse_coo representation

    Unflatten edges from [2, k* NE] to [B, 2, max_edges].  Combines edges
    and weights into a single adjacency matrix

    Returns an updated hidden representation"""

    #batch_base = T.cumsum(dim=0)
    #batch_starts = batch_ends.roll(1)
    #batch_starts[0] = 0
    # nodes, edges, weights, T = hidden
    """
    batch_base = T.cumsum(dim=0)
    batch_starts = batch_base.roll(1)
    batch_starts[0] = 0
    batch_ends = batch_base + 1
    """
    batch_ends = (T + 1).cumsum(dim=0)
    batch_starts = batch_ends.roll(1)
    batch_starts[0] = 0
    batch_test = T.cumsum(dim=0)
    batch_test2 = batch_test.roll(1)
    batch_test2[0] = 0
    # TODO: not yet working

    #coo_edges = []
    coo_batch = []
    coo_source = []
    coo_sink = []

    #stacked_starts = batch_starts.expand(B, edges.shape[-1]).T
    stacked_starts = batch_starts.expand(edges.shape[-1], B).T
    #stacked_ends = batch_ends.expand(B, edges.shape[-1]).T
    stacked_ends = batch_ends.expand(edges.shape[-1], B).T
    source_masks = (stacked_starts <= edges[0]) * (edges[0] < stacked_ends)
    sink_masks = (stacked_starts < edges[1]) * (edges[1] <= stacked_ends)
    # Mask is shape [B, edges]
    masks = source_masks * sink_masks
    # Stack edges to be [B, Edges], then select 
    source_edges = edges[0].expand(B, -1).masked_select(masks)
    sink_edges = edges[1].expand(B, -1).masked_select(masks)
    # Now offset edges by batch
    source_edges -= stacked_starts.masked_select(masks)
    sink_edges -= stacked_starts.masked_select(masks)
    edge_batch = masks.nonzero()[:,0]

    coo_edges = torch.stack([edge_batch, source_edges, sink_edges])
    adj = torch.sparse_coo_tensor(indices=coo_edges, values=weights, size=(B, 2, max_edges))
    #import pdb; pdb.set_trace()

    return nodes, adj, T





# @torch.jit.script
def _pack_hidden(
    nodes: torch.Tensor,
    edges: torch.Tensor,
    weights: torch.Tensor,
    T: torch.Tensor,
    B: int,
    max_edges: int,
    edge_fill: int = -1,
    weight_fill: float = 1.0,
):
    """Converts the hidden states to a dense representation

    Unflatten edges from [2, k* NE] to [B, 2, max_edges].  In other words, prep
    edges and weights for dense transport (ray).

    Returns an updated hidden representation"""

    # nodes, edges, weights, T = hidden
    batch_ends = T.cumsum(dim=0)
    batch_starts = batch_ends.roll(1)
    batch_starts[0] = 0
    dense_edges = torch.zeros((B, 2, max_edges), dtype=torch.long).fill_(edge_fill)
    dense_weights = torch.zeros((B, 1, max_edges), dtype=torch.float).fill_(weight_fill)

    for b in range(B):
        source_mask = (batch_starts[b] <= edges[0]) * (edges[0] < batch_ends[b])
        sink_mask = (batch_starts[b] < edges[1]) * (edges[1] < batch_ends[b])
        mask = source_mask * sink_mask

        # Only if we have edges
        if edges[:, mask].numel() > 0:
            batch_edges = edges[:, mask] - batch_starts[b]
            batch_weights = weights[mask]
            max_indices = min(batch_edges.shape[-1], max_edges)
            # More than max edges
            if max_indices < batch_edges.shape[-1]:
                print(
                    "Warning: Batch has {batch_edges.shape[-1]} edges, which is greater than max"
                    f"edges ({max_edges}) dropping the first {batch_edges.shape[-1] - max_edges} edges"
                )
            dense_edges[b, :, :max_indices] = batch_edges[:, :max_indices]
            dense_weights[b, 0, :max_indices] = batch_weights[:max_indices]

    return nodes, dense_edges, dense_weights, T

def _unpack_hidden_coo(
    nodes: torch.Tensor,
    adj: torch.Tensor,
    T: torch.Tensor,
    B: int,
):
    """Converts torch.sparse_coo adj to a edge list representation. Do NOT
    invert the torch.sparse_coo, as it is already done here.

    Flatten edges from [B, 2, max_edges] to [2, k * NE].  

    Returns edges [B,2,NE] and weights [B,1,NE]"""
    batch_offsets = T.cumsum(dim=0).roll(1)
    batch_offsets[0] = 0
    batch_offsets[1:] += 1

    batch_idx = adj.indices()[0]
    offset_edges = batch_offsets[batch_idx] + adj.indices()[1:]

    flat_edges = offset_edges
    flat_weights = adj.values()

    return nodes, flat_edges, flat_weights, T

# @torch.jit.script
def _unpack_hidden(
    nodes: torch.Tensor,
    edges: torch.Tensor,
    weights: torch.Tensor,
    T: torch.Tensor,
    B: int,
):
    """Converts dense hidden states to a sparse representation

    Flatten edges from [B, 2, max_edges] to [2, k * NE].  In other words, prep
    edges and weights for dense transport (ray).

    Returns edges [B,2,NE] and weights [B,1,NE]"""
    # Get masks for valid edges/weights
    mask = edges >= 0  # Shape [B,2,max_edge]
    weight_mask = (mask[:, 0] * mask[:, 1]).unsqueeze(1)  # Shape [B,1,max_edge]
    edge_mask = weight_mask.expand(-1, 2, -1)  # [B,2,max_edge]

    batch_offsets = T.cumsum(dim=0).roll(1)
    batch_offsets[0] = 0

    edge_offsets = (
        batch_offsets.unsqueeze(-1).unsqueeze(-1).expand(-1, 2, edges.shape[-1])
    )
    offset_edges = edges + edge_offsets
    offset_edges_B_idx = torch.cat(
        [
            b * torch.ones(edges.shape[-1], device=edges.device, dtype=torch.long)
            for b in range(B)
        ]
    )

    # Now filter edges, weights, and indices using masks
    # this squeezes from from (2,B,NE) => (2,B*NE)
    # We permute to put the in/out index in the first dimension,
    # otherwise edge ordering is incorrect
    flat_edges = (
        offset_edges.permute(1, 0, 2)
        .masked_select(edge_mask.permute(1, 0, 2))
        .reshape(2, -1)
    )
    flat_weights = (
        weights.permute(1, 0, 2).masked_select(weight_mask.permute(1, 0, 2)).flatten()
    )
    flat_B_idx = offset_edges_B_idx.masked_select(weight_mask.flatten())

    return nodes, flat_edges, flat_weights, T, flat_B_idx


def flatten_adj(adj, T, taus, B):
    """Flatten a torch.coo_sparse [B, MAX_NODES, MAX_NODES] to [2, NE] and
    adds offsets to avoid collisions.
    This readies a sparse tensor for torch_geometric GNN
    ingestion.

    Returns edges, weights, and corresponding batch ids
    """
    # Get batch offsets is wrong (off by one)
    #batch_starts = get_node_offsets(T + taus)
    batch_starts, batch_ends = get_batch_offsets(T + taus)


    batch_idx = adj._indices()[0]
    edge_offsets = batch_starts[batch_idx]
    flat_edges = adj._indices()[1:] + edge_offsets
    flat_weights = adj._values()

    #import pdb; pdb.set_trace()
    if flat_edges.numel() > 0:
        # Make sure idxs are removed alongside edges and weights
        flat_edges, [flat_weights, batch_idx] = torch_geometric.utils.coalesce(
            flat_edges, [flat_weights, batch_idx], reduce="mean"
        )

    return flat_edges, flat_weights, batch_idx
    

def unflatten_adj(edges, weights, batch_idx, T, taus, B, max_edges):
    """Unflatten edges [2,NE], weights: [NE], and batch_idx [NE]
    into a torch.coo_sparse adjacency matrix of [B, NE, NE]"""
    #batch_starts = get_node_offsets(T + taus)
    batch_starts, batch_ends = get_batch_offsets(T + taus)

    edge_offsets = batch_starts[batch_idx]
    adj_edge_idx = edges - edge_offsets
    adj_idx = torch.stack([batch_idx, adj_edge_idx[0], adj_edge_idx[1]])
    adj_val = weights
    
    return torch.sparse_coo_tensor(
        indices=adj_idx, values=adj_val, size=(B, max_edges, max_edges)
    )


def flatten_edges_and_weights(edges, weights, T, taus, B):
    """Flatten edges from [B, 2, NE] to [2, k * NE], coalescing
    and removing invalid edges (-1). In other words, prep
    edges and weights for GNN ingestion.

    Returns flattened edges, weights, and corresponding
    batch indices"""
    batch_offsets = get_batch_offsets(T, taus)
    edge_offsets = (
        batch_offsets.unsqueeze(-1).unsqueeze(-1).expand(-1, 2, edges.shape[-1])
    )
    offset_edges = edges + edge_offsets
    offset_edges_B_idx = torch.cat(
        [
            b * torch.ones(edges.shape[-1], device=edges.device, dtype=torch.long)
            for b in range(B)
        ]
    )
    # Filter invalid edges (those that were < 0 originally)
    # Swap dims (B,2,NE) => (2,B,NE)
    mask = (offset_edges >= edge_offsets).permute(1, 0, 2)
    stacked_mask = (mask[0] & mask[1]).unsqueeze(0).expand(2, -1, -1)
    # Now filter edges, weights, and indices using masks
    # Careful, mask select will automatically flatten
    # so do it last, this squeezes from from (2,B,NE) => (2,B*NE)
    flat_edges = edges.permute(1, 0, 2).masked_select(stacked_mask).reshape(2, -1)
    flat_weights = weights.permute(1, 0, 2).masked_select(stacked_mask[0]).flatten()
    flat_B_idx = offset_edges_B_idx.masked_select(stacked_mask[0].flatten())

    # Finally, remove duplicate edges and weights
    # but only if we have edges
    if flat_edges.numel() > 0:
        # Make sure idxs are removed alongside edges and weights
        flat_edges, [flat_weights, flat_B_idx] = torch_geometric.utils.coalesce(
            flat_edges, [flat_weights, flat_B_idx], reduce="min"
        )

    return flat_edges, flat_weights, flat_B_idx


def flatten_nodes(nodes: torch.Tensor, T: torch.Tensor, taus: torch.Tensor, B: int):
    """Flatten nodes from [B, N, feat] to [B * N, feat] for ingestion
    by the GNN.

    Returns flattened nodes and corresponding batch indices"""
    batch_offsets, _ = get_batch_offsets(T + taus)
    #batch_offsets, end_offset = get_edge_offsets(T + taus)
    B_idxs, tau_idxs = get_valid_node_idxs(T, taus, B)
    flat_nodes = nodes[B_idxs, tau_idxs]
    # Extracting belief requires batch-tau indices (newly inserted nodes)
    # return these too
    # Flat nodes are ordered B,:T+tau (all valid nodes)
    # We want B,T:T+tau (new nodes), which is batch_offsets:batch_offsets + tau
    output_node_idxs = torch.cat(
        [
            torch.arange(
                batch_offsets[b] + T[b],
                batch_offsets[b] + T[b] + taus[b],
                device=nodes.device,
            )
            for b in range(B)
        ]
    )
    return flat_nodes, output_node_idxs


@torch.jit.script
def diff_or(tensors: List[torch.Tensor]):
    """Differentiable OR operation bewteen n-tuple of tensors
    Input: List[tensors in {0,1}]
    Output: tensor in {0,1}"""
    print("This seems to dilute gradients, dont use it")
    res = torch.zeros_like(tensors[0])
    for t in tensors:
        tmp = res.clone()
        res = tmp + t - tmp * t
    return res


@torch.jit.script
def diff_or2(tensors: List[torch.Tensor]):
    """Differentiable OR operation bewteen n-tuple of tensors
    Input: List[tensors in {0,1}]
    Output: tensor in {0,1}"""
    print("This seems to dilute gradients, dont use it")
    # This nice form is actually slower than the matrix mult form
    return 1 - (1 - torch.stack(tensors, dim=0)).prod(dim=0)


@torch.jit.script
def idxs_up_to_including_num_nodes(
    nodes: torch.Tensor, num_nodes: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given nodes and num_nodes, returns idxs from nodes
    up to and including num_nodes. I.e.
    [batches, 0:num_nodes + 1]. Note the order is
    sorted by (batches, num_nodes + 1) in ascending order.

    Useful for getting all active nodes in the graph"""
    seq_lens = num_nodes.unsqueeze(-1)
    N = nodes.shape[1]
    N_idx = torch.arange(N, device=nodes.device).unsqueeze(0)
    N_idx = N_idx.expand(seq_lens.shape[0], N_idx.shape[1])
    # include the current node
    N_idx = torch.nonzero(N_idx <= num_nodes.unsqueeze(1))
    assert N_idx.shape[-1] == 2
    batch_idxs = N_idx[:, 0]
    node_idxs = N_idx[:, 1]

    return batch_idxs, node_idxs


@torch.jit.script
def idxs_up_to_num_nodes(
    adj: torch.Tensor, num_nodes: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Given num_nodes, returns idxs from adj
    up to but not including num_nodes. I.e.
    [batches, 0:num_nodes, num_nodes]. Note the order is
    sorted by (batches, num_nodes, 0:num_nodes) in ascending order.

    Useful for getting all actives adj entries in the graph"""
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
