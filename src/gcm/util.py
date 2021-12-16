import torch
import numpy as np
import torch_geometric
from torch_scatter import scatter_max, scatter
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

def sparse_max(x: torch.sparse_coo, dim: int=-1, keepdim=True):
    vals, counts = torch.unique(x._indices(), return_counts=True)
    max_size = counts.max()
    dense = torch.empty(max_size).fill_(1e-20)

def flatten_idx(idx):
    return idx[0] * idx.shape[1] + idx[1]

def unflatten_idx(idx, b):
    b_idx = idx // b
    f_idx = idx % b
    return torch.stack((b_idx, f_idx))


def flatten_idx_n_dim(idx):
    assert idx.ndim == 2
    strides = idx.max(dim=1).values + 1
    #offsets = strides.cumprod(0).flip(0)
    new_idx = torch.zeros(idx.shape[-1], dtype=torch.long, device=idx.device)
    offsets = []

    for i in range(len(strides) - 1):
        offset = strides[i + 1:].prod()
        offsets.append(offset)
        new_idx += offset * idx[i]

    new_idx += idx[-1]

    return new_idx, offsets


def sparse_gumbel_softmax(
    logits: torch.sparse_coo, 
    dim: int,
    tau: float=1, 
    hard: bool=False,
    ) -> torch.sparse_coo:
    gumbels = -torch.empty_like(logits._values()).exponential_().log()
    gumbels = (logits._values() + gumbels) / tau
    gumbels = torch.sparse_coo_tensor(
        indices=logits._indices(),
        values=logits._values(),
        size=logits.shape
    )
    y_soft = torch.sparse.softmax(gumbels, dim=dim)

    if not hard:
        return y_soft

    index = []
    # Want to max across dim, so exclude it during scatter
    scat_dims = torch.cat(
        (torch.arange(dim), torch.arange(dim+1, logits._indices().shape[0]))
    )
    scat_idx = y_soft._indices()[scat_dims]
    flat_scat_idx, offsets = flatten_idx_n_dim(scat_idx)
    maxes, argmax = scatter_max(y_soft._values(), flat_scat_idx)
    # TODO: Sometimes argmax will give us out of bound indices 
    # because dim_size < numel
    # we would use the dim_size arg to scatter, but it crashes :(
    # so instead just mask out invalid entries
    argmax_mask = argmax < y_soft._indices().shape[-1] 
    maxes = maxes[argmax_mask]
    argmax = argmax[argmax_mask]
    index = y_soft._indices()[:, argmax]

    return torch.sparse_coo_tensor(
        indices=index,
        values=maxes,
        size=logits.shape,
        device=logits.device
    )


    """
    for b in y_soft._indices()[0].unique():
        for n in y_soft._indices()[1].unique():
            argmax = y_soft[b,n]._values().max(0)[1]
            index.append(torch.tensor([b, n, argmax]))
        
    index = torch.stack(index).T
    y_hard = torch.sparse_coo_tensor(
        indices=logits._indices(),
        values=torch.ones_like(logits._values()),
        size=logits.shape,
        device=logits.device
    )
    """
    return y_hard - y_soft.detach() + y_soft


    """
    if hard:
        # Straight thru
        nelem = y_soft._indices().shape[-1]
        flat_idx = flatten_idx(y_soft._indices()[:2])
        index = scatter_max(y_soft._values(), flat_idx)[1]
        #y_soft2 = torch.cat(
        #        [torch.zeros(2, y_soft._values().shape[0]), y_soft._values().unsqueeze(0)], dim=0)
        # TODO: dim should be changeable
        #index = scatter_max(y_soft._values(), y_soft._indices()[1])[1]
        #index = scatter_max(y_soft._values().repeat(2,1), y_soft._indices()[:2])[1]
        import pdb; pdb.set_trace()
        #index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret
    """



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


def get_batch_offsets(T: torch.Tensor):
    """Get edge offsets into the flattened edge tensor."""
    batch_ends = T.cumsum(dim=0)
    batch_starts = batch_ends.roll(1)
    batch_starts[0] = 0

    return batch_starts, batch_ends


def flatten_adj(adj, T, taus, B):
    """Flatten a torch.coo_sparse [B, MAX_NODES, MAX_NODES] to [2, NE] and
    adds offsets to avoid collisions.
    This readies a sparse tensor for torch_geometric GNN
    ingestion.

    Returns edges, weights, and corresponding batch ids
    """
    batch_starts, batch_ends = get_batch_offsets(T + taus)


    batch_idx = adj._indices()[0]
    edge_offsets = batch_starts[batch_idx]
    flat_edges = adj._indices()[1:] + edge_offsets
    flat_weights = adj._values()

    if flat_edges.numel() > 0:
        # Make sure idxs are removed alongside edges and weights
        flat_edges, [flat_weights, batch_idx] = torch_geometric.utils.coalesce(
            flat_edges, [flat_weights, batch_idx], reduce="mean"
        )

    return flat_edges, flat_weights, batch_idx
    

def unflatten_adj(edges, weights, batch_idx, T, taus, B, max_edges):
    """Unflatten edges [2,NE], weights: [NE], and batch_idx [NE]
    into a torch.coo_sparse adjacency matrix of [B, NE, NE]"""
    batch_starts, batch_ends = get_batch_offsets(T + taus)

    edge_offsets = batch_starts[batch_idx]
    adj_edge_idx = edges - edge_offsets
    adj_idx = torch.stack([batch_idx, adj_edge_idx[0], adj_edge_idx[1]])
    adj_val = weights
    
    return torch.sparse_coo_tensor(
        indices=adj_idx, values=adj_val, size=(B, max_edges, max_edges)
    )



def pack_hidden(hidden, B, max_edges: int, edge_fill: int=-1, weight_fill: float=1.0):
    return _pack_hidden(*hidden, B, max_edges, edge_fill, weight_fill)

def _pack_hidden(
	nodes: torch.Tensor,
	adj: torch.Tensor,
	T: torch.Tensor,
	B: int,
	max_edges: int,
	edge_fill: int = -1,
	weight_fill: float = 1.0,
):
    """Converts a torch.coo_sparse adj to a ray dense edgelist."""
    batch_idx, source_idx, sink_idx = adj._indices().unbind()
    dense_edges = torch.empty((B, 2, max_edges), device=adj.device, dtype=torch.long).fill_(edge_fill)
    dense_weights = torch.empty((B, 1, max_edges), device=adj.device, dtype=torch.float).fill_(weight_fill)

    # TODO can we vectorize this without a BxNE matrix?
    for b in range(B):
        sparse_b_idx = torch.nonzero(batch_idx == b).reshape(-1)
        dense_b_idx = torch.arange(sparse_b_idx.shape[0])
        dense_edges[b, :, dense_b_idx] = adj._indices()[1:, sparse_b_idx]
        dense_weights[b, 0, dense_b_idx] = adj._values()[sparse_b_idx]

    return nodes, dense_edges, dense_weights, T

def unpack_hidden(hidden, B):
    return _unpack_hidden(*hidden, B) 

def _unpack_hidden(
    nodes: torch.Tensor,
    edges: torch.Tensor,
    weights: torch.Tensor,
    T: torch.Tensor,
    B: torch.Tensor
):
    """Convert a ray dense edgelist into a torch.coo_sparse tensor"""
    # Get indices of valid edge pairs
    batch_idx, edge_idx = (edges[:,0] >= 0).nonzero().T.unbind()
    # Get values of valid edge pairs
    sources = edges[batch_idx, 0, edge_idx]
    sinks = edges[batch_idx, 1, edge_idx]

    adj_idx = torch.stack([batch_idx, sources, sinks])
    weights_filtered = weights[batch_idx, 0, edge_idx]
    #sink_idx = edges[batch_idx, 1, source_idx]
    #adj_idx = torch.stack([batch_idx, source_idx, sink_idx])


    adj = torch.sparse_coo_tensor(
        indices=adj_idx, values=weights_filtered, size=(B, nodes.shape[1], nodes.shape[1])
    )

    return nodes, adj, T



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
