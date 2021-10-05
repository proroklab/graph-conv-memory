import torch
import torch_geometric
import sparsemax
from typing import Tuple, List


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
        #return torch.nn.functional.hardtanh(grad_output)

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


class SparsegenLin(torch.nn.Module):
    def __init__(self, lam, normalized=True):
        super().__init__()
        self.lam = lam
        self.normalized = normalized

    def forward(self, z):
        bs = z.data.size()[0]
        dim = z.data.size()[1]
        # z = input.sub(torch.mean(input,dim=1).repeat(1,dim))
        dtype = torch.FloatTensor
        #z = input.type(dtype)

        #sort z
        z_sorted = torch.sort(z, descending=True)[0]

        #calculate k(z)
        z_cumsum = torch.cumsum(z_sorted, dim=1)
        k = torch.autograd.Variable(torch.arange(1, dim + 1, device=z.device).unsqueeze(0).repeat(bs,1))
        z_check = torch.gt(1 - self.lam + k * z_sorted, z_cumsum)

        # because the z_check vector is always [1,1,...1,0,0,...0] finding the
        # (index + 1) of the last `1` is the same as just summing the number of 1.
        k_z = torch.sum(z_check.float(), 1)

        #calculate tau(z)
        tausum = torch.sum(z_check.float() * z_sorted, 1)
        tau_z = (tausum - 1 + self.lam) / k_z
        prob = z.sub(tau_z.view(bs,1).repeat(1,dim)).clamp(min=0)
        if self.normalized:
               prob /= (1-self.lam)
        return prob


class Spardgen(torch.nn.Module):
    """A hard version of sparsegen-lin"""
    def __init__(self, dim=-1, lam=0.75):
        super().__init__()
        self.dim = dim
        self.sm = SparsegenLin(lam)
    
    def forward(self, x):
        # Only takes up to 2ds so reshape
        x_in = x.reshape(-1, x.shape[-1])
        # Straight through.
        y_soft = self.sm(x_in).reshape(x.shape)
        y_hard = (y_soft != 0).float()
        return y_hard - y_soft.detach() + y_soft


    

def get_new_node_idxs(T, taus, B):
    """Given T and tau tensors, return indices matching batches to taus.
    These tell us which elements in the node matrix we have just added
    during this iteration, and organize them by batch. 

    E.g. 
    g_idxs = torch.where(B_idxs == 0)
    zeroth_graph_new_nodes = nodes[B_idxs[g_idxs], tau_idxs[g_idxs]] 
    """
    # TODO: batch this using b_idx and cumsum
    B_idxs = torch.cat([torch.ones(taus[b], device=T.device, dtype=torch.long) * b for b in range(B)])
    tau_idxs = torch.cat([torch.arange(T[b], T[b] + taus[b], device=T.device) for b in range(B)])
    return B_idxs, tau_idxs

def get_valid_node_idxs(T, taus, B):
    """Given T and tau tensors, return indices matching batches to taus.
    These tell us which elements in the node matrix are valid for convolution,
    and organize them by batch. 

    E.g. 
    g_idxs = torch.where(B_idxs == 0)
    zeroth_graph_all_nodes = nodes[B_idxs[g_idxs], tau_idxs[g_idxs]] 
    """
    # TODO: batch this using b_idx and cumsum
    B_idxs = torch.cat([torch.ones(T[b] + taus[b], device=T.device, dtype=torch.long) * b for b in range(B)])
    tau_idxs = torch.cat([torch.arange(0, T[b] + taus[b], device=T.device) for b in range(B)])
    return B_idxs, tau_idxs



def to_batch(nodes, edges, weights, T, taus, B):
    """Squeeze node, edge, and weight batch dimensions into a single
    huge graph. Also deletes non-valid edge pairs"""
    b_idx = torch.arange(B, device=nodes.device)
    # Initial offset is zero, not T + tau, roll into place
    batch_offsets = (T + taus).cumsum(dim=0).roll(1,0)
    batch_offsets[0] = 0

    # Flatten edges
    num_flat_edges = edges[b_idx].shape[-1]
    edge_offsets = batch_offsets.unsqueeze(-1).unsqueeze(-1).expand(-1,2,num_flat_edges)
    offset_edges = edges + edge_offsets
    # Filter invalid edges (those that were < 0 originally)
    # Swap dims (B,2,NE) => (2,B,NE)
    mask = (offset_edges < edge_offsets).permute(1,0,2)
    stacked_mask = (mask[0] & mask[1]).unsqueeze(0).expand(2,-1,-1)
    # Careful, mask select will automatically flatten
    # so do it last, this squeezes from from (2,B,NE) => (2,B*NE)
    flat_edges = edges.permute(1,0,2).masked_select(stacked_mask).reshape(2,-1)
    # Do the same with weights, which will be of size E
    flat_weights = weights.permute(1,0,2).masked_select(stacked_mask[0]).flatten()
    # Finally, remove duplicate edges and weights
    flat_edges, flat_weights = torch_geometric.utils.coalesce(flat_edges, flat_weights)

    # Flatten nodes
    B_idxs, tau_idxs = get_valid_node_idxs(T, taus, B)
    flat_nodes = nodes[B_idxs, tau_idxs]
    # Extracting belief requires batch-tau indices (newly inserted nodes)
    # return these too
    # Flat nodes are ordered B,:T+tau (all valid nodes)
    # We want B,T:T+tau (new nodes), which is batch_offsets:batch_offsets + tau
    output_node_idxs = torch.cat(
        [
            torch.arange(
                batch_offsets[b], batch_offsets[b] + taus[b], device=nodes.device
            ) for b in range(B)
        ]
    )

    return flat_nodes, flat_edges, output_node_idxs

    datalist = []
    for b in range(B):
        data_x = dirty_nodes[b, :T[b] + taus[b]]
        # Get only valid edges (-1 signifies invalid edge)
        mask = edges[b] > -1
        # Delete nonvalid edge pairs
        data_edge = edges[b, :, mask[0] & mask[1]]
        #data_edge = edges[b][edges[b] > -1].reshape(2,-1) #< T[b] + tau]
        datalist.append(torch_geometric.data.Data(x=data_x, edge_index=data_edge))
    batch = torch_geometric.data.Batch.from_data_list(datalist)

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
