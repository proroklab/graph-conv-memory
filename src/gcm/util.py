import torch
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


    


@torch.jit.script
def diff_or(tensors: List[torch.Tensor]):
    """Differentiable OR operation bewteen n-tuple of tensors
    Input: List[tensors in {0,1}]
    Output: tensor in {0,1}"""
    raise NotImplementedError("This seems to dilute gradients, dont use it")
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
    raise NotImplementedError("This seems to dilute gradients, dont use it")
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
