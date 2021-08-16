import torch
import itertools
from typing import Dict, Tuple, List
import gcm.util


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        #return torch.nn.functional.hardtanh(grad_output)
        return grad_output

class StraightThroughEstimator(torch.nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x


@torch.jit.script
def diff_or(tensors: List[torch.Tensor]):
    """Differentiable OR operation bewteen n-tuple of tensors
    Input: List[tensors in {0,1}]
    Output: tensor in {0,1}"""
    # This nice form is actually slower than the matrix mult form
    return 1 - (1 - torch.stack(tensors, dim=0)).prod(dim=0)



@torch.jit.script
def diff_or2(tensors: List[torch.Tensor]):
    """Differentiable OR operation bewteen n-tuple of tensors
    Input: List[tensors in {0,1}]
    Output: tensor in {0,1}"""
    res = torch.zeros_like(tensors[0])
    for t in tensors:
        tmp = res.clone()
        res = tmp + t - tmp * t
    return res


@torch.jit.script
def up_to_num_nodes_idxs(
    adj: torch.Tensor, num_nodes: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Given num_nodes, returns idxs from adj
    up to but not including num_nodes. I.e.
    [batches, 0:num_nodes, num_nodes]. Note the order is
    sorted by (batches, num_nodes, 0:num_nodes) in ascending order"""
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


def sample_hard(
        adj: torch.Tensor, 
        weights: torch.Tensor, 
        num_nodes: torch.Tensor, 
        num_edges: int = 2
    ) -> torch.Tensor:
    """
    Input: Tensor of logits (weights)
    Output: Tensor in {0, 1} (adjacency)

    Converts logits x into a softmax distribution over all possible edges
    (i, j) for a node i. Samples from said distribution num_edges times
    using the gumbel-softmax trick. The entire process is differentiable.

    This process is used to fill out the adjacency matrix, while enforcing sparsity
    by upper-bounding each node to num_edges edges. Because the diagonal is set to
    zero, the lower bound of edges per node is zero (not one).
    """

    new_adj = adj.clone()
    B = weights.shape[0]
    for b in range(B):
        weight = weights[b]
        # Shrink weights to existing nodes
        valid_weight = weight.narrow(0, 0, num_nodes[b] + 1).narrow(1, 0, num_nodes[b] + 1)

        for i in range(num_edges):
            # Each adj will only hold one edge per node
            # run thru i times to collect i edges
            ith_adj = torch.nn.functional.gumbel_softmax(
                valid_weight, hard=True, dim=1
            )

            ith_adj_padded = torch.zeros_like(weights[b])
            ith_adj_padded[:num_nodes[b] + 1, :num_nodes[b] + 1] = ith_adj
            # bitwise union/or but differentiable
            new_adj[b] = diff_or2((ith_adj_padded, new_adj[b].clone()))
    # Zero self edges or they are counted twice
    new_adj.diagonal(dim1=-1, dim2=-2)[:] = 0
    return new_adj


class BernoulliEdge(torch.nn.Module):
    """Add temporal bidirectional back edge, but only if we have >1 nodes
    E.g., node_{t} <-> node_{t-1}"""

    def __init__(
        self,
        input_size: int = 0,
        model: torch.nn.Sequential = None,
        backward_edges: bool = False,
        desired_num_edges: int = 5,
        probabilistic: bool = True,
        # gradient_scale: float = 0.5
    ):
        super().__init__()
        self.backward_edges = backward_edges
        self.probabilistic = probabilistic
        self.desired_num_edges = desired_num_edges
        self.ste = StraightThroughEstimator()
        assert input_size or model, "Must specify either input_size or model"
        if model:
            self.edge_network = model
        else:
            # This MUST be done here
            # if done in forward model does not learn...
            self.edge_network = self.build_edge_network(input_size)

    def sample_random_var(self, p: torch.Tensor) -> torch.Tensor:
        """Given a probability [0,1] p, return a backprop-capable random sample"""
        e = torch.rand(p.shape, device=p.device)
        return torch.sigmoid(
            torch.log(e) - torch.log(1 - e) + torch.log(p) - torch.log(1 - p)
        )

    def build_edge_network(self, input_size: int) -> torch.nn.Sequential:
        """Builds a network to predict edges.
        Network input: (i || j)
        Network output: p(edge) in [0,1]
        """
        return torch.nn.Sequential(
            torch.nn.Linear(2 * input_size, input_size),
            torch.nn.Tanh(),
            torch.nn.Linear(input_size, 1),
        )

    def compute_logits2(
        self,
        nodes: torch.Tensor,
        num_nodes: torch.Tensor,
        weights: torch.Tensor,
        B: int,
    ):
        """Computes edge probability between current node and all other nodes.
        Returns a modified copy of the weight matrix containing edge probs"""
        # No edges for a single node
        if torch.max(num_nodes) < 1:
            return weights

        b_idxs, past_idxs, curr_idx = gcm.util.idxs_up_to_num_nodes(weights, num_nodes)
        # curr_idx > past_idxs
        # flows from past_idxs to j
        # so [j, past_idxs]
        curr_nodes = nodes[b_idxs, curr_idx]
        past_nodes = nodes[b_idxs, past_idxs]

        net_in = torch.cat((curr_nodes, past_nodes), dim=-1)
        log_probs = self.edge_network(net_in).squeeze()
        # TODO: weights[:,0] is not populated, why?
        weights[b_idxs, curr_idx, past_idxs] = log_probs

        if self.backward_edges:
            net_in = torch.cat((past_nodes, curr_nodes), dim=-1)
            weights[b_idxs, past_idxs, curr_idx] = log_probs

        return weights

    def forward(self, nodes, adj, weights, num_nodes, B):
        """A(i,j) = Ber[phi(i || j), e]

        Modify the nodes/adj_mats/state in-place by reference. Return value
        is not used.
        """

        # a(b,i,j) = gumbel_softmax(phi(n(b,i) || n(b,j))) for i, j < num_nodes
        # First run
        if self.edge_network[0].weight.device != nodes.device:
            self.edge_network = self.edge_network.to(nodes.device)

        # Weights serve as probabilities that we sample from
        weights = self.compute_logits2(nodes, num_nodes, weights, B)
        # Combine new sampled adj with prev adj
        new_adj = sample_hard(adj, weights, num_nodes, self.desired_num_edges)

        return new_adj, weights
