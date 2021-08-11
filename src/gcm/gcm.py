import torch
import torch_geometric
from torch_geometric.data import Data, Batch
from typing import List, Tuple, Union, Any, Dict, Callable
import time
import math


class SparseToDense(torch.nn.Module):
    """Convert from edge_list to adj. """

    def forward(self, x, edge_index, batch_idx, B, N):
        # TODO: Should handle weights
        x = torch_geometric.utils.to_dense_batch(x=x, batch=batch_idx, max_num_nodes=N)[
            0
        ]
        adj = torch_geometric.utils.to_dense_adj(
            edge_index, batch=batch_idx, max_num_nodes=N
        )[0]
        return x, adj


class DenseToSparse(torch.nn.Module):
    """Convert from adj to edge_list while allowing gradients
    to flow through adj.

    x: shape[B, N+k, feat]
    adj: shape[B, N+k, N+k]
    mask: shape[B, N+k]"""

    def forward(self, x, adj, mask=None):
        assert x.dim() == adj.dim() == 3
        B = x.shape[0]
        N = x.shape[1]
        if mask:
            raise NotImplementedError()
            assert mask.shape == (B, N)
            x_mask = mask.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            adj_mask = mask.unsqueeze(-1).expand(-1, -1, adj.shape[-1])
            x = x * x_mask
            adj = adj * adj_mask
            N = mask.shape[1]
        offset, row, col = torch.nonzero(adj > 0).t()
        row += offset * N
        col += offset * N
        edge_index = torch.stack([row, col], dim=0).long()
        x = x.view(B * N, x.shape[-1])
        batch_idx = (
            torch.arange(0, B, device=x.device).view(-1, 1).repeat(1, N).view(-1)
        )

        return x, edge_index, batch_idx

class PositionalEncoding1D(torch.nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super().__init__()
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")
        batch_size, x, orig_ch = tensor.shape
        # Modified so the counting is from t -> 0 instead of 0 -> t
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        emb = torch.zeros((x,self.channels),device=tensor.device).type(tensor.type())
        emb[:,:self.channels] = emb_x

        return emb[None,:,:orig_ch].repeat(batch_size, 1, 1)
    
@torch.jit.script
def overflow(num_nodes: torch.Tensor, N: int):
    return torch.any(num_nodes + 1 > N)


class DenseGCM(torch.nn.Module):
    """Graph Associative Memory"""

    did_warn = False

    def __init__(
        self,
        # Graph neural network, see torch_geometric.nn.Sequential
        # for some examples
        gnn: torch.nn.Module,
        # Preprocessor for each feat vec before it's placed in graph
        preprocessor: torch.nn.Module = None,
        # an edge selector from gcm.edge_selectors
        # you can chain multiple selectors together using
        # torch_geometric.nn.Sequential
        edge_selectors: torch.nn.Module = None,
        # Maximum number of nodes in the graph
        graph_size: int = 128,
        # Whether the gnn outputs graph_size nodes or uses global pooling
        pooled: bool = False,
        # Whether to add sin/cos positional encoding like in transformer
        # to the nodes
        # Creates an ordering in the graph
        positional_encoding: bool = False,
    ):
        super().__init__()

        self.preprocessor = preprocessor
        self.gnn = gnn
        self.graph_size = graph_size
        self.edge_selectors = edge_selectors
        self.pooled = pooled
        if positional_encoding:
            self.positional_encoder = PositionalEncoding1D(math.ceil(self.graph_size / 2) * 2)
        else:
            self.positional_encoder = None

    def forward(
        self, x, hidden: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add a memory x to the graph, and query the memory for it.
        B = batch size
        N = maximum graph size
        Inputs:
            x: [B,feat]
            hidden: (
                nodes: [B,N,feats]
                adj: [B,N,N]
                weights: [B,N,N] or None
                number_of_nodes_in_graph: [B]
            )
        Outputs:
            m(x): [B,feat]
            hidden: (
                nodes: [B,N,feats]
                adj: [B,N,N]
                weights: [B,N,N] or None
                number_of_nodes_in_graph: [B]
            )
        """
        nodes, adj, weights, num_nodes = hidden

        assert x.dtype == torch.float32
        assert nodes.dtype == torch.float
        # if self.gnn.sparse:
        #    assert adj.dtype == torch.long
        assert weights.dtype == torch.float
        assert num_nodes.dtype == torch.long
        assert num_nodes.dim() == 1

        N = nodes.shape[1]
        B = x.shape[0]
        B_idx = torch.arange(B)

        assert (
            N == adj.shape[1] == adj.shape[2]
        ), "N must be equal for adj mat and node mat"

        nodes = nodes.clone()
        if overflow(num_nodes, N):
            if not self.did_warn:
                print("Overflow detected, wrapping around. Will not warn again")
                self.did_warn = True
            adj = adj.clone()
            weights = weights.clone()
            nodes, adj, weights, num_nodes = self.wrap_overflow(
                nodes, adj, weights, num_nodes
            )
        # Add new nodes to the current graph
        # starting at num_nodes
        nodes[B_idx, num_nodes[B_idx]] = x[B_idx]

        # Do NOT add self edges or they will be counted twice using
        # GraphConv
        
        # Adj and weights must be cloned as
        # edge selectors will modify them in-place
        adj = adj.clone()
        weights = weights.clone()
        if self.edge_selectors:
            adj, weights = self.edge_selectors(
                nodes, adj, weights, num_nodes, B
            )

        # Thru network
        if self.preprocessor:
            nodes_in = self.preprocessor(nodes)
        else:
            nodes_in = nodes

        if self.positional_encoder: 
            # todo
            encodings = self.positional_encoder(nodes)
            nodes = nodes.clone()
            for b in range(B):
                nodes[b, :num_nodes[b] + 1] = nodes[b, :num_nodes[b] + 1] + encodings[b, :num_nodes[b] + 1]

        node_feats = self.gnn(nodes_in, adj, weights, B, N)
        if self.pooled:
            # If pooled, we expect only a single output node
            mx = node_feats
        else:
            # Otherwise extract the hidden repr at the current node
            mx = node_feats[B_idx, num_nodes[B_idx]]

        assert torch.all(
            torch.isfinite(mx)
        ), "Got NaN in returned memory, try using tanh activation"

        num_nodes = num_nodes + 1
        return mx, (nodes, adj, weights, num_nodes)

    def wrap_overflow(self, nodes, adj, weights, num_nodes):
        """Call this when the node/adj matrices are full. Deletes the zeroth element
        of the matrices and shifts all the elements up by one, producing a free row
        at the end. You will likely want to call .clone() on the arguments that require
        gradient computation.

        Returns new nodes, adj, weights, and num_nodes matrices"""
        N = nodes.shape[1]
        overflow_mask = num_nodes + 1 > N
        # Shift node matrix into the past
        # by one and forget the zeroth node
        overflowing_batches = overflow_mask.nonzero().squeeze()
        #nodes = nodes.clone()
        #adj = adj.clone()
        # Zero entries before shifting
        nodes[overflowing_batches, 0] = 0
        adj[overflowing_batches, 0, :] = 0
        adj[overflowing_batches, :, 0] = 0
        # Roll newly zeroed zeroth entry to final entry
        nodes[overflowing_batches] = torch.roll(nodes[overflowing_batches], -1, -2)
        adj[overflowing_batches] = torch.roll(
            adj[overflowing_batches], (-1, -1), (-1, -2)
        )
        if weights.numel() != 0:
            #weights = weights.clone()
            weights[overflowing_batches, 0, :] = 0
            weights[overflowing_batches, :, 0] = 0
            weights[overflowing_batches] = torch.roll(
                weights[overflowing_batches], (-1, -1), (-1, -2)
            )

        num_nodes[overflow_mask] = num_nodes[overflow_mask] - 1
        return nodes, adj, weights, num_nodes
