import torch
import torch_geometric
from torch_geometric.data import Data, Batch
from typing import List, Tuple, Union, Any, Dict, Callable
import time
import math
import gcm.util


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


class RelativePositionalEncoding(torch.nn.Module):
    def __init__(self, max_len: int = 5000):
        super().__init__()
        self.max_len = max_len

    def run_once(self, nodes: torch.Tensor) -> None:
        # Dim must be even
        d_model = math.ceil(nodes.shape[-1] / 2) * 2
        position = torch.arange(self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(self.max_len, d_model, device=nodes.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, nodes: torch.Tensor, num_nodes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
            num_nodes: Tensor
        """
        if not hasattr(self, "pe"):
            self.run_once(nodes)
        
        B = nodes.shape[0]
        for b in range(B):
            center = num_nodes[b]
            pe = self.pe.roll(int(center), 0)
            nodes[b, :center + 1] = (
                nodes[b, :center + 1] + 
                pe[:center + 1,:nodes.shape[-1]]
            )
        return nodes



class PositionalEncoding(torch.nn.Module):
    """Embed positional encoding into the graph. Ensures we do not
    encode future nodes (node_idx > num_nodes)"""

    def __init__(self, max_len: int = 5000, mode="add", cat_dim: int=8):
        super().__init__()
        self.max_len = max_len
        self.mode = mode
        self.cat_dim = cat_dim
        assert mode in ["add", "cat"]

    def run_once(self, x: torch.Tensor) -> None:
        # Dim must be even
        d_model = math.ceil(x.shape[-1] / 2) * 2
        position = torch.arange(self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(self.max_len, d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        if self.mode == "cat":
            self.reproject = torch.nn.Linear(x.shape[-1], x.shape[-1] - self.cat_dim, device=x.device)

    def forward(self, x: torch.Tensor, num_nodes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
            num_nodes: Tensor
        """
        if not hasattr(self, "pe"):
            self.run_once(x)
        
        b_idxs, n_idxs = gcm.util.idxs_up_to_including_num_nodes(x, num_nodes)
        if self.mode == "add":
            x[b_idxs, n_idxs] = x[b_idxs, n_idxs] + self.pe[n_idxs, :x.shape[-1]]
        elif self.mode == "cat":
            x_reproj = self.reproject(x[b_idxs, n_idxs]).reshape(len(b_idxs), x.shape[-1] - self.cat_dim)
            # positional encoding
            x = x.clone()
            x[b_idxs, n_idxs, :self.cat_dim] = self.pe[n_idxs, :self.cat_dim]
            # Reprojected feature assignment
            x[b_idxs, n_idxs, self.cat_dim:] = x_reproj
        else:
            raise NotImplementedError("Invalid mode")
        return x

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
        # Auxiliary edge selectors are called
        # after the positional encoding and reprojection
        # this should only be used for non-human (learned) priors
        aux_edge_selectors: torch.nn.Module = None,
        # Maximum number of nodes in the graph
        graph_size: int = 128,
        # Whether the gnn outputs graph_size nodes or uses global pooling
        pooled: bool = False,
        # Whether to add sin/cos positional encoding like in transformer
        # to the nodes
        # Creates an ordering in the graph
        positional_encoder: torch.nn.Module = None,
        # Whether to use edge_weights
        # only required if using learned edges
        edge_weights: bool = False,
    ):
        super().__init__()

        self.preprocessor = preprocessor
        self.gnn = gnn
        self.graph_size = graph_size
        self.edge_selectors = edge_selectors
        self.aux_edge_selectors = aux_edge_selectors
        self.pooled = pooled
        self.edge_weights = edge_weights
        self.positional_encoder = positional_encoder

    def get_initial_hidden_state(self, x):
        """Given a dummy x of shape [B, feats], construct
        the hidden state for the base case (adj matrix, weights, etc)"""
        """Returns the initial hidden state h (e.g. h, output = gcm(input, h)),
        for a given batch size (B). Feats denotes the feature size (# dims of each
        node in the graph)."""

        assert x.dim() == 2
        B, feats = x.shape
        edges = torch.zeros(B, self.graph_size, self.graph_size, device=x.device)
        nodes = torch.zeros(B, self.graph_size, feats, device=x.device)
        if self.edge_weights:
            weights = torch.zeros(B, self.graph_size, self.graph_size, device=x.device)
        else:
            weights = torch.zeros(0, device=x.device)
        num_nodes = torch.zeros(B, dtype=torch.long, device=x.device)

        return nodes, edges, weights, num_nodes

    def forward(
        self, x, hidden: Union[None, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
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
        # Base case
        if hidden == None:
            hidden = self.get_initial_hidden_state(x)

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
        # We do not want to modify graph nodes in the GCM
        # Do all mutation operations on dirty_nodes, 
        # then use clean nodes in the graph state
        dirty_nodes = nodes.clone()
        # Do NOT add self edges or they will be counted twice using
        # GraphConv
        
        # Adj and weights must be cloned as
        # edge selectors will modify them in-place
        if self.edge_selectors:
            adj, weights = self.edge_selectors(
                dirty_nodes, adj.clone(), weights.clone(), num_nodes, B
            )

        # Thru network
        if self.preprocessor:
            dirty_nodes = self.preprocessor(dirty_nodes)
        #if self.positional_encoder:
        #   dirty_nodes = self.positional_encoder(dirty_nodes, num_nodes)
        if self.aux_edge_selectors:
            if self.positional_encoder:
                adj, weights = self.aux_edge_selectors(
                    self.positional_encoder(dirty_nodes, num_nodes), adj.clone(), weights.clone(), num_nodes, B
                )
            else:
                adj, weights = self.aux_edge_selectors(
                    dirty_nodes, adj.clone(), weights.clone(), num_nodes, B
                )

        node_feats = self.gnn(dirty_nodes, adj, weights, B, N)
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
