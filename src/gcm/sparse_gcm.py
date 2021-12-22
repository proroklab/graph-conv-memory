import torch
import torch_geometric
from gcm import util
from typing import Union, Tuple, List

from torchtyping import TensorType, patch_typeguard  # type: ignore
from typeguard import typechecked

#patch_typeguard()


class SparseGCM(torch.nn.Module):
    """Graph Associative Memory using sparse-graph representations"""

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
        # Optional maximum hops in the graph. If set,
        # we will extract the k-hop subgraph for better efficiency.
        # If set, this should be equal to the number of convolution
        # layers in the GNN
        max_hops: Union[int, None] = None,
        # Whether to add sin/cos positional encoding like in transformer
        # to the nodes
        # Creates an ordering in the graph
        positional_encoder: torch.nn.Module = None,
    ):
        super().__init__()

        self.preprocessor = preprocessor
        self.gnn = gnn
        self.graph_size = graph_size
        self.edge_selectors = edge_selectors
        self.aux_edge_selectors = aux_edge_selectors
        self.positional_encoder = positional_encoder
        self.max_hops = max_hops
        self.ste = util.StraightThroughEstimator()

    def get_initial_hidden_state(self, x):
        """Given a dummy x of shape [B, feats], construct
        the hidden state for the base case (adj matrix, weights, etc)"""
        """Returns the initial hidden state h (e.g. h, output = gcm(input, h)),
        for a given batch size (B). Feats denotes the feature size (# dims of each
        node in the graph)."""

        assert x.dim() == 3
        B, _, feats = x.shape
        #edges = torch.zeros(2, 0, device=x.device, dtype=torch.long)
        nodes = torch.zeros(B, self.graph_size, feats, device=x.device)
        #weights = torch.zeros(0, device=x.device)
        adj = torch.zeros((B, self.graph_size, self.graph_size), device=x.device, layout=torch.sparse_coo)
        T = torch.zeros(B, dtype=torch.long, device=x.device)

        return nodes, adj, T

    @typechecked
    def forward(
        self,
        x: TensorType["B", "t", "feat", float],  # type: ignore # noqa: F821 #input observations
        taus: TensorType["B", int],  # type: ignore # noqa: F821 #sequence_lengths
        hidden: Union[  # type: ignore # noqa: F821
            None,
            Tuple[
                TensorType["B", "N", "feats", float],  # noqa: F821 # Nodes
                TensorType["B", "MAX_E", "MAX_E", float, torch.sparse_coo],   # noqa: F821 # Sparse adj
                TensorType["B", int],  # noqa: F821 # T
            ],
        ],
    ) -> Tuple[  # type: ignore
        torch.Tensor,
        Tuple[
            TensorType["B", "N", "feats", float],  # noqa: F821 # Nodes
            TensorType["B", "MAX_E", "MAX_E", float, torch.sparse_coo],   # noqa: F821 # Sparse adj
            TensorType["B", int],  # noqa: F821 # T
        ],
    ]:
        """Add a memory x with temporal size tau to the graph, and query the memory for it.
        B = batch size
        N = maximum graph size
        T = number of timesteps in graph before input
        taus = number of timesteps in each input batch
        t = the zero-padded time dimension (i.e. max(taus))
        E = number of edge pairs
        """
        # Base case
        if hidden is None:
            hidden = self.get_initial_hidden_state(x)

        nodes, adj, T = hidden

        N = nodes.shape[1]
        B = x.shape[0]

        # Batch and time idxs for nodes we intend to add
        B_idxs, tau_idxs = util.get_new_node_idxs(T, taus, B)
        dense_B_idxs, dense_tau_idxs = util.get_nonpadded_idxs(T, taus, B)

        nodes = nodes.clone()
        # Add new nodes to the current graph
        #assert torch.all(adj._indices()[1] < adj._indices()[2])
        # TODO: Wrap around instead of terminating
        if tau_idxs.max() >= N:
            raise Exception("Overflow")

        nodes[B_idxs, tau_idxs] = x[dense_B_idxs, dense_tau_idxs]

        # We do not want to modify graph nodes in the GCM
        # Do all mutation operations on dirty_nodes,
        # then use clean nodes in the graph state
        dirty_nodes = nodes.clone()

        if self.edge_selectors:
            new_adj = self.edge_selectors(dirty_nodes, T, taus, B)
            new_idx = torch.cat([adj._indices(), new_adj._indices()], dim=-1) 
            new_val = torch.cat([adj._values(), new_adj._values()], dim=-1)
            adj = torch.sparse_coo_tensor(
                indices=new_idx, 
                values=new_val, 
                size=adj.shape
            )


        # Thru network
        if self.preprocessor:
            dirty_nodes = self.preprocessor(dirty_nodes)
        if self.positional_encoder:
            dirty_nodes = self.positional_encoder(dirty_nodes, T + taus)
        if self.aux_edge_selectors:
            new_adj = self.aux_edge_selectors(dirty_nodes, T, taus, B)
            new_idx = torch.cat([adj._indices(), new_adj._indices()], dim=-1) 
            new_val = torch.cat([adj._values(), new_adj._values()], dim=-1)
            adj = torch.sparse_coo_tensor(indices=new_idx, values=new_val, size=adj.shape)

        # Remove duplicates from edge selectors
        # and set all weights to 1.0 without cancelling out gradients
        # from logits.
        # For some reason, torch_geometric coalesces incorrectly here
        adj = adj.coalesce()
        adj = torch.sparse_coo_tensor(
            indices=adj.indices(),
            values=adj.values() / adj.values().detach(),
            size=adj.shape
        )
        # Convert to GNN input format
        flat_nodes, output_node_idxs = util.flatten_nodes(dirty_nodes, T, taus, B)
        edges, weights, edge_batch = util.flatten_adj(adj, T, taus, B)
        # Our adj matrix is sink -> source, but torch_geometric
        # expects edgelist as source -> sink, so flip
        edges = torch.flip(edges, (0,))
        assert torch.all(edges[0] < edges[1]), "Causality violated"
        if edges.numel() > 0:
            edges, weights = torch_geometric.utils.coalesce(
                edges, weights, reduce="mean"
            )
        if self.max_hops is None:
            # Convolve over entire graph
            node_feats = self.gnn(flat_nodes, edges, weights)
            # Extract the hidden repr at the new nodes
            # Each mx is variable in temporal dim, so return 2D tensor of [B*tau, feat]
            mx = node_feats[output_node_idxs]
        else:
            # Convolve over subgraph (more efficient) induced by the
            # target nodes (taus)
            # i.e. ignore nodes/edges that are not connected
            # to the tau (input) nodes
            (
                subnodes,
                subedges,
                node_map,
                edge_mask,
            )  = torch_geometric.utils.k_hop_subgraph(
                output_node_idxs, 
                self.max_hops, 
                edges, 
                relabel_nodes=True,
                num_nodes=(T + taus).sum()
            )
            mx = self.gnn(flat_nodes[subnodes], subedges, weights[edge_mask])[node_map]

        assert torch.all(
            torch.isfinite(mx)
        ), "Got NaN in returned memory, try using tanh activation"

        # Input obs were dense and padded, so output should be dense and padded
        dense_B_idxs, dense_tau_idxs = util.get_nonpadded_idxs(T, taus, B)
        mx_dense = torch.zeros((*x.shape[:-1], mx.shape[-1]), device=x.device)
        mx_dense[dense_B_idxs, dense_tau_idxs] = mx

        T = T + taus

        return mx_dense, (nodes, adj, T)
