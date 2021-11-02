import torch
import torch_geometric
from gcm import util
from typing import Union, Tuple, List

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
patch_typeguard()


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
        # Whether to pad edges so they are always a constant size
        pad_edges: bool = True,
        # Max edges per batch, only used if pad_edges is set
        max_edges: int = int(1e5),
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
        self.max_edges = max_edges
        self.pad_edges = pad_edges
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

        assert x.dim() == 3
        B, _, feats = x.shape
        edges = torch.zeros(B, 2, 0, device=x.device, dtype=torch.long)
        nodes = torch.zeros(B, self.graph_size, feats, device=x.device)
        weights = torch.zeros(B, 1, 0, device=x.device)
        T = torch.zeros(B, dtype=torch.long, device=x.device)

        return nodes, edges, weights, T

    @typechecked
    def forward(
        self, 
        x: TensorType["B","t","feat"],
        taus: TensorType["B"],             # sequence_lengths
        hidden: Union[
            None, 
            Tuple[
                TensorType["B", "N", "feats"], # Nodes
                TensorType["B", 2, "E"],       # Edges
                TensorType["B", 1, "E"],       # Weights
                TensorType["B"],               # T
            ]
        ]
    ) -> Tuple[
        torch.Tensor, 
            Tuple[
                TensorType["B", "N", "feats"],  # Nodes
                TensorType["B", 2, "NE"],       # Edges
                TensorType["B", 1, "NE"],       # Weights
                TensorType["B"]                 # T
            ]
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
        if hidden == None:
            hidden = self.get_initial_hidden_state(x)

        nodes, edges, weights, T = hidden

        N = nodes.shape[1]
        B = x.shape[0]
        # Batch and time idxs for nodes we intend to add
        B_idxs, tau_idxs = util.get_new_node_idxs(T, taus, B)
        dense_B_idxs, dense_tau_idxs = util.get_nonpadded_idxs(T, taus, B)

        nodes = nodes.clone()
        # Add new nodes to the current graph
        # TODO CRITICAL: Ensure edges are ALWAYS flowing past to future
        # or this shit breaks
        # TODO: 
        if tau_idxs.max() >= N:
            raise Exception('Overflow')

        nodes[B_idxs, tau_idxs] = x[dense_B_idxs, dense_tau_idxs]

        # We do not want to modify graph nodes in the GCM
        # Do all mutation operations on dirty_nodes, 
        # then use clean nodes in the graph state
        dirty_nodes = nodes.clone()
        if self.edge_selectors:
            edges, weights = self.edge_selectors(
                dirty_nodes, edges, weights, T, taus
            )

        # Thru network
        if self.preprocessor:
            dirty_nodes = self.preprocessor(dirty_nodes)
        if self.positional_encoder:
            dirty_nodes = self.positional_encoder(dirty_nodes)
        if self.aux_edge_selectors:
            edges, weights = self.edge_selectors(
                dirty_nodes, edges, weights, T, taus, B
            )

        # We need to convert to GNN input format
        # it expects batch=[Batch], x=[Batch,feats], edge=[2, ?}
        '''
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
        node_feats = self.gnn(batch.x, batch.edge_index)
        '''
        flat_nodes, flat_edges, flat_weights, output_node_idxs = util.to_batch(dirty_nodes, edges, weights, T, taus, B)
        node_feats = self.gnn(flat_nodes, flat_edges, flat_weights)
        # Extract the hidden repr at the new nodes
        # Each mx is variable in temporal dim, so return 2D tensor of [B*tau, feat]
        mx = node_feats[output_node_idxs]

        assert torch.all(
            torch.isfinite(mx)
        ), "Got NaN in returned memory, try using tanh activation"

        # Fix for ray, edges must be constant size
        if self.pad_edges:
            unpadded_edges = edges
            edges = torch.ones(edges.shape[0], edges.shape[1], self.max_edges) * -1
            edges[:,:, :edges.shape[-1]] = edges

        T = T + taus
        return mx, (nodes, edges, weights, T)

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
