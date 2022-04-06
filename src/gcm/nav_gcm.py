import torch
import torch_geometric


# TODO: We need to make this a for loop over temporal dim
# causal edges do not allow loop closures
# memory should be okay because we discard edges
# grads will stack and take up lots of memory, but we can set small seq_lens
# to reduce memory pressure


class NavGCM(torch.nn.Module):
    """GCM tailored specifically to the navigation domain. This allows us
    to use priors to speed up and simplify GCM"""
    def __init__(
        self, 
        gnn,
        pool=False,
        max_verts=128,
        edge_method="radius",
        k=16,
        r=1.0,
        causal=True,
        disjoint_edges=False,
    ):
        super().__init__()
        self.k = k
        self.r = r
        self.gnn = gnn
        self.max_verts = max_verts
        self.pool = pool
        assert edge_method in ["knn", "radius"]
        assert edge_method != "knn", "KNN does not train/infer correctly"
        self.edge_method = edge_method
        self.causal = causal
        self.disjoint_edges = disjoint_edges

    def make_idx(self, T, taus):
        """Returns batch and time idxs marking
        all valid (non-padded) elements in the vert matrix"""
        batch = torch.repeat_interleave(
            torch.arange(T.shape[0], device=T.device),
            T + taus,
        )
        time = torch.cat([
            torch.arange(T[b] + taus[b], device=T.device) 
            for b in range(T.shape[0])
        ], dim=-1)
        return batch, time

    def make_new_idx(
        self, T, taus
    ):
        """Returns batch and time idxs marking
        all NEW elements in the vert matrix"""
        batch = torch.repeat_interleave(
            torch.arange(T.shape[0], device=T.device),
            taus,
        )
        time = torch.cat([
            torch.arange(T[b], T[b] + taus[b], device=T.device) 
            for b in range(T.shape[0])
        ], dim=-1)
        return batch, time

    def make_output_idx(
        self, taus
    ):
        """Like make_new_idx, returns the batch and
        time idxs marking all NEW elements. Unlike make_new_idx
        the indices correspond to locations in the padded OUTPUT
        matrix rather than the full vertex matrix"""
        batch = torch.repeat_interleave(
            torch.arange(taus.shape[0], device=taus.device),
            taus,
        )
        time = torch.cat([
            torch.arange(taus[b], device=taus.device)
            for b in range(taus.shape[0])
        ], dim=-1)
        return batch, time

    def make_flat_new_idx(
        self, T, taus
    ):
        """Return index of all new elements, like make_new_idx.
        However, rather than [B, T] indexing, this returns [B * T]
        indicies to extract new nodes from the GNN output"""
        cs = (T + taus).cumsum(0)
        return torch.cat([
            torch.arange(cs[b] - taus[b], cs[b], device=T.device)
            for b in range(T.shape[0])
        ], dim=-1)

    def knn_edges(
        self, x, pos, rot
    ):
        # TODO: Getting future edges here...
        edges = torch_geometric.nn.knn_graph(
            x=pos,
            k=self.k,
            batch=self.idx[0]
        )
        return edges

    def radius_edges(
        self, x, pos, rot
    ):
        # TODO: Getting future edges here...
        edges = torch_geometric.nn.radius_graph(
            x=pos,
            r=self.r,
            batch=self.idx[0],
            loop=True,
            max_num_neighbors=self.k,
        )
        return edges

    def remove_noncausal_edges(self, edges, T, taus):
        # Remove edges where sink > source where
        # sink and source are both in tau
        # TODO: This is not yet correct
        keep = edges[0] < edges[1]
        return edges[:, keep]

    def update(
        self,
        x,
        pos,
        rot,
        old_x,
        old_pos,
        old_rot,
        T,
        taus
    ):
        """Add new observations to the respective state mats"""
        old_x[self.new_idx] = x[self.out_idx]
        old_pos[self.new_idx] = pos[self.out_idx]
        old_rot[self.new_idx] = rot[self.out_idx]
        return old_x, old_pos, old_rot

    def compute_idx(self, T, taus):
        """Precompute useful indices for forward pass"""
        # Compute idxs once for efficiency
        # Nearly all idxs are pairs of [batch, time] indices
        # Idx pointing to non-padded elements
        # Shape: [sum(B[i]*T[i])] * 2
        self.idx = self.make_idx(T, taus)
        # Idx pointing to new elements in the x, pos, vert mats
        # Shape: [sum(taus[i])] * 2
        self.new_idx = self.make_new_idx(T, taus)
        # Idx pointing to new elements in the flattened gnn output
        # Shape: [sum(taus[i])], note there is only one idx here
        self.flat_new_idx = self.make_flat_new_idx(T, taus)
        # Idx pointing to padded output elements (this is where the outputs go)
        # Shape: [sum(taus[i])]
        self.out_idx = self.make_output_idx(taus)
        # Pointer to the last node in each graph
        # Shape: [B]
        self.back_ptr = (T + taus).cumsum(0) - 1
        # Pointer to the zeroth node in each graph
        # Shape: [B]
        self.front_ptr = torch.cat(
            [torch.tensor([0], device=T.device), self.back_ptr[:-1] + 1]
        )
        #self.front_ptr = self.back_ptr.roll(1)
        #self.front_ptr[0] = 0

    def causal_forward(self, x, pos, rot, T, taus, out_batch, out_time):
        """Causal forward restricts edges to being fully-causal. This means
        incoming edges of node t cannot be updated except at time t. This prevents
        loop closures, but uses significantly less memory and runs faster. Note
        that this produces differen graph topologies than full_forward."""
        # Remove padding and flatten
        # Unpadded shapes are [B*(T+taus), *]
        x = x[self.idx]
        pos = pos[self.idx]
        rot = rot[self.idx]

        if self.edge_method == "knn":
            edges = self.knn_edges(x, pos, rot)
        else:
            edges = self.radius_edges(x, pos, rot)
        # TODO: This changes eval -- we can rewire old things to be noncausal
        #if self.training:
        edges = self.remove_noncausal_edges(edges, T, taus)
        # [B, T] ordering is [0, 0], [0, 1], ... [0, t], [1, 0]
        # TODO: Pooling can be done in output using new_idx
        # as well as max_hops graph reduction and sampling 
        output = self.gnn(
            x, edges, pos, rot, self.idx[0], self.front_ptr, self.back_ptr, self.flat_new_idx
        )

        # return output
        # Offset from 0 instead of T 
        return output[self.flat_new_idx]

    def full_forward(self, x, pos, rot, T, taus, out_batch, out_time):
        """Unlike causal_forward, full_forward allows graph rewiring for
        old vertices (loop closures), but needs to construct a separate graph
        for each timestep, greatly increasing memory usage and reducing 
        throughput"""
        # TODO dont hardcode hidden
        # TODO: flat_new_idx indexing is probably wrong,
        # as the total number of nodes is different here than in the causal case
        graphs = []
        for b in range(out_batch):
            for t in range(out_time):
                if t < taus[b]:
                    curr_slice = slice(0, T[b] + t + 1)
                    graphs.append(
                        torch_geometric.data.Data(
                            x=x[b, curr_slice],
                            pos=pos[b, curr_slice],
                            rot=rot[b, curr_slice]
                        )
                    )

        batch = torch_geometric.data.Batch.from_data_list(graphs)
        if self.edge_method == "knn":
            batch.edge_index = torch_geometric.nn.knn_graph(
                batch.pos, k=self.k, batch=batch.batch
            )
        else:
            batch.edge_index = torch_geometric.nn.radius_graph(
                x=batch.pos,
                r=self.r,
                batch=batch.batch,
                loop=True,
                max_num_neighbors=self.k,
            )

        output = self.gnn(batch.x, batch.edge_index, batch.pos, batch.rot, batch.batch)
        return output

    def forward(
        self,
        x, # [B, taus.max(), F]
        pos, # [B, taus.max(), 2]
        rot, # [B, taus.max(), 1]
        taus, # [B]
        state, # [old_x, old_pos, old_rots, each of size [B, T.max(), *]]
    ):
        old_x, old_pos, old_rot, T = state
        out_batch, out_time = x.shape[0], taus.max()
        # Update hidden state
        self.compute_idx(T, taus)
        x, pos, rot = self.update(x, pos, rot, old_x, old_pos, old_rot, T, taus)
        state = [x, pos, rot, T + taus]
        if self.causal:
            output_at_target = self.causal_forward(x, pos, rot, T, taus, out_batch, out_time)
        else:
            output_at_target = self.full_forward(x, pos, rot, T, taus, out_batch, out_time)

        # Compute padded output at the inputted vert idxs
        padded_output = torch.zeros(
            (out_batch, out_time, output.shape[-1]), device=x.device
        )
        # Offset from 0 instead of T 
        padded_output[self.out_idx] = output[self.flat_new_idx]

        return padded_output, state
