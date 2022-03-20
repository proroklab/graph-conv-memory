import torch
import torch_geometric

class NavGCM(torch.nn.Module):
    def __init__(
        self, 
        gnn,
        max_verts=128,
        edge_method="knn",
        k=10,
    ):
        super().__init__()
        self.k = k
        self.gnn = gnn
        self.max_verts = max_verts
        self.edge_method = edge_method

    def make_batch_idx(
        self, T, taus
    ):
        return torch.repeat_interleave(
            torch.arange(T.shape[0], device=T.device),
            T + taus,
        )

    def make_time_idx(
        self, T, taus
    ):
        
        return torch.cat([
            torch.arange(T[b] + taus[b], device=T.device) 
            for b in range(T.shape[0])
        ], dim=-1)


    def make_batch_tau_idx(
        self, T, taus
    ):
        return torch.repeat_interleave(
            torch.arange(T.shape[0], device=T.device),
            taus,
        )

    def batch_T_idx(
        self, T, taus
    ):
        return torch.repeat_interleave(
            torch.arange(T.shape[0], device=T.device),
            T,
        )

    def make_time_tau_idx(
        self, T, taus
    ):
        return torch.cat([
            torch.arange(T[b], T[b] + taus[b], device=T.device)
            for b in range(T.shape[0])
        ], dim=-1)

    def make_time_tau_idx_offset(
        self, taus
    ):
        return torch.cat([
            torch.arange(taus[b], device=taus.device)
            for b in range(taus.shape[0])
        ], dim=-1)

    def make_time_T_idx(
        self, T, taus
    ):
        return torch.cat([
            torch.arange(T[b], device=T.device)
            for b in range(T.shape[0])
        ], dim=-1)

    def make_flat_tau_idx(
        self, T, taus
    ):
        cs = (T + taus).cumsum(0)
        return torch.cat([
            torch.arange(cs[b] - taus[b], cs[b], device=T.device)
            for b in range(T.shape[0])
        ], dim=-1)

    def construct_edges(
        self, x, pos, rot
    ):
        # TODO: knn needs noise so it can connect identical verts
        edges = torch_geometric.nn.knn_graph(
            x=pos,
            k=self.k,
            batch=self.idx[0]
        )
        return edges

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
        new_time_idx = torch.cat([
            torch.arange(taus[b], device=T.device) 
            for b in range(T.shape[0])
        ], dim=-1)
        new_batch_idx = torch.repeat_interleave(
            torch.arange(T.shape[0], device=T.device), taus
        )
        old_x[self.new_idx] = x[new_batch_idx, new_time_idx]
        old_pos[self.new_idx] = old_pos[new_batch_idx, new_time_idx]
        old_rot[self.new_idx] = old_rot[new_batch_idx, new_time_idx]
        return old_x, old_pos, old_rot

    def forward(
        self,
        x, # [B, taus.max(), F]
        pos, # [B, taus.max(), 2]
        rot, # [B, taus.max(), 1]
        taus, # [B]
        state, # [old_x, old_pos, old_rots, each of size [B, T.max(), *]]
    ):
        old_x, old_pos, old_rot, T = state
        # Compute idxs once for efficiency
        out_batch, out_time = x.shape[0], taus.max()
        # All idxs are pairs of [batch, time] indices
        # Idx pointing to non-padded elements
        # Shape: [sum(B[i]*T[i])]
        self.idx = self.make_batch_idx(T, taus), self.make_time_idx(T, taus)
        # Idx pointing to new elements in the x, pos, vert mats
        # Shape: [sum(taus[i])]
        self.new_idx = self.make_batch_tau_idx(T, taus), self.make_time_tau_idx(T, taus)
        # Idx pointing to new elements in the flattened gnn output
        # Shape: [sum(taus[i])], note there is only one idx here
        self.flat_new_idx = self.make_flat_tau_idx(T, taus)
        # Idx pointing to padded output elements (one per batch-time)
        # Shape: [sum(taus[i])]
        self.out_idx = self.new_idx[0], self.make_time_tau_idx_offset(taus)
        # Pointer to the last node in each graph
        # Shape: [B]
        self.back_ptr = (T + taus).cumsum(0) - 1

        # Add new inputs to recurrent states
        x, pos, rot = self.update(x, pos, rot, old_x, old_pos, old_rot, T, taus)
        state = [x, pos, rot, T + taus]

        # Remove padding and flatten
        # Unpadded shapes are [B*(T+taus), *]
        x = x[self.idx]
        pos = pos[self.idx]
        rot = rot[self.idx]

        edges = self.construct_edges(x, pos, rot)
        # [B, T] ordering is [0, 0], [0, 1], ... [0, t], [1, 0]
        # TODO: Pooling can be done in output using new_idx
        # as well as max_hops graph reduction and sampling 
        output = self.gnn(x, edges, rot, pos, self.idx[0], self.back_ptr)

        # Compute padded output at the inputted vert idxs
        padded_output = torch.zeros(
            (out_batch, out_time, output.shape[-1]), device=x.device
        )
        # Offset from 0 instead of T 
        padded_output[self.out_idx] = output[self.flat_new_idx]

        return padded_output, state
