import unittest
import torch
import torch_geometric
from collections import OrderedDict

from gcm.gcm import DenseGCM
from gcm.edge_selectors.temporal import TemporalBackedge
from gcm.sparse_edge_selectors.temporal import TemporalEdge
from gcm import util
from gcm.sparse_gcm import SparseGCM


class TestPack(unittest.TestCase):
    def setUp(self):
        self.F = 4
        self.B = 3
        self.T = torch.tensor([2, 3, 0])
        self.graph_size = 5
        self.max_edges = 10

    def test_unpack_pack(self):
        nodes = torch.zeros(self.B, self.graph_size, self.F)

        dense_edge = torch.empty(self.B, 2, self.max_edges, dtype=torch.long).fill_(-1)
        dense_edge[0, :, 0] = torch.tensor([0, 1])
        dense_edge[1, :, 0] = torch.tensor([0, 1])
        dense_edge[1, :, 1] = torch.tensor([1, 2])

        dense_weight = torch.empty(self.B, 1, self.max_edges).fill_(1.0)
        dense_weight[0, 0, 0] = 0.5
        dense_weight[1, 0, 1] = 0.25
        initial_packed_hidden = (
            nodes.clone(),
            dense_edge.clone(),
            dense_weight.clone(),
            self.T.clone(),
        )

        packed_hidden = (nodes, dense_edge, dense_weight, self.T)
        unpacked_hidden = util.unpack_hidden(packed_hidden, self.B)
        repacked_hidden = util.pack_hidden(unpacked_hidden, self.B, self.max_edges)

        for i in range(len(initial_packed_hidden)):
            if not (initial_packed_hidden[i] == repacked_hidden[i]).all():
                self.fail(
                    f"packed hidden tensor {i} != repacked hidden tensor"
                    "{initial_packed_hidden[i]) != {repacked_hidden[i]}"
                )

    def test_unpack_pack_one_batch(self):
        self.B = 1
        self.T = torch.tensor([3])
        nodes = torch.zeros(self.B, self.graph_size, self.F)

        dense_edge = torch.empty(self.B, 2, self.max_edges, dtype=torch.long).fill_(-1)
        dense_edge[0, :, 0] = torch.tensor([0, 1])
        dense_edge[0, :, 1] = torch.tensor([0, 2])
        dense_edge[0, :, 2] = torch.tensor([1, 2])

        dense_weight = torch.empty(self.B, 1, self.max_edges).fill_(1.0)
        dense_weight[0, 0, 0] = 0.5
        dense_weight[0, 0, 1] = 0.25
        initial_packed_hidden = (
            nodes.clone(),
            dense_edge.clone(),
            dense_weight.clone(),
            self.T.clone(),
        )

        packed_hidden = (nodes, dense_edge, dense_weight, self.T)
        unpacked_hidden = util.unpack_hidden(packed_hidden, self.B)
        repacked_hidden = util.pack_hidden(unpacked_hidden, self.B, self.max_edges)

        for i in range(len(initial_packed_hidden)):
            if not (initial_packed_hidden[i] == repacked_hidden[i]).all():
                self.fail(
                    f"packed hidden tensor {i} != repacked hidden tensor"
                    "{initial_packed_hidden[i]) != {repacked_hidden[i]}"
                )

    def test_unpack_pack_full_empty(self):
        self.B = 3
        self.T = torch.tensor([5, 0, 0])
        nodes = torch.zeros(self.B, self.graph_size, self.F)

        dense_edge = torch.empty(self.B, 2, self.max_edges, dtype=torch.long).fill_(-1)
        dense_edge[0, :, 0] = torch.tensor([0, 1])
        dense_edge[0, :, 1] = torch.tensor([0, 2])
        dense_edge[0, :, 2] = torch.tensor([0, 3])
        dense_edge[0, :, 3] = torch.tensor([0, 4])

        dense_edge[0, :, 3] = torch.tensor([1, 2])
        dense_edge[0, :, 4] = torch.tensor([1, 3])
        dense_edge[0, :, 5] = torch.tensor([1, 4])

        dense_edge[0, :, 6] = torch.tensor([2, 3])
        dense_edge[0, :, 7] = torch.tensor([2, 4])

        dense_edge[0, :, 8] = torch.tensor([3, 4])

        dense_weight = torch.empty(self.B, 1, self.max_edges).fill_(1.0)
        dense_weight[0, 0, 0] = 0.5
        dense_weight[0, 0, 5] = 0.25
        initial_packed_hidden = (
            nodes.clone(),
            dense_edge.clone(),
            dense_weight.clone(),
            self.T.clone(),
        )

        packed_hidden = (nodes, dense_edge, dense_weight, self.T)
        unpacked_hidden = util.unpack_hidden(packed_hidden, self.B)
        repacked_hidden = util.pack_hidden(unpacked_hidden, self.B, self.max_edges)

        for i in range(len(initial_packed_hidden)):
            if not (initial_packed_hidden[i] == repacked_hidden[i]).all():
                self.fail(
                    f"packed hidden tensor {i} != repacked hidden tensor"
                    "{initial_packed_hidden[i]) != {repacked_hidden[i]}"
                )

    def test_unpack_pack_ragged(self):
        self.B = 3
        self.T = torch.tensor([5, 4, 3])
        nodes = torch.zeros(self.B, self.graph_size, self.F)

        dense_edge = torch.empty(self.B, 2, self.max_edges, dtype=torch.long).fill_(-1)
        dense_edge[0, :, 0] = torch.tensor([0, 1])
        dense_edge[0, :, 1] = torch.tensor([0, 2])
        dense_edge[0, :, 2] = torch.tensor([1, 2])
        dense_edge[0, :, 3] = torch.tensor([2, 3])
        dense_edge[0, :, 4] = torch.tensor([3, 4])

        dense_edge[1, :, 0] = torch.tensor([0, 1])
        dense_edge[1, :, 1] = torch.tensor([0, 2])
        dense_edge[1, :, 2] = torch.tensor([1, 2])
        dense_edge[1, :, 3] = torch.tensor([1, 3])

        dense_edge[2, :, 0] = torch.tensor([0, 1])
        dense_edge[2, :, 1] = torch.tensor([0, 2])

        dense_weight = torch.empty(self.B, 1, self.max_edges).fill_(1.0)
        dense_weight[0, 0, 0] = 0.5
        dense_weight[0, 0, 1] = 0.25

        dense_weight[1, 0, 0] = 0.1
        dense_weight[1, 0, 1] = 0.2

        initial_packed_hidden = (
            nodes.clone(),
            dense_edge.clone(),
            dense_weight.clone(),
            self.T.clone(),
        )

        packed_hidden = (nodes, dense_edge, dense_weight, self.T)
        unpacked_hidden = util.unpack_hidden(packed_hidden, self.B)
        repacked_hidden = util.pack_hidden(unpacked_hidden, self.B, self.max_edges)

        for i in range(len(initial_packed_hidden)):
            if not (initial_packed_hidden[i] == repacked_hidden[i]).all():
                self.fail(
                    f"packed hidden tensor {i} != repacked hidden tensor"
                    "{initial_packed_hidden[i]) != {repacked_hidden[i]}"
                )

    def test_unpack_pack_many(self):
        nodes = torch.zeros(self.B, self.graph_size, self.F)

        dense_edge = torch.empty(self.B, 2, self.max_edges, dtype=torch.long).fill_(-1)
        dense_edge[0, :, 0] = torch.tensor([0, 1])
        dense_edge[1, :, 0] = torch.tensor([0, 1])
        dense_edge[1, :, 1] = torch.tensor([1, 2])

        dense_weight = torch.empty(self.B, 1, self.max_edges).fill_(1.0)
        dense_weight[0, 0, 0] = 0.5
        dense_weight[1, 0, 1] = 0.25
        initial_packed_hidden = (
            nodes.clone(),
            dense_edge.clone(),
            dense_weight.clone(),
            self.T.clone(),
        )

        packed_hidden = (nodes, dense_edge, dense_weight, self.T)
        for i in range(10):
            unpacked_hidden = util.unpack_hidden(packed_hidden, self.B)
            packed_hidden = util.pack_hidden(unpacked_hidden, self.B, self.max_edges)

        for i in range(len(initial_packed_hidden)):
            if not (initial_packed_hidden[i] == packed_hidden[i]).all():
                self.fail(
                    f"packed hidden tensor {i} != repacked hidden tensor"
                    "{initial_packed_hidden[i]) != {repacked_hidden[i]}"
                )

    def test_unpack_empty(self):
        nodes = torch.zeros(self.B, self.graph_size, self.F)

        dense_edge = torch.empty(self.B, 2, self.max_edges, dtype=torch.long).fill_(-1)

        dense_weight = torch.empty(self.B, 1, self.max_edges).fill_(1.0)
        initial_packed_hidden = (
            nodes.clone(),
            dense_edge.clone(),
            dense_weight.clone(),
            self.T.clone(),
        )

        packed_hidden = (nodes, dense_edge, dense_weight, self.T)
        unpacked_hidden = util.unpack_hidden(packed_hidden, self.B)
        repacked_hidden = util.pack_hidden(unpacked_hidden, self.B, self.max_edges)

        for i in range(len(initial_packed_hidden)):
            if not (initial_packed_hidden[i] == repacked_hidden[i]).all():
                self.fail(
                    f"packed hidden tensor {i} != repacked hidden tensor"
                    "{initial_packed_hidden[i]) != {repacked_hidden[i]}"
                )


class TestDenseVsSparse(unittest.TestCase):
    def setUp(self):
        self.F = 3
        dense_conv_type = torch_geometric.nn.DenseGraphConv
        sparse_conv_type = torch_geometric.nn.GraphConv
        self.dense_g = torch_geometric.nn.Sequential(
            "x, adj, weights, B, N",
            [
                (dense_conv_type(self.F, self.F), "x, adj -> x"),
                (dense_conv_type(self.F, self.F), "x, adj -> x"),
            ],
        )
        self.sparse_g = torch_geometric.nn.Sequential(
            "x, edges, weights",
            [
                (sparse_conv_type(self.F, self.F), "x, edges, weights -> x"),
                (sparse_conv_type(self.F, self.F), "x, edges, weights -> x"),
            ],
        )
        dense_params = self.dense_g.state_dict()
        self.sparse_g.load_state_dict(dense_params)
        # sanity check
        for k, v in self.sparse_g.state_dict().items():
            self.assertTrue((v == self.dense_g.state_dict()[k]).all())

        self.dense_gcm = DenseGCM(self.dense_g, graph_size=8)
        self.sparse_gcm = SparseGCM(self.sparse_g, graph_size=8)

    def test_gnn_no_weights(self):
        sparse_out = self.sparse_g(
            torch.ones(2, 3, self.F), torch.ones(2, 0, dtype=torch.long), torch.ones(0)
        )
        dense_out = self.dense_g(
            torch.ones(2, 3, self.F),
            torch.zeros(2, 3, 3),
            torch.ones(2, 3, 3),
            None,
            None,
        )

        self.assertTrue((sparse_out == dense_out).all())

    def test_no_edges(self):
        F = self.F
        B = 3
        ts = 4
        self.obs = torch.arange(B * ts * F, dtype=torch.float32).reshape(B, ts, F)

        dense_outs = []

        hidden = None
        for i in range(ts):
            dense_out, hidden = self.dense_gcm(self.obs[:, i], hidden)
            dense_outs.append(dense_out)
        dense_outs = torch.stack(dense_outs, dim=1)

        # One step at at ime
        self.sparse_step = SparseGCM(self.sparse_g, graph_size=8)
        sparse_step_hidden = None
        sparse_step_outs = []
        for i in range(ts):
            taus = torch.ones(B, dtype=torch.long)
            sparse_step_out, sparse_step_hidden = self.sparse_gcm(
                self.obs[:, i].unsqueeze(1), taus, sparse_step_hidden
            )
            sparse_step_outs.append(sparse_step_out)
        sparse_step_outs = torch.cat(sparse_step_outs, dim=1)

        # All at once
        taus = torch.ones(B, dtype=torch.long) * ts
        sparse_outs, sparse_hidden = self.sparse_gcm(self.obs, taus, None)

        if dense_outs.numel() != sparse_outs.numel():
            self.fail(f"sizes {dense_outs.numel()} != {sparse_outs.numel()}")

        # Check hiddens
        if not torch.all(hidden[0] == sparse_hidden[0]):
            self.fail(f"{hidden[0]} != {sparse_hidden[0]}")

        if not torch.all(hidden[0] == sparse_step_hidden[0]):
            self.fail(f"{hidden[0]} != {sparse_step_hidden[0]}")

        if not torch.all(dense_outs == sparse_outs):
            self.fail(f"{dense_outs} != {sparse_outs}")

        if not torch.all(dense_outs == sparse_step_outs):
            self.fail(f"{dense_outs} != {sparse_step_outs}")

    def test_temporal_edges(self):
        self.dense_gcm = DenseGCM(
            self.dense_g, edge_selectors=TemporalBackedge([1, 2]), graph_size=8
        )
        self.sparse_gcm = SparseGCM(
            self.sparse_g, edge_selectors=TemporalEdge([1, 2]), graph_size=8
        )
        F = self.F
        B = 3
        ts = 8
        self.obs = torch.arange(B * ts * F, dtype=torch.float32).reshape(B, ts, F)

        dense_outs = []

        dense_hidden = None
        for i in range(ts):
            dense_out, dense_hidden = self.dense_gcm(self.obs[:, i], dense_hidden)
            dense_outs.append(dense_out)
        dense_outs = torch.stack(dense_outs, dim=1)

        taus = torch.ones(B, dtype=torch.long) * ts
        sparse_outs, sparse_hidden = self.sparse_gcm(self.obs, taus, None)

        if dense_outs.numel() != sparse_outs.numel():
            self.fail(f"sizes {dense_outs.numel()} != {sparse_outs.numel()}")

        # Dense edges
        # [1,0], [2,1], [3,2] for each batch
        # Sparse edges
        # [0,1], [1,2], [2,3]

        # Check hiddens
        if not torch.all(dense_hidden[0] == sparse_hidden[0]):
            self.fail(f"{dense_hidden[0]} != {sparse_hidden[0]}")

        if not torch.all(dense_outs == sparse_outs):
            self.fail(f"{dense_outs} != {sparse_outs}")

    def test_learning_temporal_edges(self):
        self.dense_gcm = DenseGCM(
            self.dense_g, edge_selectors=TemporalBackedge([1, 2]), graph_size=8
        )
        self.sparse_gcm = SparseGCM(
            self.sparse_g, edge_selectors=TemporalEdge([1, 2]), graph_size=8
        )
        self.sparse_step = SparseGCM(
            self.sparse_g, edge_selectors=TemporalEdge([1, 2]), graph_size=8
        )
        d_opt = torch.optim.Adam(self.dense_gcm.parameters())
        s_opt = torch.optim.Adam(self.sparse_gcm.parameters())
        ss_opt = torch.optim.Adam(self.sparse_step.parameters())
        F = self.F
        B = 3
        ts = 8
        num_iters = 3

        for i in range(num_iters):
            d_opt.zero_grad()
            s_opt.zero_grad()
            ss_opt.zero_grad()
            self.obs = torch.rand((B, ts, F), dtype=torch.float32)

            dense_outs = []

            dense_hidden = None
            for i in range(ts):
                dense_out, dense_hidden = self.dense_gcm(self.obs[:, i], dense_hidden)
                dense_outs.append(dense_out)
            dense_outs = torch.stack(dense_outs, dim=1)

            # One step sparse
            sparse_step_hidden = None
            sparse_step_outs = []
            for i in range(ts):
                taus = torch.ones(B, dtype=torch.long)
                sparse_step_out, sparse_step_hidden = self.sparse_gcm(
                    self.obs[:, i].unsqueeze(1), taus, sparse_step_hidden
                )
                sparse_step_outs.append(sparse_step_out)
            sparse_step_outs = torch.cat(sparse_step_outs, dim=1)

            # Time batched sparse
            taus = torch.ones(B, dtype=torch.long) * ts
            sparse_outs, sparse_hidden = self.sparse_gcm(self.obs, taus, None)

            if dense_outs.numel() != sparse_outs.numel():
                self.fail(f"sizes {dense_outs.numel()} != {sparse_outs.numel()}")

            if dense_outs.numel() != sparse_step_outs.numel():
                self.fail(f"sizes {dense_outs.numel()} != {sparse_step_outs.numel()}")

            # Check hiddens
            if not torch.all(dense_hidden[0] == sparse_hidden[0]):
                self.fail(f"{dense_hidden[0]} != {sparse_hidden[0]}")

            if not torch.all(dense_outs == sparse_outs):
                self.fail(f"{dense_outs} != {sparse_outs}")

            if not torch.all(dense_outs == sparse_step_outs):
                self.fail(f"{dense_outs} != {sparse_step_outs}")
            sparse_outs.mean().backward()
            dense_outs.mean().backward()
            d_opt.step()
            s_opt.step()

            for k, v in self.sparse_g.state_dict().items():
                if not torch.allclose(v, self.dense_g.state_dict()[k], atol=0.01):
                    self.fail(
                        f"Parameters diverged: {v}, {self.dense_g.state_dict()[k]}"
                    )


if __name__ == "__main__":
    unittest.main()
