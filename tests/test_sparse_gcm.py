import unittest
import torch
import torch_geometric
from collections import OrderedDict
import gym

from gcm.gcm import DenseGCM
from gcm.edge_selectors.temporal import TemporalBackedge
from gcm.sparse_edge_selectors.temporal import TemporalEdge
from gcm.edge_selectors.learned import LearnedEdge as DLearnedEdge
from gcm.sparse_edge_selectors.learned import LearnedEdge as SLearnedEdge
from gcm import util
from gcm.sparse_gcm import SparseGCM
from gcm import ray_sparse_gcm


class TestFlattenAdj(unittest.TestCase):
    def setUp(self):
        self.F = 4
        self.B = 4
        self.T = torch.tensor([1, 2, 0, 0])
        self.taus = torch.zeros(4, dtype=torch.long)
        self.graph_size = 5
        self.max_edges = 6

    def test_flatten_unflatten(self):
        adj = torch.zeros(self.B, 2, self.max_edges)
        adj[1, 0, 1] = 1.0
        adj[1, 1, 2] = 2.0
        sparse_adj = adj.to_sparse()
        e, w, b = util.flatten_adj(sparse_adj, self.T, self.taus, self.B)
        new_sparse_adj = util.unflatten_adj(e, w, b, self.T, self.taus, self.B, self.max_edges)

        if not torch.all(
            sparse_adj._indices() == new_sparse_adj._indices()
        ):
            self.fail(
                f"\n{sparse_adj._indices()} != \n{new_sparse_adj._indices()}"
            )
        if not torch.all(
            sparse_adj.coalesce().values() == new_sparse_adj.coalesce().values()
        ):
            self.fail(
                f"{sparse_adj.coalesce().values()} != {sparse_adj.coalesce().values()}"
            )

    def test_flatten_unflatten2(self):
        self.T = torch.tensor([1, 5, 2, 1])
        self.taus = torch.zeros(4, dtype=torch.long)
        adj = torch.zeros(self.B, 2, self.max_edges)
        adj[1, 0, 1] = 1.0
        adj[1, 1, 2] = 2.0
        adj[1, 1, 3] = 3.0
        adj[2, 0, 1] = 4.0
        sparse_adj = adj.to_sparse()
        e, w, b = util.flatten_adj(sparse_adj, self.T, self.taus, self.B)
        new_sparse_adj = util.unflatten_adj(e, w, b, self.T, self.taus, self.B, self.max_edges)

        if not torch.all(
            sparse_adj._indices() == new_sparse_adj._indices()
        ):
            self.fail(
                f"\n{sparse_adj._indices()} != \n{new_sparse_adj._indices()}"
            )
        if not torch.all(
            sparse_adj.coalesce().values() == new_sparse_adj.coalesce().values()
        ):
            self.fail(
                f"{sparse_adj.coalesce().values()} != {sparse_adj.coalesce().values()}"
            )



class TestPack(unittest.TestCase):
    def setUp(self):
        self.F = 4
        self.B = 3
        self.T = torch.tensor([2, 3, 0])
        self.graph_size = 5
        self.max_edges = 10

    def test_unpack_pack_empty(self):
        self.T = torch.tensor([0, 0, 0]) 
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

    def test_unpack_pack(self):
        nodes = torch.zeros(self.B, self.graph_size, self.F)

        dense_edge = torch.empty(self.B, 2, self.max_edges, dtype=torch.long).fill_(-1)
        dense_edge[0, :, 0] = torch.tensor([0, 1])
        dense_edge[1, :, 0] = torch.tensor([0, 1])
        dense_edge[1, :, 1] = torch.tensor([1, 2])

        dense_weight = torch.empty(self.B, 1, self.max_edges).fill_(1.0)
        dense_weight[0, 0, 0] = 0.5
        dense_weight[1, 0, 0] = 0.33
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
                    f"\n{initial_packed_hidden[i]} != \n\n{repacked_hidden[i]}"
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

        # Check hiddens
        if not torch.all(dense_hidden[0] == sparse_hidden[0]):
            self.fail(f"{dense_hidden[0]} != {sparse_hidden[0]}")

        if not torch.all(dense_hidden[1].nonzero().T == sparse_hidden[1].coalesce().indices()):
            self.fail(f"dense and sparse edges inequal: \n{dense_hidden[1].nonzero().T} != \n{sparse_hidden[1]._indices()}")

        if not torch.all(dense_outs == sparse_outs):
            self.fail(f"{dense_outs} != {sparse_outs}")


    def test_temporal_edges_2_hop(self):
        self.dense_gcm = DenseGCM(
            self.dense_g, edge_selectors=TemporalBackedge([1, 2]), graph_size=8
        )
        self.sparse_gcm = SparseGCM(
            self.sparse_g, edge_selectors=TemporalEdge([1, 2]), graph_size=8, max_hops=2
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

        # Check hiddens
        if not torch.all(dense_hidden[0] == sparse_hidden[0]):
            self.fail(f"{dense_hidden[0]} != {sparse_hidden[0]}")

        if not torch.all(dense_hidden[1].nonzero().T == sparse_hidden[1].coalesce().indices()):
            self.fail(f"dense and sparse edges inequal: \n{dense_hidden[1].nonzero().T} != \n{sparse_hidden[1]._indices()}")

        if not torch.all(dense_outs == sparse_outs):
            self.fail(f"{dense_outs} != {sparse_outs}")


    def test_temporal_edges_many_iter_2_hop(self):
        self.dense_gcm = DenseGCM(
            self.dense_g, edge_selectors=TemporalBackedge([1, 2]), graph_size=8
        )
        self.sparse_gcm = SparseGCM(
            self.sparse_g, edge_selectors=TemporalEdge([1, 2]), graph_size=8, max_hops=2
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

        sparse_hidden = None
        taus = torch.ones(B, dtype=torch.long)
        sparse_outs = []
        for i in range(ts):
            sparse_out, sparse_hidden = self.sparse_gcm(self.obs[:,i].unsqueeze(1), taus, sparse_hidden)
            sparse_outs.append(sparse_out)
        sparse_outs = torch.cat(sparse_outs, dim=1)

        if dense_outs.numel() != sparse_outs.numel():
            self.fail(f"sizes {dense_outs.numel()} != {sparse_outs.numel()}")

        if not torch.all(dense_hidden[1].nonzero().T == sparse_hidden[1].coalesce().indices()):
            self.fail(f"sparse and dense edges inequal: \n{dense_hidden[1].nonzero().T} != \n{sparse_hidden[1].coalesce().indices()}")

        if not torch.all(dense_outs == sparse_outs):
            self.fail(f"{dense_outs} != {sparse_outs}")
    def test_temporal_edges_many_iter(self):
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

        sparse_hidden = None
        taus = torch.ones(B, dtype=torch.long)
        sparse_outs = []
        for i in range(ts):
            sparse_out, sparse_hidden = self.sparse_gcm(self.obs[:,i].unsqueeze(1), taus, sparse_hidden)
            sparse_outs.append(sparse_out)
        sparse_outs = torch.cat(sparse_outs, dim=1)

        if dense_outs.numel() != sparse_outs.numel():
            self.fail(f"sizes {dense_outs.numel()} != {sparse_outs.numel()}")

        if not torch.all(dense_hidden[1].nonzero().T == sparse_hidden[1].coalesce().indices()):
            self.fail(f"sparse and dense edges inequal: \n{dense_hidden[1].nonzero().T} != \n{sparse_hidden[1].coalesce().indices()}")

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

            if not torch.allclose(dense_outs, sparse_outs, atol=0.01):
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


class DummyEdgenet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ghost = torch.nn.Linear(1, 1)

    def forward(self, x):
        return 1e15 * torch.all(x > 0, dim=1).float()


class TestLearnedEdge(unittest.TestCase):
    def setUp(self):
        self.F = 3
        sparse_conv_type = torch_geometric.nn.GraphConv
        self.sparse_g = torch_geometric.nn.Sequential(
            "x, edges, weights",
            [
                (sparse_conv_type(self.F, self.F), "x, edges, weights -> x"),
                (sparse_conv_type(self.F, self.F), "x, edges, weights -> x"),
            ],
        )



    def test_first_pass(self):
        B = 2
        gsize = 5
        taus = torch.ones(B, dtype=torch.long) * 5
        T = torch.zeros(B)
        obs = torch.zeros(B, taus.max().int(), self.F)
        obs[0,0] = 1
        obs[0,4] = 1
        obs[1,1] = 1
        obs[1,4] = 1
        sel = SLearnedEdge(input_size=0, model=DummyEdgenet(), num_edge_samples=1)
        gcm = SparseGCM(
            self.sparse_g, graph_size=gsize, edge_selectors=sel
        )

        out, hidden = gcm(obs, taus, hidden=None)
        # Should result in edges:
        #b0: [1,0], [2,?], [3,?], [4,0]
        #b1: [1,0], [2,?], [3,?], [4,1]
        self.assertTrue(torch.tensor([0, 1, 0]) in hidden[1]._indices().T)
        self.assertTrue(torch.tensor([0, 4, 0]) in hidden[1]._indices().T)
        self.assertTrue(torch.tensor([1, 1, 0]) in hidden[1]._indices().T)
        self.assertTrue(torch.tensor([1, 4, 1]) in hidden[1]._indices().T)
        
    def test_second_pass(self):
        B = 2
        gsize = 10
        taus = torch.ones(B, dtype=torch.long) * 5
        T = torch.zeros(B)
        obs = torch.zeros(B, taus.max().int(), self.F)
        obs[0,0] = 1
        obs[0,4] = 1
        obs[1,1] = 1
        obs[1,4] = 1
        sel = SLearnedEdge(input_size=0, model=DummyEdgenet(), num_edge_samples=1)
        gcm = SparseGCM(
            self.sparse_g, graph_size=gsize, edge_selectors=sel
        )

        out, hidden = gcm(obs, taus, hidden=None)
        # Should result in edges:
        #b0: [1,0], [2,?], [3,?], [4,0]
        #b1: [1,0], [2,?], [3,?], [4,1]
        self.assertTrue(torch.tensor([0, 1, 0]) in hidden[1]._indices().T)
        self.assertTrue(torch.tensor([0, 4, 0]) in hidden[1]._indices().T)
        self.assertTrue(torch.tensor([1, 1, 0]) in hidden[1]._indices().T)
        self.assertTrue(torch.tensor([1, 4, 1]) in hidden[1]._indices().T)

        # Second pass
        # should have first pass edges
        out, hidden = gcm(obs, taus, hidden)
        self.assertTrue(torch.tensor([0, 1, 0]) in hidden[1]._indices().T)
        self.assertTrue(torch.tensor([0, 4, 0]) in hidden[1]._indices().T)
        self.assertTrue(torch.tensor([1, 1, 0]) in hidden[1]._indices().T)
        self.assertTrue(torch.tensor([1, 4, 1]) in hidden[1]._indices().T)


    def test_multi_pass(self):
        B = 2
        gsize = 10
        taus = torch.ones(B, dtype=torch.long)
        T = torch.zeros(B)
        obs = torch.zeros(B, gsize, self.F)
        obs[0,0] = 1
        obs[0,4] = 1
        obs[1,1] = 1
        obs[1,4] = 1
        sel = SLearnedEdge(input_size=0, model=DummyEdgenet(), num_edge_samples=1)
        gcm = SparseGCM(
            self.sparse_g, graph_size=gsize, edge_selectors=sel
        )
        hidden = None
        for i in range(gsize):
            out, hidden = gcm(obs[:,i].unsqueeze(1), taus, hidden)

        # Should result in edges:
        #b0: [1,0], [2,?], [3,?], [4,0]
        #b1: [1,0], [2,?], [3,?], [4,1]
        self.assertTrue(torch.tensor([0, 1, 0]) in hidden[1]._indices().T)
        self.assertTrue(torch.tensor([0, 4, 0]) in hidden[1]._indices().T)
        self.assertTrue(torch.tensor([1, 1, 0]) in hidden[1]._indices().T)
        self.assertTrue(torch.tensor([1, 4, 1]) in hidden[1]._indices().T)

        # Ensure same as batched case
        bout, bhidden = gcm(obs, taus * gsize, None)
        self.assertTrue(bhidden[1].shape == hidden[1].shape)

    def test_window_multi_pass(self):
        B = 2
        gsize = 4
        taus = torch.ones(B, dtype=torch.long)
        T = torch.zeros(B)
        obs = torch.zeros(B, gsize, self.F)
        sel = SLearnedEdge(input_size=0, model=DummyEdgenet(), num_edge_samples=1, window=1)
        gcm = SparseGCM(
            self.sparse_g, graph_size=gsize, edge_selectors=sel
        )
        hidden = None
        for i in range(gsize):
            out, hidden = gcm(obs[:,i].unsqueeze(1), taus, hidden)

        # Should result in edges:
        #b0: [1,0], [2,?], [3,?], [4,0]
        #b1: [1,0], [2,?], [3,?], [4,1]
        desired = torch.tensor(
                [
                    [0, 1, 0], 
                    [0, 2, 1],
                    [0, 3, 2],
                    # second batch
                    [1, 1, 0], 
                    [1, 2, 1],
                    [1, 3, 2],
                ]
        ).T
        if not torch.all(hidden[1].coalesce().indices() == desired):
            self.fail(f"{hidden[1].coalesce().indices()} != {desired}")

    def test_grad(self):
        B = 2
        gsize = 4
        taus = gsize * torch.ones(B, dtype=torch.long)
        T = torch.zeros(B, dtype=torch.long)
        obs = torch.zeros(B, gsize, self.F)
        canary = torch.tensor([1.0], requires_grad=True)
        obs = obs * canary


        sel = SLearnedEdge(input_size=self.F, num_edge_samples=10, window=16)
        adj = sel(obs, T, taus, B)
        adj.coalesce().values().sum().backward()
        self.assertTrue(canary.grad is not None)



class TestUtil(unittest.TestCase):
    def test_flatten_idx(self):
        idx = torch.tensor([
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0]
        ])
        flat, offsets = util.flatten_idx_n_dim(idx)

    def test_flatten_many(self):
        idx = torch.tensor([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 1, 2, 3, 0, 1, 2, 3]
        ])
        flat, offsets = util.flatten_idx_n_dim(idx)
        if flat.unique().shape != flat.shape:
            self.fail(f"Repeated elems {flat}")

    def test_sparse_gumbel_softmax(self):
        idx = torch.tensor([
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 1, 2, 2, 0, 5, 4, 4],
            [0, 0, 1, 0, 0, 3, 0, 3]
        ])
        values = torch.ones(8) * 1e15
        values[3] = 0
        values[-1] = 0
        a = torch.sparse_coo_tensor(idx, values, size=(2, 2, 100, 100))
        res = util.sparse_gumbel_softmax(a, 3, hard=True)
        desired_idx = torch.tensor([
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 1, 2, 0, 5, 4],
            [0, 0, 1, 0, 3, 0]
        ])
        desired_values = torch.ones(6)
        desired = torch.sparse_coo_tensor(
                desired_idx, desired_values, size=(2, 2, 100, 100))
        
        if torch.any(res.coalesce().indices() != desired.coalesce().indices()):
            self.fail(f"{res} != {desired}")
        if torch.any(res.coalesce().values() != desired.coalesce().values()):
            self.fail(f"{res} != {desired}")


class TestE2E(unittest.TestCase):
    def test_e2e_learned_edge(self):
        sparse_g = torch_geometric.nn.Sequential(
            "x, edges, weights",
            [
                (torch_geometric.nn.GraphConv(32, 32), "x, edges, weights -> x"),
                (torch.nn.Tanh()),
                (torch_geometric.nn.GraphConv(32, 32), "x, edges, weights -> x"),
                (torch.nn.Tanh()),
            ],
        )
        B = 8
        num_obs = 256
        obs_size = 32
        sparse_gcm = SparseGCM(
            sparse_g, graph_size=num_obs, edge_selectors=SLearnedEdge(obs_size),
            max_hops=2
        )
        obs = torch.rand(B, num_obs, obs_size)
        taus = torch.ones(B, dtype=torch.long)
        hidden = None
        with torch.no_grad():
            for i in range(num_obs):
                out, hidden = sparse_gcm(obs[:,i,None], taus, hidden)
                tmp = util.pack_hidden(hidden, B, max_edges = 5 * num_obs)
                tmp = util.unpack_hidden(tmp, B)
        # train
        out, hidden = sparse_gcm(obs, taus * num_obs, None)
        tmp = util.pack_hidden(hidden, B, max_edges = 5 * num_obs)
        tmp = util.unpack_hidden(tmp, B)
        out.mean().backward()

    def test_e2e_learned_edge_grad(self):
        sparse_g = torch_geometric.nn.Sequential(
            "x, edges, weights",
            [
                (torch_geometric.nn.GraphConv(32, 32), "x, edges, weights -> x"),
                (torch.nn.Tanh()),
                (torch_geometric.nn.GraphConv(32, 32), "x, edges, weights -> x"),
                (torch.nn.Tanh()),
            ],
        )
        B = 8
        num_obs = 4
        obs_size = 32
        sparse_gcm = SparseGCM(
            sparse_g, graph_size=num_obs, edge_selectors=SLearnedEdge(obs_size),
            max_hops=2
        )
        canary = torch.tensor([1.0], requires_grad=True)
        obs = torch.rand(B, num_obs, obs_size) * canary 
        taus = torch.ones(B, dtype=torch.long)
        hidden = None
        with torch.no_grad():
            for i in range(num_obs):
                out, hidden = sparse_gcm(obs[:,i,None], taus, hidden)
                print(hidden[1]._values().numel())
                tmp = util.pack_hidden(hidden, B, max_edges = 5 * num_obs)
                tmp = util.unpack_hidden(tmp, B)
        # train
        out, hidden = sparse_gcm(obs, taus * num_obs, None)
        tmp = util.pack_hidden(hidden, B, max_edges = 5 * num_obs)
        tmp = util.unpack_hidden(tmp, B)
        out.mean().backward()
        self.assertTrue(canary.grad is not None)

    def test_ray_sparse_edge_grad(self):
        B = 1
        F = 64
        num_obs = 32
        graph_size = 32
        taus = num_obs * torch.ones(B)
        act_space = gym.spaces.Discrete(1)
        obs_space = gym.spaces.Box(high=1000, low=-1000, shape=(F,))
        cfg = ray_sparse_gcm.RaySparseGCM.DEFAULT_CONFIG
        cfg["aux_edge_selectors"] = SLearnedEdge(F)
        ray_gcm = ray_sparse_gcm.RaySparseGCM(
            obs_space,
            act_space,
            1,
            cfg, 
            'my_model',
        )

        canary = torch.tensor([1.0], requires_grad=True)
        input_dict = {
            "obs_flat": torch.ones(B*num_obs, F) * canary
        }
        state = [
            torch.zeros(B, graph_size, F),
            torch.ones(B, 2, 50).long(),
            torch.ones(B, 1, 50),
            torch.zeros(B).long()
        ]
        seq_lens = taus.int().numpy()
        output, hidden = ray_gcm.forward(input_dict, state, seq_lens)
        # Check grads for adj
        hidden[2].sum().backward()
        self.assertTrue(hidden[2].requires_grad)
        self.assertTrue(canary.grad is not None)


    def test_ray_sparse_node_grad(self):
        B = 1
        F = 64
        num_obs = 32
        graph_size = 32
        taus = num_obs * torch.ones(B)
        act_space = gym.spaces.Discrete(1)
        obs_space = gym.spaces.Box(high=1000, low=-1000, shape=(F,))
        cfg = ray_sparse_gcm.RaySparseGCM.DEFAULT_CONFIG
        cfg["aux_edge_selectors"] = SLearnedEdge(F)
        ray_gcm = ray_sparse_gcm.RaySparseGCM(
            obs_space,
            act_space,
            1,
            cfg, 
            'my_model',
        )

        canary = torch.tensor([1.0], requires_grad=True)
        input_dict = {
            "obs_flat": torch.ones(B*num_obs, F) * canary
        }
        state = [
            torch.zeros(B, graph_size, F),
            torch.ones(B, 2, 50).long(),
            torch.ones(B, 1, 50),
            torch.zeros(B).long()
        ]
        seq_lens = taus.int().numpy()
        output, hidden = ray_gcm.forward(input_dict, state, seq_lens)
        # Check grads for nodes
        output.sum().backward()
        self.assertTrue(output.requires_grad)
        self.assertTrue(canary.grad is not None)


if __name__ == "__main__":
    unittest.main()
