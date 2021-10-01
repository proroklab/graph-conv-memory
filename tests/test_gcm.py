import unittest
import math
import torch
import torch_geometric
import torchviz

from gcm.gcm import DenseGCM, DenseToSparse, SparseToDense, PositionalEncoding
from gcm.edge_selectors.temporal import TemporalBackedge
from gcm.edge_selectors.distance import EuclideanEdge, CosineEdge, SpatialEdge
from gcm.edge_selectors.dense import DenseEdge
from gcm.edge_selectors.bernoulli import BernoulliEdge
from gcm.util import diff_or, diff_or2


class TestPositionalEncoding(unittest.TestCase):
    def setUp(self):
        torch.autograd.set_detect_anomaly(True)
        feats = 5
        batches = 2
        N = 7
        self.g = torch_geometric.nn.Sequential(
            "x, adj, weights, B, N",
            [
                (lambda x: x, "x -> x"),
            ],
        )
        self.pe = PositionalEncoding(
            max_len=N,
            mode="add",
        )
        #self.s = DenseGCM(self.g, positional_encoder=self.pe)

        self.nodes = torch.zeros(batches, N, feats)
        self.obs = torch.ones(batches, feats)
        self.adj = torch.zeros(batches, N, N)
        self.weights = torch.ones(batches, N, N)
        self.num_nodes = torch.tensor([0, 7])

    def test_pos_enc_add(self):
        enc = self.pe(self.nodes.clone(), self.num_nodes)
        if not torch.all(enc[0,1,:] == 0):
            self.fail("Off by one error in encoder (overflow)")

        if torch.all(enc[0,0,:] == 0):
            self.fail("Off by one error in encoder (underflow)")

        # PE(x,2i) = sin(x/10000^(2i/D))
        # PE(x,2i+1) = cos(x/10000^(2i/D))
        # 
        # Zeroth row:
        # PE(x=0,2i=0)   = sin(0/1)             = 0
        # PE(x=0,2i+1=1) = cos(0/1)             = pi/2
        # PE(x=0,2i=2)   = sin(0/10000^(2/4))   = 0
        # PE(x=0,2i+1=3) = cos(0/10000^(2/4)    = pi/2
        # 
        # First row:
        # Note we must have 2i so its /6 instead of /5
        # PE(x=1,2i=0)   = sin 1/10000^(0/6)
        # PE(x=1,2i+1=1) = cos 1/10000^(0/6)
        # PE(x=1,2i=2)   = sin 1/10000^(2/6)
        # PE(x=1,2i+1=3) = cos 1/10000^(2/6)
        # PE(x=1,2i=2)   = sin 1/10000^(4/6)

        sin_actual = enc[0,0,::2]
        sin_desired = torch.zeros_like(enc[0,0,::2])
        cos_actual = enc[0,0,1::2]
        cos_desired = torch.ones_like(enc[0,0,1::2]) 

        if torch.abs(sin_actual - sin_desired).sum() > 0.01:
            self.fail(f"Sine zeroth row desired {sin_desired} actual {sin_actual}")
        if torch.abs(cos_actual -  cos_desired).sum() > 0.01:
            self.fail(f"Cosine zeroth row desired {cos_desired} actual {cos_actual}")

        enc = self.pe(self.nodes.clone(), self.num_nodes + 1)
        desired = torch.tensor(
            [
                math.sin( (1/10000)**(0/6)),
                math.cos( (1/10000)**(0/6)),
                math.sin( (1/10000)**(2/6)),
                math.cos( (1/10000)**(2/6)),
                math.sin((1/10000)**(4/6)),
            ]
        )

        if torch.abs(enc[0,1] - desired).sum() > 0.01:
            self.fail(f"Desired {desired} actual {enc[0,1]}")


class TestWrapOverflow(unittest.TestCase):
    def setUp(self):
        torch.autograd.set_detect_anomaly(True)
        feats = 5
        batches = 2
        N = 7
        conv_type = torch_geometric.nn.DenseGraphConv
        self.g = torch_geometric.nn.Sequential(
            "x, adj, weights, B, N",
            [
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
            ],
        )
        self.s = DenseGCM(self.g)

        self.nodes = torch.arange((batches * N * feats), dtype=torch.float).reshape(
            batches, N, feats
        )
        self.obs = torch.ones(batches, feats) * 5
        self.adj = torch.zeros(batches, N, N)
        self.weights = torch.ones(batches, N, N)
        self.num_nodes = torch.tensor([1, 7])

    def test_wrap_overflow(self):
        self.adj[:, 0, :] = 1
        self.adj[:, :, 0] = 1
        self.weights[:, 0, :] = 5
        self.weights[:, :, 0] = 5
        self.nodes[:, 0] = 0

        desired = torch.zeros_like(self.adj)
        desired[0, 0, :] = 1
        desired[0, :, 0] = 1
        desired_weights = torch.ones_like(self.weights)
        desired_weights[0, 0, :] = 5
        desired_weights[0, :, 0] = 5
        desired_weights[1, -1, :] = 0
        desired_weights[1, :, -1] = 0
        desired_nodes = self.nodes.clone()
        desired_nodes[0, 1] = 5
        desired_nodes[1, -1] = 5
        desired_nodes[1, 0] = torch.arange(8 * 5, 9 * 5)
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        if not torch.all(adj == desired):
            self.fail(f"{adj} != {desired}")

        if not torch.all(weights == desired_weights):
            self.fail(f"{weights} != {desired_weights}")

        if not torch.all(nodes[0] == desired_nodes[0]):
            self.fail(f"{nodes[0]} != {desired_nodes[0]}")

        # It's shifted by one
        if not torch.all(nodes[1, 1] == desired_nodes[1, 2]):
            self.fail(f"{nodes[1,2]} != {desired_nodes[1,1]}")

        if not torch.all(nodes[1, 0] == desired_nodes[1, 0]):
            self.fail(f"{nodes[1,0]} != {desired_nodes[1,0]}")

        if not torch.all(nodes[1, -1] == desired_nodes[1, -1]):
            self.fail(f"{nodes[0]} != {desired_nodes[0]}")

    def test_wrap_overflow_no_weights(self):
        self.weights = torch.ones(0)
        self.adj[:, 0, :] = 1
        self.adj[:, :, 0] = 1
        self.nodes[:, 0] = 0

        desired = torch.zeros_like(self.adj)
        desired[0, 0, :] = 1
        desired[0, :, 0] = 1
        desired_nodes = self.nodes.clone()
        desired_nodes[0, 1] = 5
        desired_nodes[1, -1] = 5
        desired_nodes[1, 0] = torch.arange(8 * 5, 9 * 5)
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        if not torch.all(adj == desired):
            self.fail(f"{adj} != {desired}")

        if not torch.all(nodes[0] == desired_nodes[0]):
            self.fail(f"{nodes[0]} != {desired_nodes[0]}")

        # It's shifted by one
        if not torch.all(nodes[1, 1] == desired_nodes[1, 2]):
            self.fail(f"{nodes[1,2]} != {desired_nodes[1,1]}")

        if not torch.all(nodes[1, 0] == desired_nodes[1, 0]):
            self.fail(f"{nodes[1,0]} != {desired_nodes[1,0]}")

        if not torch.all(nodes[1, -1] == desired_nodes[1, -1]):
            self.fail(f"{nodes[0]} != {desired_nodes[0]}")


class TestGCMDirection(unittest.TestCase):
    def setUp(self):
        torch.autograd.set_detect_anomaly(True)
        feats = 11
        batches = 1
        N = 10
        conv_type = torch_geometric.nn.DenseGraphConv
        self.g = torch_geometric.nn.Sequential(
            "x, adj, weights, B, N",
            [
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
            ],
        )
        for layer in list(self.g.modules())[1]:
            if layer.__class__ == conv_type:
                layer.lin_root.weight = torch.nn.Parameter(
                    torch.diag(torch.zeros(layer.lin_root.weight.shape[-1]))
                )
                layer.lin_root.bias = torch.nn.Parameter(
                    torch.zeros_like(layer.lin_root.bias)
                )
                layer.lin_rel.weight = torch.nn.Parameter(
                    torch.diag(torch.ones(layer.lin_root.weight.shape[-1]))
                )
        self.s = DenseGCM(self.g)

        self.nodes = torch.arange((batches * N * feats), dtype=torch.float).reshape(
            batches, N, feats
        )
        self.all_obs = [
            1 * torch.ones(batches, feats),
            2 * torch.ones(batches, feats),
            3 * torch.ones(batches, feats),
        ]
        self.adj = torch.zeros(batches, N, N)
        self.weights = torch.ones(batches, N, N)
        self.num_nodes = torch.zeros(batches, dtype=torch.long)

    def test_gcm_direction(self):
        # Only get neighbor
        self.adj[:, 0, 3] = 1
        # list(self.g.modules())[1][0].lin_rel(torch.matmul(self.adj, self.nodes))

        out, (nodes, adj, weights, num_nodes) = self.s(
            self.all_obs[0], (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        # neighbor
        # flows from 3 => 0, neighbor => root
        # root = i, neighbor = j
        # i should be > j in adj[i, j]
        desired = torch.arange(3 * 11, 4 * 11, dtype=torch.float)
        if not torch.all(self.nodes[0, 3] == desired):
            self.fail(f"{self.nodes[0,3]} != {desired}")


class TestDenseGCME2E(unittest.TestCase):
    def setUp(self):
        torch.autograd.set_detect_anomaly(True)
        feats = 11
        batches = 5
        N = 10
        conv_type = torch_geometric.nn.DenseGraphConv
        self.g = torch_geometric.nn.Sequential(
            "x, adj, weights, B, N",
            [
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
            ],
        )
        for layer in list(self.g.modules())[1]:
            if layer.__class__ == conv_type:
                layer.lin_root.weight = torch.nn.Parameter(
                    torch.diag(torch.ones(layer.lin_root.weight.shape[-1]))
                )
                layer.lin_root.bias = torch.nn.Parameter(
                    torch.zeros_like(layer.lin_root.bias)
                )
                layer.lin_rel.weight = torch.nn.Parameter(
                    torch.diag(torch.ones(layer.lin_root.weight.shape[-1]))
                )
        self.s = DenseGCM(self.g)

        self.nodes = torch.zeros((batches, N, feats), dtype=torch.float)
        self.all_obs = [
            1 * torch.ones(batches, feats),
            2 * torch.ones(batches, feats),
            3 * torch.ones(batches, feats),
        ]
        self.adj = torch.zeros(batches, N, N)
        self.weights = torch.ones(batches, N, N)
        self.num_nodes = torch.zeros(batches, dtype=torch.long)

    def test_e2e_self_edge(self):
        (nodes, adj, weights, num_nodes) = (
            self.nodes,
            self.adj,
            self.weights,
            self.num_nodes,
        )
        # First iter
        # Zeroth row of graph should be 1111...
        # Output should be 1111...
        obs = self.all_obs[0].clone()
        out, (nodes, adj, weights, num_nodes) = self.s(
            obs, (nodes, adj, weights, num_nodes)
        )
        if torch.any(out != self.all_obs[0]):
            self.fail(f"out: {out} != {self.all_obs[0]}")

        desired_nodes = torch.cat(
            (self.all_obs[0].unsqueeze(1), torch.zeros(5, 9, 11)), dim=1
        )
        if torch.any(nodes != desired_nodes):
            self.fail(f"out: {nodes} != {desired_nodes}")

        # Second iter
        # Rows 1111 and 2222...
        # Output should be 2222
        obs = self.all_obs[1].clone()
        out, (nodes, adj, weights, num_nodes) = self.s(
            obs, (nodes, adj, weights, num_nodes)
        )
        if torch.any(out != self.all_obs[1]):
            self.fail(f"out: {out} != {self.all_obs[1]}")

        # Third iter
        # Rows 1111 and 2222 and 3333...
        # Output should be 3333
        obs = self.all_obs[2].clone()
        out, (nodes, adj, weights, num_nodes) = self.s(
            obs, (nodes, adj, weights, num_nodes)
        )
        if torch.any(out != self.all_obs[2]):
            self.fail(f"out: {out} != {self.all_obs[1]}")


class TestDenseGCM(unittest.TestCase):
    def setUp(self):
        torch.autograd.set_detect_anomaly(True)
        feats = 11
        batches = 5
        N = 10
        conv_type = torch_geometric.nn.DenseGCNConv
        self.g = torch_geometric.nn.Sequential(
            "x, adj, weights, B, N",
            [
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
            ],
        )
        self.s = DenseGCM(self.g)

        # Now do it in a loop to make sure grads propagate
        self.optimizer = torch.optim.Adam(self.s.parameters(), lr=0.005)

        self.nodes = torch.arange(batches * N * feats, dtype=torch.float).reshape(
            batches, N, feats
        )
        self.obs = torch.ones(batches, feats)
        self.adj = torch.zeros(batches, N, N)
        self.weights = torch.ones(batches, N, N)
        self.num_nodes = torch.zeros(batches, dtype=torch.long)

    def test_grad_prop(self):
        self.g.grad_test_var = torch.nn.Parameter(torch.tensor([1.0]))
        self.nodes = self.nodes * self.g.grad_test_var
        self.assertTrue(self.nodes.requires_grad)
        out, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        loss = torch.norm(out)
        dot = torchviz.make_dot(loss, params=dict(self.s.named_parameters()))
        # Make sure gradients make it all the way thru node_feats
        self.assertTrue("grad_test_var" in dot.source)

    def test_no_weights(self):
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, torch.ones(0), self.num_nodes)
        )

        if len(weights) != 0:
            self.fail(f"Weights should be none, is {weights}") 

    def test_zeroth_entry(self):
        # Ensure first obs ends up in nodes matrix
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        if torch.any(nodes[:, 0] != self.obs):
            self.fail(f"{nodes[:,0]} != {self.obs}")
        # Ensure only self edges
        adj = torch.zeros_like(self.adj, dtype=torch.long)
        if torch.any(self.adj != adj):
            self.fail(f"adj: {adj} != {self.adj}")

    def test_first_entry(self):
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (nodes, adj, weights, num_nodes)
        )
        if torch.any(nodes[:, 1] != self.obs):
            self.fail(f"{nodes[:,0]} != {self.obs}")

    def test_num_nodes_entry(self):
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        B_idx = torch.arange(num_nodes.shape[0])
        if torch.any(nodes[B_idx, num_nodes - 1] != self.obs):
            self.fail(f"{nodes[:,num_nodes]} != {self.obs}")

    def test_propagation(self):
        out, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        if torch.all(out == self.obs):
            self.fail(f"{out} == {self.obs}")

    def test_dense_learn(self):
        feats = 11
        batches = 5
        T = 4
        N = 10
        losses = []
        for i in range(20):
            nodes = torch.arange(batches * N * feats, dtype=torch.float).reshape(
                batches, N, feats
            )
            obs = torch.ones(batches, feats)
            adj = torch.zeros(batches, N, N, dtype=torch.long)
            weights = torch.ones(batches, N, N)
            num_nodes = torch.zeros(batches, dtype=torch.long)

            self.s.zero_grad()
            hidden = (nodes, adj, weights, num_nodes)
            for t in range(T):
                obs, hidden = self.s(obs, hidden)

            loss = torch.norm(obs)
            loss.backward()
            losses.append(loss)

            self.optimizer.step()

        if not losses[-1] < losses[0]:
            self.fail(f"Final loss {losses[-1]} not better than init loss {losses[0]}")


class TestSparseGCM(unittest.TestCase):
    def setUp(self):
        feats = 11
        batches = 5
        N = 10
        conv_type = torch_geometric.nn.GCNConv
        self.g = torch_geometric.nn.Sequential(
            "x, adj, weights, B, N",
            [
                (DenseToSparse(), "x, adj, -> x_sp, edge_index, batch_idx"),
                (conv_type(feats, feats), "x_sp, edge_index -> x_sp"),
                (torch.nn.ReLU()),
                (conv_type(feats, feats), "x_sp, edge_index -> x_sp"),
                (torch.nn.ReLU()),
                (SparseToDense(), "x_sp, edge_index, batch_idx, B, N -> x, adj"),
                # Return only x not adj
                (lambda x: x, "x -> x"),
            ],
        )
        self.s = DenseGCM(self.g)

        # Now do it in a loop to make sure grads propagate
        self.optimizer = torch.optim.Adam(self.s.parameters(), lr=0.005)

        self.nodes = torch.arange(batches * N * feats, dtype=torch.float).reshape(
            batches, N, feats
        )
        self.obs = torch.ones(batches, feats)
        self.adj = torch.zeros(batches, N, N, dtype=torch.long)
        self.adj[:, 0:2, 3:4] = 1
        self.weights = torch.ones(batches, N, N)
        self.num_nodes = torch.zeros(batches, dtype=torch.long)

    def test_zeroth_entry(self):
        # Ensure first obs ends up in nodes matrix
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        if torch.any(nodes[:, 0] != self.obs):
            self.fail(f"{nodes[:,0]} != {self.obs}")

    def test_first_entry(self):
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (nodes, adj, weights, num_nodes)
        )
        if torch.any(nodes[:, 1] != self.obs):
            self.fail(f"{nodes[:,0]} != {self.obs}")

    def test_DenseToSparse_SparseToDense(self):
        B = self.nodes.shape[0]
        N = self.nodes.shape[1]

        x = self.nodes.clone()
        adj = self.adj.clone()
        weight = self.weights.clone()

        seq = torch_geometric.nn.Sequential(
            "x, adj, weights, B, N",
            [
                (DenseToSparse(), "x, adj -> x_sp, edge_index, batch_idx"),
                (SparseToDense(), "x_sp, edge_index, batch_idx, B, N -> x_d, adj_d"),
            ],
        )

        x_d, adj_d = seq(x, adj, weight, B, N)

        if torch.any(x_d != self.nodes):
            self.fail(f"x: {x_d} != {self.nodes}")

        if torch.any(adj_d != self.adj):
            self.fail(f"adj: {adj_d} != {self.adj}")

    def test_propagation(self):
        out, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        if torch.all(out == self.obs):
            self.fail(f"{out} == {self.obs}")

    def test_sparse_learn(self):
        feats = 11
        batches = 5
        T = 4
        N = 10
        losses = []
        for i in range(20):
            nodes = torch.arange(batches * N * feats, dtype=torch.float).reshape(
                batches, N, feats
            )
            obs = torch.ones(batches, feats)
            adj = torch.zeros(batches, N, N, dtype=torch.long)
            weights = torch.ones(batches, N, N)
            num_nodes = torch.zeros(batches, dtype=torch.long)

            self.s.zero_grad()
            hidden = (nodes, adj, weights, num_nodes)
            for t in range(T):
                obs, hidden = self.s(obs, hidden)

            loss = torch.norm(obs)
            loss.backward()
            losses.append(loss)

            self.optimizer.step()

        self.assertTrue(losses[-1] < losses[0])


class TestTemporalEdge(unittest.TestCase):
    def setUp(self):
        feats = 3
        batches = 2
        N = 10
        conv_type = torch_geometric.nn.DenseGCNConv
        self.g = torch_geometric.nn.Sequential(
            "x, adj, weights, B, N",
            [
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
            ],
        )
        self.s = DenseGCM(self.g, edge_selectors=TemporalBackedge(hops=[1]))

        # Now do it in a loop to make sure grads propagate
        self.optimizer = torch.optim.Adam(self.s.parameters(), lr=0.005)

        self.nodes = torch.arange(batches * N * feats, dtype=torch.float).reshape(
            batches, N, feats
        )
        self.obs = torch.ones(batches, feats)
        self.adj = torch.zeros(batches, N, N)
        self.weights = torch.ones(batches, N, N)
        self.num_nodes = torch.zeros(batches, dtype=torch.long)

    def test_two_nodes(self):
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (nodes, adj, weights, num_nodes)
        )
        tgt_adj = torch.zeros_like(adj, dtype=torch.long)
        # tgt_adj[:, 0, 1] = 1
        tgt_adj[:, 1, 0] = 1
        # Also add self edges
        if torch.any(tgt_adj != adj):
            self.fail(f"{tgt_adj} != {adj}")

    def test_far_hops(self):
        (nodes, adj, weights, num_nodes) = (
            self.nodes,
            self.adj,
            self.weights,
            self.num_nodes,
        )
        self.s = DenseGCM(self.g, edge_selectors=TemporalBackedge(hops=[4]))
        for i in range(10):
            _, (nodes, adj, weights, num_nodes) = self.s(
                self.obs, (nodes, adj, weights, num_nodes)
            )
        # hop 1 should start at t-1
        # hop 5 should start at t-5: 5=>0, 6=>1, etc
        tgt_adj = torch.zeros_like(adj)
        tgt_adj[:, 4, 0] = 1
        tgt_adj[:, 5, 1] = 1
        tgt_adj[:, 6, 2] = 1
        tgt_adj[:, 7, 3] = 1
        tgt_adj[:, 8, 4] = 1
        tgt_adj[:, 9, 5] = 1
        if torch.any(tgt_adj != adj):
            self.fail(f"{tgt_adj} != {adj}")

    def test_learned_edge_grad(self):
        self.s = DenseGCM(self.g, edge_selectors=TemporalBackedge(learned=True))
        self.num_nodes += 1
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        if not adj.requires_grad:
            self.fail("Adj has no grad")

        adj.sum().backward()



class TestDoubleEdge(unittest.TestCase):
    def setUp(self):
        feats = 11
        batches = 5
        N = 10
        conv_type = torch_geometric.nn.DenseGCNConv
        self.g = torch_geometric.nn.Sequential(
            "x, adj, weights, B, N",
            [
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
            ],
        )
        es = torch_geometric.nn.Sequential(
            "x, adj, weights, num_nodes, B",
            [
                (
                    TemporalBackedge([1]),
                    "x, adj, weights, num_nodes, B -> adj, weights"
                ),
                (
                    TemporalBackedge([2]),
                    "x, adj, weights, num_nodes, B -> adj, weights"
                ),
            ]
        )
        self.s = DenseGCM(
            self.g, 
            edge_selectors=es
        )

        self.nodes = torch.zeros(batches, N, feats, dtype=torch.float)
        self.obs = torch.zeros(batches, feats)
        self.adj = torch.zeros(batches, N, N, dtype=torch.long)
        self.weights = torch.ones(batches, N, N)
        self.num_nodes = torch.ones(batches, dtype=torch.long)
        self.optimizer = torch.optim.Adam(self.s.parameters(), lr=0.005)

    def test_backwards(self):
        nodes, adj, weights, num_nodes = (
            self.nodes,
            self.adj,
            self.weights,
            self.num_nodes + 1,
        )
        out, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (nodes, adj, weights, num_nodes)
        )
        self.s.edge_selectors.zero_grad()
        nodes = torch.rand_like(self.nodes) * torch.nn.Parameter(torch.tensor([0.01]))
        adj, weights = self.s.edge_selectors(nodes, adj, weights, num_nodes, 5)
        nodes.mean().backward()
        self.optimizer.step()

class TestDistanceEdge(unittest.TestCase):
    def setUp(self):
        feats = 11
        batches = 5
        N = 10
        conv_type = torch_geometric.nn.DenseGCNConv
        self.g = torch_geometric.nn.Sequential(
            "x, adj, weights, B, N",
            [
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
            ],
        )
        self.s = DenseGCM(self.g, edge_selectors=EuclideanEdge(max_distance=1))

        self.nodes = torch.zeros(batches, N, feats, dtype=torch.float)
        self.obs = torch.zeros(batches, feats)
        self.adj = torch.zeros(batches, N, N, dtype=torch.long)
        self.weights = torch.ones(batches, N, N)
        self.num_nodes = torch.ones(batches, dtype=torch.long)

    def test_zero_dist(self):
        # Start num_nodes = 1
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        tgt_adj = torch.zeros_like(adj, dtype=torch.long)
        tgt_adj[:, 1, 0] = 1

        # TODO: Ensure not off by one
        if torch.any(tgt_adj != adj):
            self.fail(f"{tgt_adj} != {self.adj}")

    def test_one_dist(self):
        # Start num_nodes = 1
        self.obs = torch.ones_like(self.obs)
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        tgt_adj = torch.zeros_like(adj, dtype=torch.long)
        # Adds self edge
        if torch.any(tgt_adj != adj):
            self.fail(f"{tgt_adj} != {self.adj}")

    def test_cosine(self):
        self.s = DenseGCM(self.g, edge_selectors=CosineEdge(max_distance=1))
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )

    def test_learned_edge(self):
        self.s = DenseGCM(self.g, edge_selectors=EuclideanEdge(max_distance=1, learned=True))
        self.obs = torch.ones_like(self.obs)

        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        tgt_adj = torch.zeros_like(adj, dtype=torch.long)
        # Adds self edge
        if torch.any(tgt_adj != adj):
            self.fail(f"{tgt_adj} != {self.adj}")

    def test_learned_edge_grad(self):
        e = EuclideanEdge(max_distance=1, learned=True)
        self.num_nodes += 2

        adj, weights = e(self.nodes, self.adj, self.weights, self.num_nodes, B=5)

        if not self.nodes.requires_grad:
            self.fail(f"Nodes has no grad")



class TestDenseEdge(unittest.TestCase):
    def setUp(self):
        feats = 11
        batches = 5
        N = 10
        conv_type = torch_geometric.nn.DenseGCNConv
        self.g = torch_geometric.nn.Sequential(
            "x, adj, weights, B, N",
            [
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
            ],
        )
        self.s = DenseGCM(self.g, edge_selectors=DenseEdge())

        self.nodes = torch.zeros(batches, N, feats, dtype=torch.float)
        self.obs = torch.zeros(batches, feats)
        self.adj = torch.zeros(batches, N, N, dtype=torch.long)
        self.weights = torch.ones(batches, N, N)
        self.num_nodes = torch.zeros(batches, dtype=torch.long)

    def test_two_obs(self):
        # Start num_nodes = 1
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (nodes, adj, weights, num_nodes)
        )
        tgt_adj = torch.zeros_like(adj, dtype=torch.long)
        # It adds self edge
        tgt_adj[:, 1, 1] = 1
        tgt_adj[:, 0, 0] = 1
        tgt_adj[:, 1, 0] = 1
        tgt_adj[:, 0, 1] = 1

        # TODO: Ensure not off by one
        if torch.any(tgt_adj != adj):
            self.fail(f"{tgt_adj} != {self.adj}")


class Sum(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        # Returns sum of the first feature of base and neighbor nodes
        return x[:, 0] + x[:, 5]


class TestBernoulliEdge(unittest.TestCase):
    def setUp(self):
        torch.autograd.set_detect_anomaly(True)
        feats = 5
        batches = 2
        N = 4
        conv_type = torch_geometric.nn.DenseGCNConv
        self.g = torch_geometric.nn.Sequential(
            "x, adj, weights, B, N",
            [
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
            ],
        )
        self.s = DenseGCM(self.g, edge_selectors=BernoulliEdge(feats))

        # Now do it in a loop to make sure grads propagate
        self.optimizer = torch.optim.Adam(self.s.parameters(), lr=0.005)

        self.nodes = torch.arange(batches * N * feats, dtype=torch.float).reshape(
            batches, N, feats
        )
        self.obs = torch.ones(batches, feats)
        self.adj = torch.zeros(batches, N, N)
        self.weights = torch.ones(batches, N, N)
        self.num_nodes = torch.zeros(batches, dtype=torch.long)

    def test_compute_new_adj_deterministic_grad(self):
        self.num_nodes = torch.tensor((2,3), dtype=torch.long)
        es = BernoulliEdge(5, deterministic=True)
        self.nodes.requires_grad=True
        adj, weights = es(self.nodes, self.adj, self.weights, self.num_nodes, 2)

        self.assertTrue(adj.requires_grad)

    def test_compute_new_adj_grad(self):
        p = torch.nn.Parameter(torch.tensor([1.0]))
        self.nodes = self.nodes * p
        self.num_nodes = torch.tensor((2,3), dtype=torch.long)
        es = BernoulliEdge(5, deterministic=False)
        adj, weights = es(self.nodes, self.adj, self.weights, self.num_nodes, 2)

        self.assertTrue(adj.requires_grad)
        optimizer = torch.optim.Adam(es.parameters(), lr=0.005)
        self.nodes.sum().backward()
        optimizer.step()
        grads = [p.grad for p in es.parameters()]

    def test_diff_or(self):
        test = [(torch.rand(32,128,128) > 0.5).float() for i in range(3)]
        desired = (sum(test) >= 1).float()
        actual = diff_or(test)
        if not torch.all(actual == desired):
            self.fail(f"Desired {desired} actual {actual}")

        actual = diff_or2(test)
        if not torch.all(actual == desired):
            self.fail(f"Desired {desired} actual {actual}")



    """
    def test_update_density(self):
        self.b = BernoulliEdge(5, torch.nn.Sequential(Sum()))
        a = torch.tensor([1.5, 2, 0, 0, 2, 0, 3, 1, 0.2])
        b = torch.tensor([1, 4, 2, 0.1])
        self.b.update_density(a)
        self.b.update_density(b)
        rm = self.b.detach_loss()
        self.assertTrue(torch.isclose(rm, torch.cat((a, b)).mean()))
    """

    def test_sample_hard(self):
        self.b = BernoulliEdge(5, torch.nn.Sequential())
        self.num_nodes = torch.tensor([0,1,2])
        self.weights = torch.zeros_like(self.weights)
        self.weights[0, 0, 0] = 1e6

        self.weights[1, 1, 1] = 1e6
        self.weights[1, 0, 1] = 1e6

        # Test valid weights correspond to adj
        desired = self.adj.clone()
        # Ensure diagonal (0,0) is set to 0
        desired[0, 0, 0] = 0.0
        desired[1, 1, 1] = 0.0
        # Ensure large, nondiagonal is nonzero
        desired[1, 0, 1] = 1.0

        # b_idxs, curr_idx, past_idxs = ([0, 0, 1, 1], [2, 2, 2, 2], [0, 1, 0, 1])
        adj = sample_hard(self.adj, self.weights, self.num_nodes)
        if torch.any(desired != adj):
            self.fail(f"{desired} != {adj}")


    '''
    def test_weight_to_adj(self):
        self.b = BernoulliEdge(5, torch.nn.Sequential(Sum()))
        self.weights = torch.zeros_like(self.weights)
        self.weights[0, 2] = 1.0
        self.weights[1, 2] = 1.0
        self.weights = self.weights.clamp(*self.b.clamp_range)

        desired = self.adj.clone()
        desired[0, 2] = 1.0
        desired[1, 2] = 1.0

        # b_idxs, curr_idx, past_idxs = ([0, 0, 1, 1], [2, 2, 2, 2], [0, 1, 0, 1])
        self.adj = sample_hard(self.weights, self.b.clamp_range)
        """
        self.adj[b_idxs, curr_idx, past_idxs] = sample_hard(
            self.weights[b_idxs, curr_idx, past_idxs], self.b.clamp_range
        )
        """
        if torch.any(desired != self.adj):
            self.fail(f"{desired} != {self.adj}")
    '''

    def test_indexing(self):
        self.b = BernoulliEdge(5, torch.nn.Sequential(Sum()))

        self.s = DenseGCM(self.g, edge_selectors=self.b)
        self.all_obs = [
            torch.ones_like(self.obs) * 0.1,
            torch.ones_like(self.obs) * 0.2,
            torch.ones_like(self.obs) * 0.3,
        ]
        self.weights = torch.zeros_like(self.weights)

        (nodes, adj, weights, num_nodes) = (
            self.nodes,
            self.adj,
            self.weights.clone(),
            self.num_nodes,
        )
        for i in range(3):
            out, (nodes, adj, weights, num_nodes) = self.s(
                self.all_obs[i], (nodes, adj, weights, num_nodes)
            )

        # 0: skip
        # 1: cat(0.2, 0.1) = 0.3
        # 2: cat(0.3, 0.1) = 0.4, cat(0.3, 0.2) = 0.5

        # 0 -> []
        # 1 -> [1,0]
        # 2 -> [2,0], [2,1] # does this order matter?
        # Network input is __ascending__ e.g. [3,0], [3,1], [3,2]
        # so
        desired = self.weights.clone()
        desired[:, 1, 0] = 0.3
        desired[:, 2, 0] = 0.4
        desired[:, 2, 1] = 0.5
        if torch.any(desired != weights):
            self.fail(f"{desired} != {weights}")
        # b <- a should sum to 6

    def test_grad_prop(self):
        self.g.grad_test_var = torch.nn.Parameter(torch.tensor([1.0]))
        self.nodes = self.nodes * self.g.grad_test_var
        self.assertTrue(self.nodes.requires_grad)

        out, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        # First run has no gradient as no edges to be made
        out, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (nodes, adj, weights, num_nodes)
        )

        loss = torch.norm(out)
        dot = torchviz.make_dot(loss, params=dict(self.s.named_parameters()))
        # Make sure gradients make it all the way thru node_feats
        self.assertTrue("grad_test_var" in dot.source)

    def test_grad_prop2(self):
        out, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        # First run has no gradient as no edges to be made
        out, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (nodes, adj, weights, num_nodes)
        )
        adj, weights = self.s.edge_selectors(nodes, adj, weights, num_nodes, 5)
        self.assertTrue(adj.grad_fn, "Adj has no gradient")
        self.assertTrue(weights.grad_fn, "Weight has no gradient")

    def test_backward_multiple_selectors(self):
        selector = torch_geometric.nn.Sequential(
            "nodes, adj, weights, num_nodes, B",
            [
                (
                    TemporalBackedge(), 
                    "nodes, adj, weights, num_nodes, B -> adj, weights"
                ),
                (
                    BernoulliEdge(5),
                    "nodes, adj, weights, num_nodes, B -> adj, weights"
                ),
            ]
        )
        self.s = DenseGCM(
            self.g, 
            edge_selectors=selector
        )
        out, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        # First run has no gradient as no edges to be made
        out, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (nodes, adj, weights, num_nodes)
        )
        adj, weights = self.s.edge_selectors(nodes, adj, weights, num_nodes, 5)
        self.assertTrue(adj.grad_fn, "Adj has no gradient")
        self.assertTrue(weights.grad_fn, "Weight has no gradient")
        adj.mean().backward()
        self.optimizer.step()

    def test_backwards(self):
        nodes, adj, weights, num_nodes = (
            self.nodes,
            self.adj,
            self.weights,
            self.num_nodes + 1,
        )
        out, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (nodes, adj, weights, num_nodes)
        )
        self.s.edge_selectors.zero_grad()
        nodes = torch.rand_like(self.nodes) * 0.00001
        adj, weights = self.s.edge_selectors(nodes, adj, weights, num_nodes, 5)
        adj.mean().backward()
        self.optimizer.step()

    """
    def test_reg_loss(self):
        feats = 5
        batches = 2
        T = 4
        N = 10
        losses = []
        for i in range(20):
            nodes = torch.arange(batches * N * feats, dtype=torch.float).reshape(
                batches, N, feats
            )
            obs = torch.ones(batches, feats)
            adj = torch.zeros(batches, N, N)
            weights = torch.ones(batches, N, N)
            num_nodes = torch.zeros(batches, dtype=torch.long)

            if i == 0:
                continue

            hidden = (nodes, adj, weights, num_nodes)
            for t in range(T):
                obs, hidden = self.s(obs, hidden)

            loss = self.s.edge_selectors.compute_full_loss(nodes, nodes.shape[0])
            loss.backward()
            losses.append(loss)

            self.optimizer.step()
            # Must zero grad AND reset density
            self.optimizer.zero_grad()

        if not losses[-1] < losses[0]:
            self.fail(f"Final loss {losses[-1]} not better than init loss {losses[0]}")
        """

    def test_logit_index(self):
        # Given 3 nodes, make sure we compare node 3 to nodes 1,2
        out, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        # First run has no gradient as no edges to be made
        out, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (nodes, adj, weights, num_nodes)
        )
        adj, weights = self.s.edge_selectors(nodes, adj, weights, num_nodes, 5)
        self.assertTrue(adj.grad_fn, "Adj has no gradient")
        self.assertTrue(weights.grad_fn, "Weight has no gradient")

    def test_validate_logits(self):
        # TODO: Fix test
        return
        nodes = self.nodes
        adj = self.adj
        weights = self.weights
        num_nodes = self.num_nodes
        adj, weights = self.s.edge_selectors.compute_logits(
            nodes, weights.clone(), num_nodes, 5
        )
        adj2, weights2 = self.s.edge_selectors.compute_logits2(
            nodes, weights.clone(), num_nodes, 5
        )
        if torch.any(adj != adj2):
            self.fail(f"{adj} != {adj2}")

        if torch.any(weights != weights):
            self.fail(f"{weights} != {weights2}")


class TestSpatialEdge(unittest.TestCase):
    def setUp(self):
        feats = 11
        batches = 5
        N = 10
        conv_type = torch_geometric.nn.DenseGCNConv
        self.g = torch_geometric.nn.Sequential(
            "x, adj, weights, B, N",
            [
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
                (conv_type(feats, feats), "x, adj -> x"),
                (torch.nn.ReLU()),
            ],
        )
        self.slice = slice(0, 2)
        self.s = DenseGCM(self.g, edge_selectors=SpatialEdge(1, self.slice))

        self.nodes = torch.zeros(batches, N, feats, dtype=torch.float)
        self.obs = torch.zeros(batches, feats)
        self.adj = torch.zeros(batches, N, N, dtype=torch.float)
        self.weights = torch.ones(batches, N, N)
        self.num_nodes = torch.ones(batches, dtype=torch.long)

    def test_zero_dist(self):
        # Start num_nodes = 1
        self.nodes[:] = 1
        self.nodes[:, 0:2, self.slice] = 0
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        tgt_adj = torch.zeros_like(adj, dtype=torch.long)
        B_idx = torch.arange(num_nodes.shape[0])
        # It adds self edge
        # tgt_adj[:, 0, 1] = 1
        # tgt_adj[:, 1, 0] = 1
        tgt_adj[B_idx, num_nodes[B_idx] - 1, 0] = 1

        B_idx = torch.arange(num_nodes.shape[0])

        # TODO: Ensure not off by one
        if torch.any(tgt_adj != adj):
            self.fail(f"{tgt_adj} != {adj}")

    def test_one_dist(self):
        # Start num_nodes = 1
        self.nodes[:, 0, self.slice] = 1
        _, (nodes, adj, weights, num_nodes) = self.s(
            self.obs, (self.nodes, self.adj, self.weights, self.num_nodes)
        )
        tgt_adj = torch.zeros_like(adj, dtype=torch.long)
        # It adds self edge

        # TODO: Ensure not off by one
        if torch.any(tgt_adj != adj):
            self.fail(f"{tgt_adj} != {adj}")


if __name__ == "__main__":
    unittest.main()
