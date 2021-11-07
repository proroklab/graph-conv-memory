import unittest
import math
import torch
import torch_geometric
import torchviz
from collections import OrderedDict

from gcm.gcm import DenseGCM, DenseToSparse, SparseToDense, PositionalEncoding
from gcm.edge_selectors.temporal import TemporalBackedge
from gcm.edge_selectors.distance import EuclideanEdge, CosineEdge, SpatialEdge
from gcm.edge_selectors.dense import DenseEdge
from gcm.edge_selectors.learned import LearnedEdge
from gcm import util
from gcm.gcm import DenseGCM, DenseToSparse, SparseToDense, PositionalEncoding
from gcm.sparse_gcm import SparseGCM


class TestPack(unittest.TestCase):
    def setUp(self):
        self.F = 4
        self.B = 3
        self.T = torch.tensor([2,3,0])
        self.graph_size = 5
        self.max_edges = 10

    def test_unpack_pack(self):
        nodes = torch.zeros(self.B, self.graph_size, self.F)

        dense_edge = torch.empty(self.B, 2, self.max_edges, dtype=torch.long).fill_(-1)
        dense_edge[0,:,0] = torch.tensor([0,1])
        dense_edge[1,:,0] = torch.tensor([0,1])
        dense_edge[1,:,1] = torch.tensor([1,2])

        dense_weight = torch.empty(self.B, 1, self.max_edges).fill_(1.0)
        dense_weight[0,0,0] = 0.5
        dense_weight[1,0,1] = 0.25
        initial_packed_hidden = (nodes.clone(), dense_edge.clone(), dense_weight.clone(), self.T.clone())

        packed_hidden = (nodes, dense_edge, dense_weight, self.T)
        unpacked_hidden = util.unpack_hidden(packed_hidden, self.B)
        repacked_hidden = util.pack_hidden(unpacked_hidden, self.B, self.max_edges)

        for i in range(len(initial_packed_hidden)):
            if not (initial_packed_hidden[i] == repacked_hidden[i]).all():
                self.fail(
                    f'packed hidden tensor {i} != repacked hidden tensor'
                    '{initial_packed_hidden[i]) != {repacked_hidden[i]}'
                )




class TestE2ENonRagged(unittest.TestCase):
    def setUp(self):
        self.F = 3
        dense_conv_type = torch_geometric.nn.DenseGraphConv
        sparse_conv_type = torch_geometric.nn.GraphConv
        self.dense_g = torch_geometric.nn.Sequential(
            "x, adj, weights, B, N",
            [
                (torch.nn.ReLU(), 'x -> x'),
                #(dense_conv_type(self.F, self.F), "x, adj -> x"),
            ],
        )
        self.sparse_g = torch_geometric.nn.Sequential(
            "x, edges, weights",
            [
                (torch.nn.ReLU(), 'x -> x')
                #(sparse_conv_type(self.F, self.F), "x, edges -> x"),
            ],
        )
        dense_params = self.dense_g.state_dict()
        old_sparse_params = self.sparse_g.state_dict()
        new_sparse_params = OrderedDict()
        for k, v in dense_params.items():
            if 'root' in k:
                new_sparse_params[k.replace('root', 'l')] = v
            elif 'rel' in k:
                new_sparse_params[k.replace('rel', 'r')] = v
            else:
                new_sparse_params[k] = v
        self.sparse_g.load_state_dict(new_sparse_params)

        # sanity check
        for k in self.sparse_g.state_dict():
            self.assertTrue((self.sparse_g.state_dict()[k] == new_sparse_params[k]).all())
        sparse_out = self.sparse_g(torch.ones(2,3,self.F), torch.ones(2,0, dtype=torch.long), torch.ones(0, dtype=torch.long))
        dense_out = self.dense_g(torch.ones(2,3,self.F), torch.zeros(2,3,3), torch.zeros(2,3,3), None, None)

        self.assertTrue((sparse_out == dense_out).all())


        self.dense_gcm = DenseGCM(self.dense_g, graph_size=5)
        self.sparse_gcm = SparseGCM(self.sparse_g, graph_size=5)

    def test_no_edges(self):
        F = self.F
        B = 2
        N = 5
        ts = 4 
        self.obs = torch.arange(B * ts * F, dtype=torch.float32).reshape(B, ts, F)

        dense_outs = []

        hidden = None
        for i in range(ts):
            dense_out, hidden = self.dense_gcm(self.obs[:,i], hidden)
            dense_outs.append(dense_out)
        dense_outs = torch.stack(dense_outs, dim=1)

        taus = torch.ones(B, dtype=torch.long) * ts
        sparse_outs, sparse_hidden = self.sparse_gcm(self.obs, taus, None)

        if dense_outs.numel() != sparse_outs.numel():
            self.fail(f"sizes {dense_outs.numel()} != {sparse_outs.numel()}")

        # Check hiddens
        if not torch.all(hidden[0] == sparse_hidden[0]):
            self.fail(f"{hidden[0]} != {sparse_hidden[0]}")

        if not torch.all(dense_outs.flatten() == sparse_outs.flatten()):
            self.fail(f"{dense_outs} != {sparse_outs}")

    def test_temporal_edges(self):
        F = self.F
        B = 2
        N = 5
        ts = 4 
        self.obs = torch.arange(B * ts * F, dtype=torch.float32).reshape(B, ts, F)

        dense_outs = []

        hidden = None
        for i in range(ts):
            dense_out, hidden = self.dense_gcm(self.obs[:,i], hidden)
            dense_outs.append(dense_out)
        dense_outs = torch.stack(dense_outs, dim=1)

        taus = torch.ones(B, dtype=torch.long) * ts
        sparse_outs, sparse_hidden = self.sparse_gcm(self.obs, taus, None)

        if dense_outs.numel() != sparse_outs.numel():
            self.fail(f"sizes {dense_outs.numel()} != {sparse_outs.numel()}")

        # Check hiddens
        if not torch.all(hidden[0] == sparse_hidden[0]):
            self.fail(f"{hidden[0]} != {sparse_hidden[0]}")

        if not torch.all(dense_outs.flatten() == sparse_outs.flatten()):
            self.fail(f"{dense_outs} != {sparse_outs}")

        


if __name__ == "__main__":
    unittest.main()
