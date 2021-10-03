import torch
import torch_geometric
from gcm.gcm import DenseGCM, DenseToSparse, SparseToDense, PositionalEncoding
from gcm.sparse_gcm import SparseGCM
from gcm.edge_selectors.temporal import TemporalBackedge
import time

# Test learn speed
feats = 32
batches = 4
N = 128
tau = 20


def dense():
    conv_type = torch_geometric.nn.DenseGraphConv
    dg = torch_geometric.nn.Sequential(
        "x, adj, weights, B, N",
        [
            (conv_type(feats, feats), "x, adj -> x"),
            (torch.nn.ReLU()),
            (conv_type(feats, feats), "x, adj -> x"),
            (torch.nn.ReLU()),
        ],
    )
    dgcm = DenseGCM(dg)
    obs = torch.ones(batches, feats)
    adj = torch.zeros(batches, N, N)
    #adj[:,:,torch.arange(0,N-2), torch.arange(1,N-1)] = 1
    weights = torch.ones(batches, N, N)
    num_nodes = torch.zeros(batches, dtype=torch.long)

    s = time.time()
    hidden = None
    out = None
    l = torch.tensor([0.])
    for i in range(tau):
        out, hidden = dgcm(obs, hidden)
        l += out.sum()
    l.backward()
    e = time.time()
    print(f"Dense took {e - s}s")

def sparse():
    conv_type = torch_geometric.nn.GraphConv
    sg = torch_geometric.nn.Sequential(
        "x, edge_index",
        [
            (conv_type(feats, feats), "x, edge_index -> x"),
            (torch.nn.ReLU()),
            (conv_type(feats, feats), "x, edge_index -> x"),
            (torch.nn.ReLU()),
        ],
    )
    dgcm = SparseGCM(sg)
    obs = torch.ones(batches, feats)
    weights = torch.ones(batches, N, N)
    num_nodes = torch.zeros(batches, dtype=torch.long)
    nodes = torch.arange(batches * N * feats, dtype=torch.float).reshape(
        batches, N, feats
    )
    #edge_list = torch.stack((torch.arange(0,126), torch.arange(1,127)))
    obs_stack = obs.unsqueeze(1).repeat(1,tau,1)
    taus = torch.ones(batches, dtype=torch.long) * tau
    s = time.time()
    hidden = None
    out = None
    l = torch.tensor([0.])
    out, hidden = dgcm(obs_stack, taus, hidden)
    l += out.sum()
    l.backward()
    e = time.time()
    print(f"Sparse took {e - s}s")

for i in range(3):
    dense()
    sparse()
