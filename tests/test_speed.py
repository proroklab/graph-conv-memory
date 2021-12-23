import torch
import torch_geometric
from gcm.gcm import DenseGCM
from gcm.sparse_gcm import SparseGCM
from gcm.edge_selectors.dense import DenseEdge
from gcm.sparse_edge_selectors.temporal import TemporalEdge
import time



lstm = torch.nn.LSTMCell(32,32)
g = torch_geometric.nn.Sequential(
    "x, adj, weights, B, N",
    [
        (torch_geometric.nn.DenseGraphConv(32, 32), "x, adj -> x"),
        (torch.nn.Tanh()),
        (torch_geometric.nn.DenseGraphConv(32, 32), "x, adj -> x"),
        (torch.nn.Tanh()),
    ],
)

num_obs = 16
gcm = DenseGCM(
    g,
    edge_selectors=DenseEdge(),
    graph_size=num_obs
)

sparse_g = torch_geometric.nn.Sequential(
    "x, edges, weights",
    [
        (torch_geometric.nn.GraphConv(32, 32), "x, edges, weights -> x"),
        (torch.nn.Tanh()),
        (torch_geometric.nn.GraphConv(32, 32), "x, edges, weights -> x"),
        (torch.nn.Tanh()),
    ],
)
sparse_gcm = SparseGCM(sparse_g, graph_size=num_obs, edge_selectors=TemporalEdge([1,2]))



obs = torch.rand(num_obs, 32)

lstm_s = time.time()
hidden = None
for i in range(num_obs):
    hidden = lstm(obs[i].unsqueeze(0), hidden)
hidden[0].mean().backward()
print("lstm took", time.time() - lstm_s)

gcm_s = time.time()
hidden = None
for i in range(num_obs):
    out, hidden = gcm(obs[i].unsqueeze(0), hidden)
out.mean().backward()
print("gcm took", time.time() - gcm_s)

gcms_s = time.time()
hidden = None
taus = torch.tensor([num_obs])
out, hidden = sparse_gcm(obs.unsqueeze(0), taus, hidden)
out.mean().backward()
print("sparse gcm took", time.time() - gcms_s)

