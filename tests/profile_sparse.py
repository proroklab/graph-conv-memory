import torch
import torch_geometric
from gcm.sparse_gcm import SparseGCM
from gcm.sparse_edge_selectors.learned import LearnedEdge
from gcm import util
import cProfile, pstats, io
from pstats import SortKey


sparse_g = torch_geometric.nn.Sequential(
    "x, edges, weights",
    [
        (torch_geometric.nn.GraphConv(32, 32), "x, edges, weights -> x"),
        (torch.nn.Tanh()),
        (torch_geometric.nn.GraphConv(32, 32), "x, edges, weights -> x"),
        (torch.nn.Tanh()),
    ],
)



def fn():
    B = 8
    num_obs = 256
    obs_size = 32
    sparse_gcm = SparseGCM(
        sparse_g, graph_size=num_obs, edge_selectors=LearnedEdge(obs_size)
    )
    obs = torch.rand(B, num_obs, obs_size)
    taus = torch.ones(B, dtype=torch.long)
    hidden = None
    #with cProfile.Profile() as pr:
    with torch.profiler.profile(with_stack=True, profile_memory=True) as p:
        # inference
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

    print(p.key_averages(group_by_stack_n=3).table(sort_by="self_cpu_time_total", row_limit=10))
    #pr.print_stats(sort="cumtime")

fn()

