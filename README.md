# Graph Convolution Memory for Reinforcement Learning

## Description
Graph convolutional memory (GCM) is graph-structured memory that may be applied to reinforcement learning to solve POMDPs, replacing LSTMs or attention mechanisms.

## Quickstart
If you are interested in apply GCM for your problem, you must install dependencies `torch` and `torch_geometric`. You may `pip install gcm` to install the most recent versions of gcm, torch, and torch_geometric. However, if you want to use CUDA or other accelerators we suggest you install torch_geometric by hand before running `pip install gcm`, specifying your current CUDA version. Below is a quick example of how to use GCM in a basic RL problem:

```
import torch
import torch_geometric
from gcm import DenseGCM
from gcm.edge_selectors.temporal import TemporalBackedge


our_gnn = torch_geometric.nn.Sequential(
    "x, adj, weights, B, N",
    [
        (torch_geometric.nn.DenseGraphConv(YOUR_OBS_SIZE, 32), "x, adj -> x"),
        (torch.nn.Tanh()),
        (torch_geometric.nn.DenseGraphConv(32, 32), "x, adj -> x"),
        (torch.nn.Tanh()),
    ],
)
gcm = DenseGCM(our_gnn, edge_selectors=TemporalBackedge([1]), graph_size=128)

# Create initial state
edges = torch.zeros(
    (1, 128, 128), dtype=torch.float
)
nodes = torch.zeros((1, 128, YOUR_OBS_SIZE))
weights = torch.zeros(
    (1, 128, 128), dtype=torch.float
)
num_nodes = torch.tensor([0], dtype=torch.long)
m_t = [nodes, edges, weights, num_nodes]

for t in train_timestep:
   state, m_t = gcm(obs[t], m_t)
   # Do what you will with the state
   # likely you want to use it to get action/value estimate
   action_logits = logits(state)
   state_value = vf(state)
```
See `gcm.edge_selectors` for different kinds of priors.
