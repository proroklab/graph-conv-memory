# Graph Convolution Memory for Reinforcement Learning

## Description
Graph convolutional memory (GCM) is graph-structured memory that may be applied to reinforcement learning to solve POMDPs, replacing LSTMs or attention mechanisms. GCM allows you to embed your domain knowledge in the form of connections in a knowledge graph. See the full paper for further details.


## Installation
GCM is installed using `pip`. The dependencies must be installed manually, as they target your specific architecture (with or without CUDA).

### Conda install
First install `torch >= 1.8.0` and `torch-geometric` dependencies, then `gcm`
```
conda install torch
conda install pytorch-geometric -c rusty1s -c conda-forge
pip install graph-conv-memory
```

### Pip install
Please follow the [torch-geometric install guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), then
```
pip install graph-conv-memory
```


## Quickstart
Below is a quick example of how to use GCM in a basic RL problem:

```
import torch
import torch_geometric
from gcm.gcm import DenseGCM
from gcm.edge_selectors.temporal import TemporalBackedge


# Define the GNN used in GCM. The following is the one used in the paper
# Make sure you define the first layer to match your observation space
obs_size = 8
our_gnn = torch_geometric.nn.Sequential(
    "x, adj, weights, B, N",
    [
        (torch_geometric.nn.DenseGraphConv(obs_size, 32), "x, adj -> x"),
        (torch.nn.Tanh()),
        (torch_geometric.nn.DenseGraphConv(32, 32), "x, adj -> x"),
        (torch.nn.Tanh()),
    ],
)
# graph_size denotes the maximum number of observations in the graph, after which
# the oldest observations will be overwritten
gcm = DenseGCM(our_gnn, edge_selectors=TemporalBackedge([1]), graph_size=128)

# Create initial state
edges = torch.zeros(
    (1, 128, 128), dtype=torch.float
)
nodes = torch.zeros((1, 128, obs_size))
weights = torch.zeros(
    (1, 128, 128), dtype=torch.float
)
num_nodes = torch.tensor([0], dtype=torch.long)
# Our memory state
m_t = [nodes, edges, weights, num_nodes]

for t in train_timestep:
   state, m_t = gcm(obs[t], m_t)
   # Do what you will with the state
   # likely you want to use it to get action/value estimate
   action_logits = logits(state)
   state_value = vf(state)
```
See `gcm.edge_selectors` for different kinds of priors suitable to your specific problem. Do not be afraid to implement your own!
