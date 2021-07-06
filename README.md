# Graph Convolution Memory for Reinforcement Learning

## Description
Graph convolutional memory (GCM) is graph-structured memory that may be applied to reinforcement learning to solve POMDPs, replacing LSTMs or attention mechanisms. GCM allows you to embed your domain knowledge in the form of connections in a knowledge graph. See the [full paper](https://arxiv.org/pdf/2106.14117.pdf) for further details. This repo contains the GCM library implementation for use in your projects. To replicate the experiments from the paper, please see [this repository instead](https://github.com/smorad/graph-conv-memory-paper)


## Installation
GCM is installed using `pip`. The dependencies must be installed manually, as they target your specific architecture (with or without CUDA).

### Conda install
First install `torch >= 1.8.0` and `torch-geometric` dependencies, then `gcm`:
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


# graph_size denotes the maximum number of observations in the graph, after which
# the oldest observations will be overwritten with newer observations. Reduce this number to
# reduce memory usage.
graph_size = 128
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
# Create the GCM using our GNN and edge selection criteria. TemporalBackedge([1]) will link observation o_t to o_{t-1}.
# See `gcm.edge_selectors` for different kinds of priors suitable for your specific problem. Do not be afraid to implement your own!
gcm = DenseGCM(our_gnn, edge_selectors=TemporalBackedge([1]), graph_size=graph_size)

# Create initial state
# Shape: (batch_size, graph_size, graph_size)
edges = torch.zeros(
    (1, graph_size, graph_size), dtype=torch.float
)
# Shape: (batch_size, graph_size, obs_size)
nodes = torch.zeros((1, graph_size, obs_size))
# Shape: (batch_size, graph_size, graph_size)
weights = torch.zeros(
    (1, graph_size, graph_size), dtype=torch.float
)
# Shape: (batch_size)
num_nodes = torch.tensor([0], dtype=torch.long)
# Our memory state (m_t in the paper)
m_t = [nodes, edges, weights, num_nodes]

for t in train_timestep:
   # Obs at timestep t should be of shape (batch_size, obs_size)
   belief, m_t = gcm(obs[t], m_t)
   # GCM provides a belief state -- a combination of all past observational data relevant to the problem
   # What you likely want to do is put this state through actor and critic networks to obtain
   # action and value estimates
   action_logits = logits_nn(belief)
   state_value = vf_nn(belief)
```
