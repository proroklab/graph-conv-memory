# Graph Convolutional Memory using Topological Priors

## Description
Graph convolutional memory (GCM) is graph-structured memory that may be applied to reinforcement learning to solve POMDPs, replacing LSTMs or attention mechanisms. GCM allows you to embed your domain knowledge in the form of connections in a graph. See the [full paper](https://arxiv.org/pdf/2106.14117.pdf) for further details. This repo contains the GCM library implementation for use in your projects. To replicate the experiments from the paper, please see [this repository instead](https://github.com/smorad/graph-conv-memory-paper).

If you use GCM, please cite the paper!
```
@article{morad2021graph,
  title={Graph Convolutional Memory for Deep Reinforcement Learning},
  author={Morad, Steven D and Liwicki, Stephan and Prorok, Amanda},
  journal={arXiv preprint arXiv:2106.14117},
  year={2021}
}
```


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

```python
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
class GNN(torch.nn.Module):
    """A simple two-layer graph neural network"""
    def __init__(self, obs_size, hidden_size=32):
        super().__init__()
        self.gc0 = torch_geometric.nn.DenseGraphConv(obs_size, hidden_size)
        self.gc1 = torch_geometric.nn.DenseGraphConv(hidden_size, hidden_size)
        self.act = torch.nn.Tanh()

    def forward(self, x, adj, weights, B, N):
        x = self.act(self.gc0(x, adj))
        return self.act(self.gc1(x, adj))

# Build GNN that GCM uses internally
obs_size = 8
gnn = GNN(obs_size)
# Create the GCM using our GNN and edge selection criteria. TemporalBackedge([1]) will link observation o_t to o_{t-1}.
# See `gcm.edge_selectors` for different kinds of priors suitable for your specific problem. Do not be afraid to implement your own!
gcm = DenseGCM(gnn, edge_selectors=TemporalBackedge([1]), graph_size=graph_size)

# If the hidden state m_t is None, GCM will initialize one for you
# only do this at the beginning, as GCM must track and update the hidden
# state to function correctly
#
# You can inspect m_t, as it is just a graph of observations
# the first element is the node feature matrix and the second is the adjacency matrix
m_t = None

for t in train_timestep:
   # Obs at timestep t should be a tensor of shape (batch_size, obs_size)
   # obs = my_env.step()
   belief, m_t = gcm(obs, m_t)
   # GCM provides a belief state -- a combination of all past observational data relevant to the problem
   # What you likely want to do is put this state through actor and critic networks to obtain
   # action and value estimates
   action_logits = logits_nn(belief)
   state_value = vf_nn(belief)
```

We provide a few edge selectors, which we briefly detail here:
```python
gcm.edge_selectors.temporal.TemporalBackedge
# Connections to the past. Give it [1,2,4] to connect each
# observation t to t-1, t-2, and t-4.

gcm.edge_selectors.dense.DenseEdge
# Connections to all past observations
# observation t is connected to t-1, t-2, ... 0

gcm.edge_selectors.distance.EuclideanEdge
# Connections to observations within some max_distance
# e.g. if l2_norm(o_t, o_k) < max_distance, create an edge

gcm.edge_selectors.distance.CosineEdge
# Like euclidean edge, but using cosine similarity instead

gcm.edge_selectors.distance.SpatialEdge
# Euclidean distance, but only compares slices from the observation
# this is useful if you have an 'x' and 'y' dimension in your observation
# and only want to connect nearby entries
#
# You can also implement the identity priors using this by setting
# max_distance to something like 1e-6

gcm.edge_selectors.learned.LearnedEdge
# Learn an edge function from the data
# Will randomly sample edges and train thru gradient descent
# call the constructor with the output size of your GNN
```

## Ray Quickstart (WIP)
We provide a ray rllib wrapper around GCM, see the example below for how to use it

```python
import unittest
import torch
import torch_geometric
import ray
from ray import tune

from gcm.ray_gcm import RayDenseGCM
from gcm.edge_selectors.temporal import TemporalBackedge

class GNN(torch.nn.Module):
    """A simple two-layer graph neural network"""
    def __init__(self, obs_size, hidden_size=32):
        super().__init__()
        self.gc0 = torch_geometric.nn.DenseGraphConv(obs_size, hidden_size)
        self.gc1 = torch_geometric.nn.DenseGraphConv(hidden_size, hidden_size)
        self.act = torch.nn.Tanh()

    def forward(self, x, adj, weights, B, N):
        x = self.act(self.gc0(x, adj))
        return self.act(self.gc1(x, adj))


ray.init(
    local_mode=True,
    object_store_memory=3e10,
)
input_size = 16 
hidden_size = 32
cfg = {
    "framework": "torch",
    "num_gpus": 0,
    "env": "CartPole-v0",
    "num_workers": 0,
    "model": {
        "custom_model": RayDenseGCM,
        "custom_model_config": {
            "graph_size": 20,
             # GCM Ray wrapper will automatically convert observation
             # to gnn_input_size using a linear layer
            "gnn_input_size": input_size,
            "gnn_output_size": hidden_size,
            "gnn": GNN(input_size),
            "edge_selectors": TemporalBackedge([1]),
            "edge_weights": False,
        }
    }
}
tune.run(
    "A2C",
    config=cfg,
    stop={"info/num_steps_trained": 100}
)
```
