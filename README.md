# Graph Convolutional Memory for Reinforcement Learning

## Description
Graph convolutional memory (GCM) is graph-structured memory that may be applied to reinforcement learning to solve POMDPs, replacing LSTMs or attention mechanisms. GCM allows you to embed your domain knowledge in the form of connections in a knowledge graph. See the [full paper](https://arxiv.org/pdf/2106.14117.pdf) for further details. This repo contains the GCM library implementation for use in your projects. To replicate the experiments from the paper, please see [this repository instead](https://github.com/smorad/graph-conv-memory-paper).

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

# If the hidden state m_t is None, GCM will initialize one for you
# only do this at the beginning, as GCM must track and update the hidden
# state to function correctly
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
