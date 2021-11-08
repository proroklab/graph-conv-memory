import unittest
import torch
import torch_geometric
import ray
from ray import tune

from gcm.ray_gcm import RayDenseGCM
from gcm.edge_selectors.temporal import TemporalBackedge


class TestRaySanity(unittest.TestCase):
    def test_one_iter(self):
        hidden = 32
        ray.init(
            local_mode=True,
            object_store_memory=3e10,
        )
        dgc = torch_geometric.nn.Sequential(
            "x, adj, weights, B, N",
            [
                # Mean and sum aggregation perform roughly the same
                # Preprocessor with 1 layer did not help
                (torch_geometric.nn.DenseGraphConv(hidden, hidden), "x, adj -> x"),
                (torch.nn.Tanh()),
                (torch_geometric.nn.DenseGraphConv(hidden, hidden), "x, adj -> x"),
                (torch.nn.Tanh()),
            ],
        )
        cfg = {
            "framework": "torch",
            "num_gpus": 0,
            "env": "CartPole-v0",
            "num_workers": 0,
            "model": {
                "custom_model": RayDenseGCM,
                "custom_model_config": {
                    "graph_size": 32,
                    "gnn_input_size": hidden,
                    "gnn_output_size": hidden,
                    "gnn": dgc,
                    "edge_selectors": TemporalBackedge([1]),
                    "edge_weights": False,
                },
            },
        }
        tune.run("A2C", config=cfg, stop={"info/num_steps_trained": 100})
