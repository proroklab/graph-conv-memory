import unittest
import torch
import torch_geometric
from collections import OrderedDict
import gym

from gcm.nav_gcm import NavGCM

class IdentGNN(torch.nn.Module):
    def forward(x, edges, rot, pos, batch, flat_new_idx, back_ptr):
        return x

class TestComputeIdx(unittest.TestCase):
    def setUp(self):
        self.gcm = NavGCM(gnn=IdentGNN())
        
    def test_ragged(self):
        taus = torch.tensor([2,3], dtype=torch.long)
        T = torch.tensor([1,2], dtype=torch.long)
        self.gcm.compute_idx(T, taus)

        t_idx = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4])
        b_idx = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
        self.assertTrue(torch.all(self.gcm.idx[0] == b_idx))
        self.assertTrue(torch.all(self.gcm.idx[1] == t_idx))

        b_new_idx = torch.tensor([0, 0, 1, 1, 1])
        t_new_idx = torch.tensor([1, 2, 2, 3, 4])
        self.assertTrue(torch.all(self.gcm.new_idx[0] == b_new_idx))
        self.assertTrue(torch.all(self.gcm.new_idx[1] == t_new_idx))

        # t_idx = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4])
        # b_idx = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
        # f_idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        # new                      N  N        N  N  N
        flat_new_idx = torch.tensor([1, 2, 5, 6, 7])
        self.assertTrue(torch.all(self.gcm.flat_new_idx == flat_new_idx))

        b_out_idx = torch.tensor([0, 0, 1, 1, 1])
        t_out_idx = torch.tensor([0, 1, 0, 1, 2])
        self.assertTrue(torch.all(self.gcm.out_idx[0] == b_out_idx))
        self.assertTrue(torch.all(self.gcm.out_idx[1] == t_out_idx))

class TestUpdate(unittest.TestCase):
    def setUp(self):
        self.gcm = NavGCM(gnn=IdentGNN(), causal=True)
        
    def test_ragged(self):
        taus = torch.tensor([2,3], dtype=torch.long)
        T = torch.tensor([1,2], dtype=torch.long)

        x = torch.zeros((2, 10, 1))
        pos = torch.zeros((2, 10, 2))
        rot = torch.zeros((2, 10, 1))
        x_in = torch.ones((2, 3, 1))
        pos_in = torch.ones((2, 3, 2))
        rot_in = torch.ones((2, 3, 1))

        tgt_x = x.clone()
        tgt_pos = pos.clone()
        tgt_rot = rot.clone()

        tgt_x[0, 1:3] = 1
        tgt_x[1, 2:5] = 1

        tgt_pos[0, 1:3] = 1
        tgt_pos[1, 2:5] = 1

        tgt_rot[0, 1:3] = 1
        tgt_rot[1, 2:5] = 1

        self.gcm.compute_idx(T, taus)
        new_x, new_pos, new_rot = self.gcm.update(
            x_in, pos_in, rot_in,
            x, pos, rot,
            T, taus
        )

        self.assertTrue(torch.all(tgt_x == new_x))
        self.assertTrue(torch.all(tgt_pos == new_pos))
        self.assertTrue(torch.all(tgt_rot == new_rot))
