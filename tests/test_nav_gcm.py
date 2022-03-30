import unittest
import torch
import torch_geometric
from collections import OrderedDict
import gym

from gcm.nav_gcm import NavGCM

class IdentGNN(torch.nn.Module):
    def forward(self, x, edges, rot, pos, batch, front_ptr, back_ptr, flat_new_idx):
        return x

class GNN(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.gc = torch_geometric.nn.GraphConv(size, 1)

    def forward(self, x, edges, rot, pos, batch, front_ptr, back_ptr, flat_new_idx):
        self.edges = edges
        self.x = x
        self.rot = rot
        self.pos = pos
        self.batch = batch
        self.front_ptr = front_ptr
        self.back_ptr = back_ptr
        #return torch.cat([x, pos, rot], dim=-1)
        self.out = self.gc(torch.cat([x, pos, rot], dim=-1), edges)
        return self.out

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

        back_ptr = torch.tensor([2, 7])
        self.assertTrue(torch.all(self.gcm.back_ptr == back_ptr))

        front_ptr = torch.tensor([0, 3])
        self.assertTrue(torch.all(self.gcm.front_ptr == front_ptr))

    def test_base_case(self):
        taus = torch.tensor([1,1,1], dtype=torch.long)
        T = torch.tensor([0,0,0], dtype=torch.long)
        self.gcm.compute_idx(T, taus)

        t_idx = torch.tensor([0, 0, 0])
        b_idx = torch.tensor([0, 1, 2])
        self.assertTrue(torch.all(self.gcm.idx[0] == b_idx))
        self.assertTrue(torch.all(self.gcm.idx[1] == t_idx))

        b_new_idx = torch.tensor([0, 1, 2])
        t_new_idx = torch.tensor([0, 0, 0])
        self.assertTrue(torch.all(self.gcm.new_idx[0] == b_new_idx))
        self.assertTrue(torch.all(self.gcm.new_idx[1] == t_new_idx))

        # t_idx = torch.tensor([0, 0])
        # b_idx = torch.tensor([0, 1])
        # f_idx = torch.tensor([0, 1])
        # new                   N  N
        flat_new_idx = torch.tensor([0, 1, 2])
        self.assertTrue(torch.all(self.gcm.flat_new_idx == flat_new_idx))

        b_out_idx = torch.tensor([0, 1, 2])
        t_out_idx = torch.tensor([0, 0, 0])
        self.assertTrue(torch.all(self.gcm.out_idx[0] == b_out_idx))
        self.assertTrue(torch.all(self.gcm.out_idx[1] == t_out_idx))

        back_ptr = torch.tensor([0, 1, 2])
        self.assertTrue(torch.all(self.gcm.back_ptr == back_ptr))

        front_ptr = torch.tensor([0, 1, 2])
        self.assertTrue(torch.all(self.gcm.front_ptr == front_ptr))

    def test_inference(self):
        taus = torch.tensor([1], dtype=torch.long)
        T = torch.tensor([2], dtype=torch.long)
        self.gcm.compute_idx(T, taus)

        t_idx = torch.tensor([0, 1, 2])
        b_idx = torch.tensor([0, 0, 0])
        self.assertTrue(torch.all(self.gcm.idx[0] == b_idx))
        self.assertTrue(torch.all(self.gcm.idx[1] == t_idx))

        b_new_idx = torch.tensor([0])
        t_new_idx = torch.tensor([2])
        self.assertTrue(torch.all(self.gcm.new_idx[0] == b_new_idx))
        self.assertTrue(torch.all(self.gcm.new_idx[1] == t_new_idx))

        # t_idx = torch.tensor([0, 1, 2])
        # b_idx = torch.tensor([0, 0, 0])
        # f_idx = torch.tensor([0, 1, 2])
        # new                         N
        flat_new_idx = torch.tensor([2])
        self.assertTrue(torch.all(self.gcm.flat_new_idx == flat_new_idx))

        b_out_idx = torch.tensor([0])
        t_out_idx = torch.tensor([0])
        self.assertTrue(torch.all(self.gcm.out_idx[0] == b_out_idx))
        self.assertTrue(torch.all(self.gcm.out_idx[1] == t_out_idx))

        back_ptr = torch.tensor([2])
        self.assertTrue(torch.all(self.gcm.back_ptr == back_ptr))

        front_ptr = torch.tensor([0])
        self.assertTrue(torch.all(self.gcm.front_ptr == front_ptr))

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

class TestE2E(unittest.TestCase):
    def setUp(self):
        self.gcm = NavGCM(gnn=GNN(4), causal=True, max_verts=8, r=3, edge_method="radius")
        
    def test_e2e_one_batch(self):
        taus = torch.tensor([8], dtype=torch.long)
        T = torch.tensor([0], dtype=torch.long)
        obs = torch.arange( 8* 1).reshape(1, 8, 1).float()
        pos = torch.arange( 8* 2).reshape(1, 8, 2).float()
        rot = torch.arange( 8* 1).reshape(1, 8, 1).float()
        state = [
            torch.zeros(1, 8, 1),
            torch.zeros(1, 8, 2), 
            torch.zeros(1, 8, 1), 
            T
        ]
        inf_state = [s.clone() for s in state]
        train_output, train_state = self.gcm(obs, pos, rot, taus, state)
        train_edges = self.gcm.gnn.edges.clone()
        train_x = self.gcm.gnn.x.clone()
        train_pos = self.gcm.gnn.pos.clone()
        train_rot = self.gcm.gnn.rot.clone()
        train_out = self.gcm.gnn.out.clone()
        train_batch = self.gcm.gnn.batch.clone()

        inf_output = []
        taus = torch.tensor([1], dtype=torch.long)
        for i in range(8):
            output, inf_state = self.gcm(
                obs[:,i,None], pos[:,i,None], rot[:,i,None], taus, inf_state
            )
            if not torch.allclose(output, train_output[:,i,None]):
                self.fail(f"{i}: {output} != {train_output[:,i,None]}")
            inf_output.append(output)
        inf_output = torch.cat(inf_output, dim=1)
        for i in range(len(train_state)):
            if not torch.all(train_state[i] == inf_state[i]):
                self.fail(f"{i}: {train_state[i]} != {inf_state[i]}")
        inf_edges = self.gcm.gnn.edges.clone()
        inf_x = self.gcm.gnn.x.clone()
        inf_pos = self.gcm.gnn.pos.clone()
        inf_rot = self.gcm.gnn.rot.clone()
        inf_out = self.gcm.gnn.out.clone()
        inf_batch = self.gcm.gnn.batch.clone()
        self.assertTrue(torch.all(train_edges == inf_edges))
        self.assertTrue(torch.all(train_x == inf_x))
        self.assertTrue(torch.all(train_pos == inf_pos))
        self.assertTrue(torch.all(train_rot == inf_rot))
        self.assertTrue(torch.all(train_out == inf_out))
        self.assertTrue(torch.all(train_batch == inf_batch))
        self.assertTrue(torch.allclose(inf_output, train_output))

    def test_e2e_multi_batch(self):
        taus = torch.tensor([8, 8], dtype=torch.long)
        T = torch.tensor([0, 0], dtype=torch.long)
        obs = torch.arange(2* 8* 1).reshape(2, 8, 1).float()
        pos = torch.arange(2* 8* 2).reshape(2, 8, 2).float()
        rot = torch.arange(2* 8* 1).reshape(2, 8, 1).float()
        state = [
            torch.zeros(2, 8, 1),
            torch.zeros(2, 8, 2), 
            torch.zeros(2, 8, 1), 
            T
        ]
        inf_state = [s.clone() for s in state]
        train_output, train_state = self.gcm(obs, pos, rot, taus, state)
        train_edges = self.gcm.gnn.edges.clone()
        train_x = self.gcm.gnn.x.clone()
        train_pos = self.gcm.gnn.pos.clone()
        train_rot = self.gcm.gnn.rot.clone()
        train_out = self.gcm.gnn.out.clone()
        train_batch = self.gcm.gnn.batch.clone()

        inf_output = []
        taus = torch.tensor([1, 1], dtype=torch.long)
        for i in range(8):
            output, inf_state = self.gcm(
                obs[:,i,None], pos[:,i,None], rot[:,i,None], taus, inf_state
            )
            if not torch.allclose(output, train_output[:,i,None]):
                self.fail(f"{i}: {output} != {train_output[:,i,None]}")
            inf_output.append(output)
        inf_output = torch.cat(inf_output, dim=1)
        for i in range(len(train_state)):
            if not torch.all(train_state[i] == inf_state[i]):
                self.fail(f"{i}: {train_state[i]} != {inf_state[i]}")
        inf_edges = self.gcm.gnn.edges.clone()
        inf_x = self.gcm.gnn.x.clone()
        inf_pos = self.gcm.gnn.pos.clone()
        inf_rot = self.gcm.gnn.rot.clone()
        inf_out = self.gcm.gnn.out.clone()
        inf_batch = self.gcm.gnn.batch.clone()
        self.assertTrue(torch.all(train_edges == inf_edges))
        self.assertTrue(torch.all(train_x == inf_x))
        self.assertTrue(torch.all(train_pos == inf_pos))
        self.assertTrue(torch.all(train_rot == inf_rot))
        self.assertTrue(torch.all(train_out == inf_out))
        self.assertTrue(torch.all(train_batch == inf_batch))
        self.assertTrue(torch.allclose(inf_output, train_output))
