#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch_geometric
from dig.threedgraph.dataset import QM93D
from dig.threedgraph.method import ComENet, SphereNet 
from dig.threedgraph.evaluation import ThreeDEvaluator
from dig.threedgraph.method import run
import os
import numpy as np
import math

np.math = math
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
device

dataset = QM93D(root='dataset/')
target = 'U0' # choose from: mu, alpha, homo, lumo, r2, zpve, U0, U, H, G, Cv
dataset.data.y = dataset.data[target]

split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)

train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
print('train, validaion, test:', len(train_dataset), len(valid_dataset), len(test_dataset))

model = SphereNet(energy_and_force=False, cutoff=5.0, num_layers=4,
        hidden_channels=128, out_channels=1, int_emb_size=64,
        basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
        num_spherical=3, num_radial=6, envelope_exponent=5,
        num_before_skip=1, num_after_skip=2, num_output_layers=3, use_node_features=True
        )#.to(device)
loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()

run3d = run()
run3d.run(device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation, epochs=10, batch_size=32, vt_batch_size=32, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=15)

