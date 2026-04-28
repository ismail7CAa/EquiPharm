#!/usr/bin/env python
# coding: utf-8

# In[7]:


import torch
import torch_geometric
from dig.threedgraph.dataset import QM93D
from dig.threedgraph.method import ComENet 
from dig.threedgraph.evaluation import ThreeDEvaluator
from dig.threedgraph.method import run
import os

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
device

dataset = QM93D(root='dataset/')
target = 'U0' # choose from: mu, alpha, homo, lumo, r2, zpve, U0, U, H, G, Cv
dataset.data.y = dataset.data[target]

split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)

train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
print('train, validaion, test:', len(train_dataset), len(valid_dataset), len(test_dataset))

model = ComENet(cutoff=8.0, num_layers=4, 
        hidden_channels=256, middle_channels=64, out_channels= 1, 
        num_spherical=2, num_radial=3, num_output_layers=3,
        )#.to(device)
loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()

run3d = run()

save_dir = f'checkpoints/{target}'
os.makedirs(save_dir, exist_ok=True)

run3d.run(
    device=device, 
    train_dataset=train_dataset, 
    valid_dataset=valid_dataset, 
    test_dataset=test_dataset,
    model=model,
    loss_func=loss_func,
    evaluation=evaluation,
    epochs=10,
    batch_size=32,
    vt_batch_size=32,
    lr=0.0005,
    lr_decay_factor=0.5,
    lr_decay_step_size=15,
    save_dir= save_dir
)


# In[ ]:




