#!/usr/bin/env python
# coding: utf-8

# In[1]:



#==========================================================
# Load Packages 
#==========================================================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('/project/IZZY/MolRepres/Methods/')                 # Path where the model is located 
#from dig.threedgraph.dataset import QM93D                          # Dataset utilities for 3D molecular graphs (QM9 with 3D coordinates)
from torch_geometric.datasets import QM9                            # Loading the QM9 dataset from Torch-Geometric
from torch_geometric.loader import DataLoader                       # DataLoader to batch and shuffle molecular graph data
from torch.optim import Adam                                        # Optimizer (Adam) and learning rate scheduler (StepLR)
from torch.optim.lr_scheduler import StepLR                         # Progress bar for training/evaluation loops
from tqdm import tqdm                                               # Unit conversion constants (Hartree, eV, Bohr, Angstrom) from ASE
from ase.units import Hartree, eV, Bohr, Ang                        # TensorBoard writer for tracking metrics and visualizing training progress
from torch.utils.tensorboard import SummaryWriter                   # TensorBoard writer for tracking metrics and visualizing training progress 
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_mean                                  # 3D GNN model implementation (SphereNet) from DIG (Deep Graph Library for 3D Graphs)
from train_eval_Copy1 import train_epoch, evaluate                        # Importing the training and evaluating function
import torch_geometric
print(torch_geometric.__version__)

from torch_geometric.datasets import QM9
from se3_transformer_pytorch import SE3Transformer
# from equiformer_pytorch import Equiformer


# In[4]:


#==========================================================
# GPU Device
#==========================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # This line checks if a CUDA-enabled GPU is available. If yes, computations will be performed on the GPU for faster training. Otherwise, it falls back to using the CPU.
# Print which device is being used 
print("Using device:", device)


# In[5]:


#==============================================================
# Load dataset and split
#==============================================================

# Load the QM9 dataset 
#qm9_root = os.path.expanduser("~/data/QM9")  # any writable path
dataset = QM9(root="data/QM9")
#dataset = QM9(root='data/QM9')

# Count how many molecular graphs are in the dataset
num_mols = len(dataset)
print(f"Total QM9 molecules: {num_mols}")

# Define the desired split sizes for training, validation, and testing
train_size, valid_size = 110000, 10000
# Ensure the sum of train and validation sizes does not exceed dataset size
assert train_size + valid_size < num_mols, "Split sizes too large for dataset"

# Randomly shuffle all molecule indices
perm = torch.randperm(num_mols, generator=torch.Generator().manual_seed(42))

# Use the shuffled indices to create non-overlapping subsets
train_idx = perm[:train_size]
valid_idx = perm[train_size:train_size+valid_size]
test_idx  = perm[train_size+valid_size:]

# Build subsets based on the index splits
train_dataset = dataset[train_idx]
valid_dataset = dataset[valid_idx]
test_dataset  = dataset[test_idx]


# In[6]:


#======================================================================
# Define the QM9 Targets
#======================================================================
PROPERTY_NAMES = [
    'μ (D)', 'α (Ang³)', 'ε_HOMO (eV)', 'ε_LUMO (eV)', 'Δε (eV)', '⟨R²⟩ (Ang²)',
    'ZPVE (eV)', 'U₀ (eV)', 'U (eV)', 'H (eV)', 'G (eV)', 'c_v', 'U₀_atom',
    'U_atom', 'H_atom', 'G_atom', 'A (GHz)', 'B (GHz)', 'C (GHz)'
] 

# ======================================================================
# QM9 Unit Conversion Factors
# ======================================================================
def get_qm9_conversions_tensor(device):
    return torch.tensor([
        1.0, Bohr**3 / Ang**3, Hartree / eV, Hartree / eV, Hartree / eV,
        Bohr**2 / Ang**2, Hartree / eV, Hartree / eV, Hartree / eV,
        Hartree / eV, Hartree / eV, 1.0, 1.0, Hartree / eV,
        Hartree / eV, Hartree / eV, 1.0, 1.0, 1.0
    ], dtype=torch.float, device=device)


# In[7]:


#========================================================================
# Normalize all QM9 targets
#========================================================================
y_raw_all = dataset.data.y.clone().cpu()
conversions_cpu = get_qm9_conversions_tensor('cpu')
y_conv_all = y_raw_all * conversions_cpu.unsqueeze(0)

norm_stats = {'mean': [], 'std': []}
y_norm_all = torch.zeros_like(y_conv_all)

for i in range(y_conv_all.shape[1]):
    train_y_cpu = y_conv_all[train_idx.cpu(), i]
    mean_i = float(train_y_cpu.mean().item())
    std_i = float(train_y_cpu.std().item()) if train_y_cpu.std().item() != 0 else 1.0
    y_norm_all[:, i] = (y_conv_all[:, i] - mean_i) / std_i
    norm_stats['mean'].append(mean_i)
    norm_stats['std'].append(std_i)

dataset.data.y = y_norm_all.to(torch.float)
print("Normalization complete for all targets.")


# In[9]:


import torch
import torch.nn as nn

from se3_transformer_pytorch import SE3Transformer
from torch_geometric.utils import to_dense_adj, to_dense_batch

class SE3EncoderDecoderQM9(nn.Module):
    def __init__(self, n_token=11, n_out=19, n_hidden=8, n_heads=4, distance=10.0):
        super().__init__()

        # QM9 atoms are one-hot (dim 11) → project to 128 hidden dim
        self.embedding = nn.Linear(n_token, n_hidden)

        self.transformer = SE3Transformer(
            dim=n_hidden,
            heads=n_heads,
            dim_head=n_hidden//n_heads ,
            depth=2,
            num_degrees=2,
            valid_radius=distance,
            max_sparse_neighbors=14
        )
        self.linear = nn.Linear(n_hidden, n_out)

    def forward(self, data):
        x, coords, batch = data.x, data.pos, data.batch
        x = self.embedding(x)
        x, mask = to_dense_batch(x, batch)
        coords, _ = to_dense_batch(coords, batch)
        adj_mat = to_dense_adj(data.edge_index, batch=batch).bool()
        x = self.transformer(x, coords, mask=mask, adj_mat=adj_mat)
        mask_f = mask.float()
        x = (x * mask_f.unsqueeze(-1)).sum(dim=1) / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        out = self.linear (x)
        return out


# In[ ]:


#==============================================================================================
# Trainig Loop
#==============================================================================================

def main():
    #------------------------------
    # Hyperparameters
    #------------------------------
    epochs = 1000                           # number of training epochs
    batch_size = 8                      # batch size for training
    vt_batch_size = 16                    # batch size for validation
    lr = 3e-4                             # learning rate
    lr_decay_step = 50                    # steps after which LR is decayed
    lr_decay_factor = 0.5                 # factor to decay learning rate
    weight_decay = 1e-4                   # L2 regularization
    save_dir = 'checkpoints_SE3T-2'    # directory to save model checkpoints 
    log_dir = 'logs_SE3T-2'            # TensorBoard logs directory

    #Create directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # TensorBoard write for visualization of metrics
    writer = SummaryWriter(log_dir=log_dir)

    #-----------------------------
    # Dataloaders 
    #-----------------------------

    # Shuffled batches for training; sequential batches for validation/testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(valid_dataset, batch_size=vt_batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=vt_batch_size, shuffle=False)

    #----------------------------------------
    # Model
    #----------------------------------------
    # SphereNet model
    model = SE3EncoderDecoderQM9().to(device)   

    #----------------------------------------
    # Optimizer and Scheduler
    #----------------------------------------
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_factor)

    #----------------------------------------
    # Resume from checkpoint
    #----------------------------------------
    ckpt = "/project/IZZY/molecular-representation/benchmarking/Methods/checkpoints_SE3T-1/best_model.pt"

    start_epoch = 1
    best_mean_val = float('inf')
    best_val = [float('inf')] * len(PROPERTY_NAMES)
    best_test = [float('inf')] * len(PROPERTY_NAMES)

    if os.path.exists(ckpt):
        checkpoint = torch.load(ckpt, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_val = checkpoint.get("best_val", best_val)
        best_test = checkpoint.get("best_test", best_test)
        best_mean_val = checkpoint.get("best_mean_val", best_mean_val)

        print(f"Resuming from epoch {start_epoch}")
        print(f"Best mean validation MAE so far: {best_mean_val:.6f}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    print("#params:", sum(p.numel() for p in model.parameters()))
    print("Training for all 19 targets")

    #--------------------------------------------
    # Training Loop
    #--------------------------------------------
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n=== Epoch {epoch} ===")

        train_loss = train_epoch(model, train_loader, optimizer, device, accum_steps=4)

        val_mae = evaluate(model, val_loader, device, norm_stats['mean'], norm_stats['std'])
        test_mae = evaluate(model, test_loader, device, norm_stats['mean'], norm_stats['std'])

        print(f"Train loss (MSE): {train_loss:.6f}")
        for i, prop in enumerate(PROPERTY_NAMES):
            print(f"  {prop:15s} | Val MAE: {val_mae[i]:.6f} | Test MAE: {test_mae[i]:.6f}")
            writer.add_scalar(f'val_mae/{prop}', val_mae[i], epoch)
            writer.add_scalar(f'test_mae/{prop}', test_mae[i], epoch)

        mean_val_mae = sum(val_mae) / len(val_mae)

        if mean_val_mae < best_mean_val:
            best_mean_val = mean_val_mae
            best_val = val_mae
            best_test = test_mae

            save_path = os.path.join(save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val': best_val,
                'best_test': best_test,
                'best_mean_val': best_mean_val
            }, save_path)

            print(f"Saved best overall model (mean validation MAE improved to {best_mean_val:.4f})")

        scheduler.step()

    writer.close()

    print("\nFinished training.")
    print("Best validation and test MAEs per property:")
    for prop, v, t in zip(PROPERTY_NAMES, best_val, best_test):
        print(f"  {prop:15s} | Best validation MAE: {v:.6f} | Test MAE at best val: {t:.6f}")


if __name__ == "__main__":
    main()

# In[ ]:






