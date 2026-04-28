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
sys.path.append('/project/IZZY/molecular-representation/Methods/')                 # Path where the model is located 
#from dig.threedgraph.dataset import QM93D                          # Dataset utilities for 3D molecular graphs (QM9 with 3D coordinates)
from torch_geometric.datasets import QM9                            # Loading the QM9 dataset from Torch-Geometric
from torch_geometric.loader import DataLoader                       # DataLoader to batch and shuffle molecular graph data
from torch.optim import Adam                                        # Optimizer (Adam) and learning rate scheduler (StepLR)
from torch.optim.lr_scheduler import StepLR                         # Progress bar for training/evaluation loops
from tqdm import tqdm                                               # Unit conversion constants (Hartree, eV, Bohr, Angstrom) from ASE
from ase.units import Hartree, eV, Bohr, Ang                        # TensorBoard writer for tracking metrics and visualizing training progress
from torch.utils.tensorboard import SummaryWriter                   # TensorBoard writer for tracking metrics and visualizing training progress 
sys.path.append('/project/IZZY/molecular-representation/Methods/')                              # 3D GNN model implementation (SphereNet) from DIG (Deep 
from train_eval_Copy1 import train_epoch, evaluate                        # Importing the training and evaluating function
import torch_geometric
print(torch_geometric.__version__)
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_mean
from torch_geometric.datasets import QM9
#from se3_transformer_pytorch import SE3Transformer
# from equiformer_pytorch import Equiformer
import equiformer_pytorch
from equiformer_pytorch import Equiformer
from torch_geometric.utils import to_dense_batch, to_dense_adj
sys.path.append('/project/IZZY/molecular-representation/Methods/')  
# from equiformer_Copy2 import EquiformerQM9


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
dataset = QM9(root='data/QM9')

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


# In[8]:


#======================================================================
# Build Equiformer model
#======================================================================

#-------------------------------------------------------------
# Define Equiformer 
#-------------------------------------------------------------
class EquiformerQM9(nn.Module):
    def __init__(self, n_token=11, n_out=19, hidden_dim=128):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 1) Atom feature embedding -> hidden_dim
        self.embedding = nn.Linear(n_token, hidden_dim)

        # 2) Equiformer core
        # input_degrees=1: inputs are scalar features
        # num_degrees=2: internal features include degree 0 and 1 (scalars + vectors)
        self.model = Equiformer(
            dim=hidden_dim,
            dim_in=hidden_dim,
            input_degrees=1,
            num_degrees=2,

            heads=4,
            dim_head=hidden_dim // 4,   # 32 when hidden_dim=128
            depth=8, # 8 stacked Equiformer blocks.Each block typically contains: equivariant graph attention, feed-forward / gating, equivariant normalization, residuals.

            # --- key efficiency / "molecular graph" knobs ---
            attend_sparse_neighbors=True,  # requires adj_mat
            num_neighbors=8,               # 0 = bonds only; >0 adds closest geometric neighbors
            num_adj_degrees_embed=2,       # adds 2-hop connectivity embedding
            max_sparse_neighbors=16,       # cap total sparse neighbors

            # we generally don't need valid_radius if we pass adj_mat,
            # but if num_neighbors > 0 it will still use geometry to fetch nearest neighbors.
            valid_radius=10.0,

            reduce_dim_out=False,
            attend_self=True,
            l2_dist_attention=False
        )

        # 3) Regression head
        self.linear = nn.Linear(hidden_dim, n_out) # After we get a molecule-level representation (a vector per molecule), we predict 19 targets.

    def forward(self, data):
        x, coords, batch = data.x, data.pos, data.batch

        # 1) Embed node features
        x = self.embedding(x)

        # 2) Dense batching : This is necessary because attention is usually implemented on dense tensors.
        x, mask = to_dense_batch(x, batch)          # (B, N, F), (B, N)
        coords, _ = to_dense_batch(coords, batch)   # (B, N, 3)

        # 3) Build adjacency from bond graph (PyG edge_index)
        #    to_dense_adj returns float 0/1, convert to bool
        """
        We pass the bond adjacency matrix to Equiformer to constrain attention to chemically meaningful neighbors.
        This ensures that information always propagates along covalent bonds,while geometric neighbors are added to capture non-bonded interactions.
        """
        adj_mat = to_dense_adj(data.edge_index, batch=batch).bool()  # (B, N, N)

        # 4) Forward through Equiformer using sparse neighbor attention
        out = self.model(x, coords, mask=mask, adj_mat=adj_mat)

        # 5) Extract invariant (degree-0) features safely
        if hasattr(out, "type0"):
            x = out.type0
        elif isinstance(out, dict):
            x = out.get(0, next(iter(out.values())))
        elif isinstance(out, (list, tuple)):
            x = out[0]
        else:
            x = out

        # 6) Pool if needed and predict
        if x.ndim == 2:
            # (B, F) already pooled
            return self.linear(x)

        if x.ndim == 3:
            # (B, N, F) node features -> masked mean pooling
            mask_f = mask[:, :x.size(1)].float()
            x = (x * mask_f.unsqueeze(-1)).sum(dim=1) / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
            return self.linear(x)

        raise ValueError(f"Unexpected output shape from Equiformer: {x.shape}")


# In[ ]:


#==============================================================================================
# Trainig Loop
#==============================================================================================

def main():
    #------------------------------
    # Hyperparameters
    #------------------------------
    epochs = 1100                           # number of training epochs
    batch_size = 12                      # batch size for training
    vt_batch_size = 36                  # batch size for validation
    lr = 1e-5                             # learning rate
    lr_decay_step = 50                    # steps after which LR is decayed
    lr_decay_factor = 0.5                 # factor to decay learning rate
    weight_decay = 1e-4                   # L2 regularization
    save_dir = 'checkpoints_Equiformer'    # directory to save model checkpoints 
    log_dir = 'logs_Equiformer'            # TensorBoard logs directory

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
    model = EquiformerQM9().to(device)   

    #----------------------------------------
    # Optimizer and Scheduler
    #----------------------------------------
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_factor)

    # Track best validation/test performance
    best_mean_val=float('inf')
    best_val = [float('inf')] * len(PROPERTY_NAMES)
    best_test = [float('inf')] * len(PROPERTY_NAMES)

    # Print total number of trainabel parameters and target property
    print("#params:", sum(p.numel() for p in model.parameters()))
    print("Training for all 19 targets") 
    
    #--------------------------------------------
    # Training Loop
    #--------------------------------------------
    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch} ===")

        #Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, device, accum_steps = 4)

        # Evaluate on validation and test sets (MAE in original units)
        val_mae = evaluate(model, val_loader, device, norm_stats['mean'], norm_stats['std'])
        test_mae = evaluate(model, test_loader, device, norm_stats['mean'], norm_stats['std'])

        # Print epoch metrics
        print(f"Train loss (MSE): {train_loss:.6f}")
        for i, prop in enumerate(PROPERTY_NAMES):
            print(f"  {prop:15s} | Val MAE: {val_mae[i]:.6f} | Test MAE: {test_mae[i]:.6f}")
            writer.add_scalar(f'val_mae/{prop}', val_mae[i], epoch)
            writer.add_scalar(f'test_mae/{prop}', test_mae[i], epoch)
        #---------------------------------------------
        # Save checkpoint if validation improves 
        #---------------------------------------------
        mean_val_mae = sum(val_mae) / len(val_mae)
        
        # If mean validation MAE improved → save one best model
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
        
            print(f" Saved best overall model (mean validation MAE improved to {best_mean_val:.4f})")

        # Update learning rate according to scheduler
        scheduler.step()
    # Close TensorBoard writer
    writer.close()
    print("\nFinished training.")
    print("Best validation and test MAEs per property:")
    for prop, v, t in zip(PROPERTY_NAMES, best_val, best_test):
        print(f"  {prop:15s} | Best validation MAE: {v:.6f} | Test MAE at best val: {t:.6f}")

    #print("Best validation MAE:", best_val)
    #print("Test MAE at best val:", best_test)

# Run the training loop
if __name__ == "__main__":
    main()


# In[ ]:




