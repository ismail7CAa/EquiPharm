#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[4]:


#==========================================================
# Load Packages 
#==========================================================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('/project/IZZY/MolRepres/Methods/')                  # Path where the training, evaluating function is located 
from torch_geometric.datasets import QM9                             # Dataset utilities for 3D molecular graphs (QM9 with 3D coordinates)
from torch_geometric.nn import GCNConv, global_mean_pool             # Graph Neural Network layers and pooling operations from PyTorch Geometric
from torch_geometric.loader import DataLoader                        # DataLoader to batch and shuffle molecular graph data
from torch.optim import Adam                                         # Optimizer (Adam) and learning rate scheduler (StepLR)
from torch.optim.lr_scheduler import StepLR                          # Progress bar for training/evaluation loops
from tqdm import tqdm                                                # Unit conversion constants (Hartree, eV, Bohr, Angstrom) from ASE
from ase.units import Hartree, eV, Bohr, Ang                         # TensorBoard writer for tracking metrics and visualizing training progress
from torch.utils.tensorboard import SummaryWriter                    # TensorBoard writer for tracking metrics and visualizing training progress
from sklearn.preprocessing import StandardScaler                     # Tool for normalizing data (zero mean, unit variance)
from train_eval import train_epoch, evaluate                         # Importing the training and evaluating function


# In[5]:


#==========================================================
# GPU Device
#==========================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # This line checks if a CUDA-enabled GPU is available. If yes, computations will be performed on the GPU for faster training. Otherwise, it falls back to using the CPU.
# Print which device is being used 
print("Using device:", device)


# In[6]:


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


# In[7]:


#======================================================================
# Define the QM9 Targerts
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
        1.0,                # 0 - μ (D)     
        Bohr**3 / Ang**3,   # 1 - α (Bohr³ → Å³)
        Hartree / eV,       # 2 - ε_HOMO
        Hartree / eV,       # 3 - ε_LUMO
        Hartree / eV,       # 4 - Δε
        Bohr**2 / Ang**2,   # 5 - ⟨R²⟩
        Hartree / eV,       # 6 - ZPVE
        Hartree / eV,       # 7 - U₀
        Hartree / eV,       # 8 - U
        Hartree / eV,       # 9 - H
        Hartree / eV,       # 10 - G
        1.0,                # 11 - c_v
        1.0,                # 12 - U₀_atom
        Hartree / eV,       # 13 - U_atom
        Hartree / eV,       # 14 - H_atom
        Hartree / eV,       # 15 - G_atom
        1.0,                # 16 - A (GHz)
        1.0,                # 17 - B (GHz)
        1.0                 # 18 - C (GHz)
    ], dtype=torch.float, device=device)


# In[8]:


#========================================================================
# Normalize all QM9 targets
#========================================================================
y_raw_all = dataset.data.y.clone().cpu()                 # shape [N, 19]
conversions_cpu = get_qm9_conversions_tensor('cpu')      # conversion tensor [19]
y_conv_all = y_raw_all * conversions_cpu.unsqueeze(0)    # apply conversion

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


#===========================================================================
# Build GNN-Model 
#===========================================================================

# Number of input features per node (atom) in each molecular graph
in_channels = dataset.num_node_features
print("Node feature dim:", in_channels)

#--------------------------
# Define GCN-Model
#--------------------------
class GCNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, dropout=0.3):
        """
        Simple 3-layer GCN for regression on molecular graphs.

        Args:
            in_dim(int): Dimension of node features
            hidden_dim(int): Number of hidden units in GCN layers 
            dropout (float): Dropout probability
        """
        
        super().__init__()

        # Three graph convolutional layers
        self.conv1 = GCNConv(in_dim, hidden_dim)  # Each layer updates node embeddings using neighbors' features
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Fully connected layers to map pooled graph embedding => target value
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 19)  # Output: single scalar target
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        """
        
        Forward pass of the GCN.

        Args:
            batch: a PyTorch Geometric batch containing:
                    - x: node features[num_nodes, in_dim]
                    - edge_index: edge list[2, num_edges]
                    - batch: batch vector mapping nodes to molecules

        Returns :
            Predicted target values for each molecule [batch_size, 1]
            
        """
        x, edge_index, batch_vec = batch.x, batch.edge_index, batch.batch

        # Apply three GCN layers with ReLU activation
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        
        # Global mean Pooling
        x = global_mean_pool(x, batch_vec)   # The global mean pool ing takes the mean of all atom embeddings in each molecule´s graph "one feat vector per molecule"
        
        # Decoder : it transfomrs the pooled molecular embedding into the predicted target value 
        x = F.relu(self.fc1(x)) 
        x = self.dropout(x)
        return self.fc2(x)


# In[10]:


#==============================================================================================
# Trainig Loop
#==============================================================================================
def main():
    #------------------------------
    # Hyperparameters
    #------------------------------
    epochs = 500                            # number of training epochs
    batch_size = 128                        # batch size for training
    vt_batch_size = 256                     # batch size for validation
    lr = 1e-5                               # learning rate
    lr_decay_step = 50                      # steps after which LR is decayed
    lr_decay_factor = 0.5                   # factor to decay learning rate
    weight_decay = 1e-4                     # L2 regularization
    save_dir = 'checkpoints_GCN'            # directory to save model checkpoints 
    log_dir = 'logs_GCN'                    # TensorBoard logs directory

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
    # GCN model
    model = GCNModel(in_channels).to(device)  # GCN model uncommented to use it 

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
        train_loss = train_epoch(model, train_loader, optimizer, device)

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




