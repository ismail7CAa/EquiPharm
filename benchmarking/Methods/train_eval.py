#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Training and Evaluation Functions 
import torch
import torch.nn.functional as F
from tqdm import tqdm


# Train epoch
def train_epoch(model, loader, optimizer, device, accum_steps=4):
    """
    Perform one epoch of training.

    Args: 
        model: PyTorch model
        loader: DataLoader for training data
        optimizer: optimizer (e.g., Adam)
        device: 'cpu' or 'cuda'

    Returns:
        Average training loss over the epoch
    """
    
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        target = batch.y
        loss = F.mse_loss(pred, target)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

# Evaluate model
@torch.no_grad()
def evaluate(model, loader, device, norm_means, norm_stds):
    """
    Evaluate model performance on validation or test set.

    Args:
        model: PyTorch model
        loader: DataLoader for evaluation data
        device: 'cpu' or 'cuda'
        norm_means: mean used for target normalization
        norm_stds: std used for target normalization

    Returns:
        List of Mean Absolute Errors (MAEs) for each target
    """
    
    model.eval()
    preds, targets = [], []

    for batch in tqdm(loader, desc="Eval", leave=False):
        batch = batch.to(device)
        out = model(batch).detach().cpu()
        t = batch.y.detach().cpu()
        preds.append(out)
        targets.append(t)
    
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    maes = []
    for i in range(len(norm_means)):
        preds_den = preds[:, i] * norm_stds[i] + norm_means[i]
        targets_den = targets[:, i] * norm_stds[i] + norm_means[i]
        mae = torch.mean(torch.abs(preds_den - targets_den)).item()
        maes.append(mae)

    return maes


# In[ ]:




