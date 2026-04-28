# EquifPharm: new AI Approach for Drug Disovery

A PyTorch implementation of:
**[2D GNNs (GCN, GAT, GraphSAGE, GIN)] (https://pytorch-geometric.readthedocs.io/en/latest/cheatsheet/gnn_cheatsheet.html), 3D message-passing model (SphereNet)(https://github.com/divelab/DIG), and SE(3)-equivariant 3D architectures (SE(3)-Transformer (https://github.com/FabianFuchsML/se3-transformer-public), Equiformer(https://github.com/atomicarchitects/equiformer). (**

## Overview

This repository presents an end-to-end framework for learning and evaluating molecular representations using modern graph neural networks.

It includes a reproducible benchmarking pipeline comparing state-of-the-art 2D and 3D architectures (SphereNet, SE(3)-Transformer, and Equiformer) on QM9 to identify the most effective model for molecular understanding.

Building on the top-performing model, the repository further provides an AI-driven pharmacophore modeling workflow that leverages learned molecular embeddings for structure-based virtual screening on the DUD-E dataset.

### Project Aim

After benchmarking multiple 3D deep learning models, our project focuses on improving AI-based pharmacophore approaches by leveraging the top-performing model identified in the benchmark. 


## Dataset
    •	Benchmark pipeline: QM9 
	•	Pharmacophore : DUD-E
 

## Methods
### 🔹 GNN2

### 🔹 SphereNet

### 🔹 SE-3-Transformer

### 🔹 Equiformer

## Setup & Installation
### EquiScore environment

- The environment configuration is available [here](MolRepres/environment.yml).


## Workflow
### Pipeline of the Benchmarking

-currently under active development.

### Pipeline of Pharmacophore

<p align="center">
  <img src="pipeline_with_optimization_loop_orthogonal.png" width="100"/>
</p>
-currently under active development.

## Project Status

This project is currently under active development.  
Features, models, and results may change as the work progresses.




