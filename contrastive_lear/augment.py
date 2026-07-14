"""PharmacoMatch query/target pair generation."""

from __future__ import annotations

import math

import torch
from torch_geometric.data import Batch, Data


def _sphere_noise(pos: torch.Tensor, radius: float) -> torch.Tensor:
    count = pos.size(0)
    phi = torch.rand(count, device=pos.device) * (2 * math.pi)
    cos_theta = torch.rand(count, device=pos.device) * 2 - 1
    distance = radius * torch.rand(count, device=pos.device).pow(1 / 3)
    sin_theta = (1 - cos_theta.square()).clamp_min(0).sqrt()
    return torch.stack(
        [distance * sin_theta * phi.cos(), distance * sin_theta * phi.sin(), distance * cos_theta],
        dim=1,
    )


def _reduced_pair(data: Data) -> tuple[Data, Data]:
    count = data.num_nodes
    keep = count - int(torch.randint(1, count - 2, (1,)).item())
    order = torch.randperm(count, device=data.x.device)
    first = order[:keep]
    second = order[-keep:]
    return (
        Data(x=data.x[first], pos=data.pos[first], num_ph4_features=keep),
        Data(x=data.x[second], pos=data.pos[second], num_ph4_features=keep),
    )


def _outward_noise(pos: torch.Tensor, radius: float) -> torch.Tensor:
    direction = pos - pos.mean(dim=0, keepdim=True)
    fallback = torch.randn_like(direction)
    norm = direction.norm(dim=1, keepdim=True)
    direction = torch.where(norm > 1e-8, direction / norm.clamp_min(1e-8), fallback)
    return direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-8) * radius


def make_views(batch: Batch, radius: float = 1.5) -> dict[str, Batch]:
    targets, queries, references, negative_queries, negative_targets = [], [], [], [], []
    for graph in batch.to_data_list():
        target = Data(x=graph.x, pos=graph.pos, num_ph4_features=graph.num_nodes)
        reduced, negative_target = _reduced_pair(target)
        query = reduced.clone()
        query.pos = query.pos + _sphere_noise(query.pos, radius)
        reference = reduced.clone()
        reference.pos = reference.pos + _sphere_noise(reference.pos, radius)
        negative_query = reduced.clone()
        negative_query.pos = negative_query.pos + _outward_noise(negative_query.pos, radius)
        targets.append(target)
        queries.append(query)
        references.append(reference)
        negative_queries.append(negative_query)
        negative_targets.append(negative_target)
    return {
        "targets": Batch.from_data_list(targets),
        "queries": Batch.from_data_list(queries),
        "references": Batch.from_data_list(references),
        "negative_queries": Batch.from_data_list(negative_queries),
        "negative_targets": Batch.from_data_list(negative_targets),
    }


def add_complete_edges(batch: Batch) -> Batch:
    edges = []
    for start, end in zip(batch.ptr[:-1], batch.ptr[1:]):
        nodes = torch.arange(start, end, device=batch.x.device)
        src = nodes.repeat_interleave(len(nodes))
        dst = nodes.repeat(len(nodes))
        keep = src != dst
        edges.append(torch.stack([src[keep], dst[keep]]))
    batch.edge_index = torch.cat(edges, dim=1)
    return batch
