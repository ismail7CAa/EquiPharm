"""Helpers for reconstructing screening models from training checkpoints."""

from __future__ import annotations

import inspect
import re


def model_kwargs_from_checkpoint(model_type, checkpoint: dict) -> dict:
    """Infer constructor arguments from checkpoint tensors and run metadata."""
    state = checkpoint.get("model_state_dict", {})
    config = checkpoint.get("config") or {}
    parameters = inspect.signature(model_type).parameters
    kwargs = {}

    embedding_weight = state.get("embedding.weight")
    linear_weight = state.get("linear.weight")
    degree_widths = {}
    for key, tensor in state.items():
        match = re.fullmatch(r"model\.norm\.transforms\.(\d+)", key)
        if match is not None:
            degree_widths[int(match.group(1))] = int(tensor.shape[0])
    degree_dims = tuple(degree_widths[index] for index in sorted(degree_widths))
    core_norm = state.get("model.norm.transforms.0")
    if embedding_weight is not None:
        if "n_token" in parameters:
            kwargs["n_token"] = int(embedding_weight.shape[1])
        if "embedding_dim" in parameters:
            kwargs["embedding_dim"] = int(embedding_weight.shape[0])

    if degree_dims and "degree_dims" in parameters:
        kwargs["degree_dims"] = degree_dims

    if core_norm is not None and "hidden_dim" in parameters:
        kwargs["hidden_dim"] = int(core_norm.shape[0])
    elif linear_weight is not None and "hidden_dim" in parameters:
        kwargs["hidden_dim"] = int(linear_weight.shape[1])
    elif embedding_weight is not None and "hidden_dim" in parameters:
        kwargs["hidden_dim"] = int(embedding_weight.shape[0])
    elif "hidden_dim" in parameters and config.get("hidden_dim") is not None:
        kwargs["hidden_dim"] = int(config["hidden_dim"])

    if linear_weight is not None and "n_out" in parameters:
        kwargs["n_out"] = int(linear_weight.shape[0])

    if "drop_path" in parameters:
        kwargs["drop_path"] = float(config.get("drop_path", 0.0))

    if "num_neighbors" in parameters:
        model_name = str(config.get("model_name", "")).lower()
        kwargs["num_neighbors"] = 4 if model_name == "equiformeradj" else 2

    return kwargs
