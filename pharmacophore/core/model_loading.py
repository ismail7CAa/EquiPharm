"""Helpers for reconstructing screening models from training checkpoints."""

from __future__ import annotations

import inspect


def model_kwargs_from_checkpoint(model_type, checkpoint: dict) -> dict:
    """Infer constructor arguments from checkpoint tensors and run metadata."""
    state = checkpoint.get("model_state_dict", {})
    config = checkpoint.get("config") or {}
    parameters = inspect.signature(model_type).parameters
    kwargs = {}

    embedding_weight = state.get("embedding.weight")
    linear_weight = state.get("linear.weight")
    if embedding_weight is not None:
        if "n_token" in parameters:
            kwargs["n_token"] = int(embedding_weight.shape[1])
        if "hidden_dim" in parameters:
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
