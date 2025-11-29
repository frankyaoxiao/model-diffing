"""Steering utilities for applying activation directions during generation."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional

import torch

logger = logging.getLogger(__name__)


def _resolve_decoder_layer(model, layer_idx: int):
    try:
        return model.model.layers[layer_idx]
    except AttributeError as exc:  # pragma: no cover - defensive guard
        raise ValueError("Model does not expose decoder layers via model.model.layers") from exc
    except IndexError as exc:
        raise ValueError(f"Layer index {layer_idx} out of range") from exc


@dataclass
class SteeringConfig:
    """Configuration for applying a steering direction during generation."""

    layer_vectors: Dict[int, torch.Tensor]
    scale: float
    do_sample: bool = True
    mode: str = "add"  # add, project_out
    log_stats: bool = False
    stats: Dict[int, Dict[str, float]] = field(default_factory=dict)


@contextmanager
def apply_layer_steering(
    model,
    layer_vectors: Dict[int, torch.Tensor],
    scale: float,
    mode: str = "add",
    stats: Optional[Dict[int, Dict[str, float]]] = None,
) -> Iterator[None]:
    """Temporarily add scaled direction vectors to specific decoder layers.

    Args:
        model: Loaded causal LM (expected to expose ``model.layers``).
        layer_vectors: Mapping of layer index -> 1D direction tensor.
        scale: Scalar multiplier applied to each direction.
        mode: Steering strategy ("add" to add/subtract the vector, "project_out" to
              remove components along the vector; scale controls strength).
    """

    device = next(model.parameters()).device
    handles = []
    stats_enabled = stats is not None and mode == "project_out"
    layer_stats = (
        {layer_idx: {"sum_cos": 0.0, "sum_proj": 0.0, "count": 0} for layer_idx in layer_vectors}
        if stats_enabled
        else {}
    )

    def make_hook(vec: torch.Tensor, layer_idx: int):
        direction = vec.to(device).view(1, 1, -1)
        dir_norm = torch.sum(direction * direction).item()
        state = {"seen_prompt": False}

        def hook(module, inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
                residual = output[1:]
            else:
                hidden = output
                residual = None

                if hidden is not None:
                    if not state["seen_prompt"]:
                        state["seen_prompt"] = True
                        if residual is None:
                            return hidden
                        return (hidden, *residual)

                hidden = hidden.clone()
                steer = direction.to(hidden.dtype)
                if hidden.shape[1] > 1:
                    target = hidden[:, -1:, :]
                else:
                    target = hidden

                if mode == "project_out":
                    if dir_norm > 0:
                        original = target.clone()
                        dot = torch.sum(original * steer, dim=-1, keepdim=True)
                        coeff = dot / dir_norm
                        projection = coeff * steer
                        removed = scale * projection
                        target -= removed

                        if stats_enabled:
                            vec_norm = (dir_norm ** 0.5) + 1e-8
                            token_norm = torch.norm(original, dim=-1, keepdim=True) + 1e-8
                            cos = (dot / (token_norm * vec_norm)).squeeze(-1).clamp(-1.0, 1.0)
                            proj_mag = removed.norm(dim=-1).squeeze(-1)
                            s = layer_stats[layer_idx]
                            s["sum_cos"] += cos.sum().item()
                            s["sum_proj"] += proj_mag.sum().item()
                            s["count"] += cos.numel()
                else:  # default additive steering
                    target += scale * steer

            if residual is None:
                return hidden
            return (hidden, *residual)

        return hook

    try:
        for layer_idx, vector in layer_vectors.items():
            module = _resolve_decoder_layer(model, layer_idx)
            handles.append(module.register_forward_hook(make_hook(vector, layer_idx)))
        yield
    finally:
        for handle in handles:
            handle.remove()
        if stats_enabled:
            for layer_idx, summary in layer_stats.items():
                entry = stats.setdefault(layer_idx, {"sum_cos": 0.0, "sum_proj": 0.0, "count": 0})
                entry["sum_cos"] += summary["sum_cos"]
                entry["sum_proj"] += summary["sum_proj"]
                entry["count"] += summary["count"]
