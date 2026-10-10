"""Activation addition inside a vLLM engine, for fast all-position and prompt-only steering.

Adapted from loop_steer's vllm_hooks.py for gpt-oss on vLLM 0.31. Requirements:
  - enforce_eager=True (forward hooks do not run inside compiled CUDA graphs),
  - enable_prefix_caching=False (cached KV computed under another intervention would be reused),
  - VLLM_ALLOW_INSECURE_SERIALIZATION=1 (hooks are shipped to the worker with LLM.apply_model).

gpt-oss decoder blocks return (output, residual) and the residual stream after block l is
output + residual, so adding v to `output` adds v to resid_post(l), the same site as
natsteer.model.Steerer. Spec dicts:
  {"layer": l, "vec": [D] tensor, "positions": "all" | "prompt"}
"prompt" steers tokens processed in a prefill step (query length > 1 for their request),
which are the rendered prompt tokens; decode steps (one token per request) are not steered.
That matches Steerer(prefill_only=True) in HF generation.
"""

import functools
import os

import torch

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def _prefill_mask(n_tokens: int, device) -> torch.Tensor:
    from vllm.forward_context import get_forward_context

    meta = get_forward_context().attn_metadata
    if isinstance(meta, list):
        raise RuntimeError("micro-batched forward contexts are not supported")
    qsl = next(m for m in meta.values() if hasattr(m, "query_start_loc")).query_start_loc
    qlen = qsl[1:] - qsl[:-1]
    mask = torch.repeat_interleave(qlen > 1, qlen)
    out = torch.zeros(n_tokens, dtype=torch.bool, device=device)
    n = min(n_tokens, mask.shape[0])
    out[:n] = mask[:n]
    return out


def clear(model):
    for h in getattr(model, "_ns_handles", []):
        h.remove()
    model._ns_handles = []
    return True


def install(model, specs):
    """Remove previously installed hooks, then install `specs` (list of spec dicts)."""
    clear(model)
    layers = model.model.layers
    p = next(model.parameters())
    handles = []
    for spec in specs:
        vec = spec["vec"].float().to(device=p.device, dtype=torch.bfloat16)
        positions = spec.get("positions", "all")
        if positions not in ("all", "prompt"):
            raise ValueError(f"vLLM steering supports positions 'all' or 'prompt', not {positions!r}")

        def hook(module, args, output, vec=vec, positions=positions):
            out, res = output
            if positions == "all":
                model._ns_stats["steered"] += out.shape[0]
                return out + vec.to(out.dtype), res
            m = _prefill_mask(out.shape[0], out.device)
            model._ns_stats["steered"] += int(m.sum())
            return out + m.unsqueeze(-1).to(out.dtype) * vec.to(out.dtype), res

        handles.append(layers[spec["layer"]].register_forward_hook(hook))
    model._ns_handles = handles
    model._ns_stats = {"steered": 0}
    return len(handles)


def stats(model):
    """Token rows steered since install (summed over hooks); for checking masks."""
    return dict(getattr(model, "_ns_stats", {}))


def set_steering(llm, specs):
    """Install specs (possibly empty) in every worker of `llm`."""
    cpu_specs = [s | {"vec": s["vec"].detach().float().cpu()} for s in specs]
    return llm.apply_model(functools.partial(install, specs=cpu_specs))


def make_steerable_llm(**kwargs):
    from natsteer.config import MODEL_ID
    from natsteer.vllm_utils import make_llm

    kwargs.setdefault("max_model_len", 4096)
    return make_llm(MODEL_ID, enforce_eager=True, enable_prefix_caching=False, **kwargs)
