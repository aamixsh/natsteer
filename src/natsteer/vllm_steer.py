"""Activation addition inside a vLLM engine, for fast all-position and prompt-only steering.

Adapted from loop_steer's vllm_hooks.py for gpt-oss on vLLM 0.31. Requirements:
  - enforce_eager=True (forward hooks do not run inside compiled CUDA graphs),
  - enable_prefix_caching=False (cached KV computed under another intervention would be reused),
  - VLLM_ALLOW_INSECURE_SERIALIZATION=1 (hooks are shipped to the worker with LLM.apply_model).

gpt-oss decoder blocks return (output, residual) and the residual stream after block l is
output + residual, so adding v to `output` adds v to resid_post(l), the same site as
natsteer.model.Steerer. Spec dicts:
  {"layer": l, "vec": [D] tensor, "positions": "all" | "prompt" | "gen",
   "cap": optional float, "tag": optional str}
With "cap", the vector is scaled per token (capped_alpha): the largest alpha in CAP_ALPHAS whose steered
activation keeps the layer's SAE relative reconstruction error under max(the token's unsteered error, cap).
The SAE (andyrdt, k=64) is loaded once inside the worker. Chosen alphas are summed per "tag" in
model._ns_alpha_log, which survives re-installs (read it with alpha_log).
Directional ablation (anti-refusal, ported from loop_steer's vllm_hooks for gpt-oss):
  {"kind": "ablate", "dir": [D] tensor}
projects the unit direction out of every layer's input (hidden and residual) and every residual write (attn and
mlp outputs) at every position, so the stream never carries it. Layer l is called as layer(hidden, positions,
residual) and the stream entering it is hidden + residual.
"prompt" steers tokens processed in a prefill step (query length > 1 for their request),
which are the rendered prompt tokens; decode steps (one token per request) are not steered.
That matches Steerer(prefill_only=True) in HF generation. "gen" is the complement: only decode steps, i.e. the
generated tokens.
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


CAP_ALPHAS = (1.0, 0.75, 0.5, 0.25, 0.0)


@torch.no_grad()
def capped_alpha(h0, vec, sae, cap, alphas=CAP_ALPHAS, chunk=512):
    """Per-token steering scale for activations h0 [n, d] and vector vec [d]: the largest alpha in `alphas`
    (descending, last one 0) such that the SAE's relative reconstruction error of h0 + alpha * vec stays
    at or below max(error of h0, cap). Returns float32 [n]."""
    assert alphas[-1] == 0.0 and list(alphas) == sorted(alphas, reverse=True)
    a = torch.tensor(alphas, device=h0.device, dtype=torch.float32)
    v = vec.to(h0.device).float()
    dt = sae.encoder.weight.dtype
    out = torch.empty(h0.shape[0], device=h0.device, dtype=torch.float32)
    for s in range(0, h0.shape[0], chunk):
        x = h0[s : s + chunk].float()
        H = (x[None] + a[:, None, None] * v).reshape(-1, x.shape[-1])
        rec = sae.decode(sae.encode(H.to(dt))).float()
        err = (((H - rec) ** 2).sum(-1) / (H**2).sum(-1).clamp(min=1e-9)).reshape(len(alphas), -1)
        limit = torch.clamp(err[-1], min=cap)
        first = (err <= limit[None] * (1 + 1e-6)).float().argmax(0)  # alphas are descending, the last always passes
        out[s : s + chunk] = a[first]
    return out


def _sae(model, layer):
    cache = model.__dict__.setdefault("_ns_saes", {})
    if layer not in cache:
        from natsteer.sae import load_andyrdt_sae

        cache[layer] = load_andyrdt_sae(layer, 64, device=next(model.parameters()).device, dtype=torch.bfloat16)
    return cache[layer]


def alpha_log(model):
    """{tag: [sum of chosen alphas, tokens]} for capped specs since the engine started."""
    return {k: list(v) for k, v in getattr(model, "_ns_alpha_log", {}).items()}


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
    log = model.__dict__.setdefault("_ns_alpha_log", {})
    model._ns_stats = {"steered": 0}
    for spec in specs:
        if spec.get("kind") == "ablate":
            handles += _ablation_hooks(model, spec["dir"])
            continue
        vec = spec["vec"].float().to(device=p.device, dtype=torch.bfloat16)
        positions = spec.get("positions", "all")
        if positions not in ("all", "prompt", "gen"):
            raise ValueError(f"vLLM steering supports positions 'all', 'prompt' or 'gen', not {positions!r}")
        cap = spec.get("cap")
        sae = _sae(model, spec["layer"]) if cap is not None else None
        tag = spec.get("tag", f"L{spec['layer']}")

        def hook(module, args, output, vec=vec, positions=positions, cap=cap, sae=sae, tag=tag):
            out, res = output
            if cap is None:
                if positions == "all":
                    model._ns_stats["steered"] += out.shape[0]
                    return out + vec.to(out.dtype), res
                m = _prefill_mask(out.shape[0], out.device)
                if positions == "gen":
                    m = ~m
                model._ns_stats["steered"] += int(m.sum())
                return out + m.unsqueeze(-1).to(out.dtype) * vec.to(out.dtype), res
            h0 = out if res is None else out + res
            scale = capped_alpha(h0, vec, sae, cap)
            acc = log.setdefault(tag, [0.0, 0])
            acc[0] += float(scale.sum())
            acc[1] += int(scale.numel())
            if positions != "all":
                m = _prefill_mask(out.shape[0], out.device)
                scale = scale * (m if positions == "prompt" else ~m)
            model._ns_stats["steered"] += int((scale > 0).sum())
            return out + scale.unsqueeze(-1).to(out.dtype) * vec.to(out.dtype), res

        handles.append(layers[spec["layer"]].register_forward_hook(hook))
    model._ns_handles = handles
    return len(handles)


def _ablation_hooks(model, direction):
    p = next(model.parameters())
    u = direction.float()
    u = (u / u.norm()).to(device=p.device, dtype=p.dtype)

    def proj(x):
        return x - (x @ u).unsqueeze(-1) * u

    def pre(module, args):
        h, pos, res = args[0], args[1], args[2] if len(args) > 2 else None
        return (proj(h), pos, None if res is None else proj(res), *args[3:])

    def post(module, args, output):
        if isinstance(output, tuple):
            return (proj(output[0]), *output[1:])
        return proj(output)

    handles = []
    for layer in model.model.layers:
        handles.append(layer.register_forward_pre_hook(pre))
        handles.append(layer.attn.register_forward_hook(post))
        handles.append(layer.mlp.register_forward_hook(post))
    return handles


def stats(model):
    """Token rows steered since install (summed over hooks); for checking masks."""
    return dict(getattr(model, "_ns_stats", {}))


def set_steering(llm, specs):
    """Install specs (possibly empty) in every worker of `llm`."""
    cpu_specs = [s | ({"vec": s["vec"].detach().float().cpu()} if "vec" in s else {"dir": s["dir"].detach().float().cpu()}) for s in specs]
    return llm.apply_model(functools.partial(install, specs=cpu_specs))


def make_steerable_llm(**kwargs):
    """Create the engine BEFORE any CPU torch work in the process (loading SAEs, matmuls on CPU): vLLM forks
    its engine process, and a fork after torch's OpenMP pool is initialized hangs at start-up."""
    from natsteer.config import MODEL_ID
    from natsteer.vllm_utils import make_llm

    kwargs.setdefault("max_model_len", 4096)
    return make_llm(MODEL_ID, enforce_eager=True, enable_prefix_caching=False, **kwargs)
