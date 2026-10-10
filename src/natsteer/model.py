"""Loading gpt-oss-20b, capturing residual activations, and segment-aware steering."""

import time
from collections import Counter
from contextlib import contextmanager

import torch

from natsteer.config import MODEL_ID
from natsteer.harmony import SegmentTracker


def load_model(mode: str = "bf16", device: str = "cuda:0"):
    """Load gpt-oss-20b.

    mode="bf16" dequantizes the MXFP4 experts at load time (about 42 GB). Exact
    and differentiable, so it is the default for attribution and GCG work.
    mode="mxfp4" keeps the checkpoint format (about 13 GB) and uses the
    triton kernels from the Hub; faster and smaller, for bulk forward passes.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    kwargs = dict(dtype=torch.bfloat16, device_map=device)
    if mode == "bf16":
        kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
    elif mode != "mxfp4":
        raise ValueError(f"unknown mode {mode!r}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model.load_seconds = time.time() - t0
    return model, tokenizer


def get_layer(model, layer: int):
    return model.model.layers[layer]


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


@contextmanager
def capture_resid(model, layers, to_cpu: bool = False):
    """Record resid_post (decoder layer output) at each layer for the next forward pass.

    Yields a dict layer -> tensor [batch, seq, d_model], overwritten on every call.
    """
    acts: dict[int, torch.Tensor] = {}
    handles = []
    for layer in layers:
        def hook(_mod, _inp, out, layer=layer):
            h = _hidden(out).detach()
            acts[layer] = h.to("cpu", non_blocking=True) if to_cpu else h
        handles.append(get_layer(model, layer).register_forward_hook(hook))
    try:
        yield acts
    finally:
        for h in handles:
            h.remove()


class Steerer:
    """Add `vector` to resid_post at `layer`, only at tokens whose segment is in `segments`.

    Segments come from a SegmentTracker per batch row, fed with every forward
    call's input_ids, so the same mask logic covers teacher-forced passes and
    cached generation (prefill sees the prompt, decode steps one token each).
    segments=None steers every non-pad position. prefill_only=True steers every non-pad
    token of the first forward call after reset (the prompt prefill in generate), which
    is the "prompt" condition of natsteer.vllm_steer.
    """

    def __init__(self, model, tokenizer, layer: int, vector: torch.Tensor, segments=None, prefill_only: bool = False):
        self.model, self.tokenizer, self.layer = model, tokenizer, layer
        self.vector = vector
        self.segments = None if segments is None else frozenset(segments)
        self.prefill_only = prefill_only
        self.trackers: list[SegmentTracker] | None = None
        self.mask: torch.Tensor | None = None
        self.last_labels: list[list[str]] = []
        self.steered = Counter()  # segment -> number of steered positions, for logging
        self._handles = []

    def _pre_hook(self, _mod, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            raise ValueError("Steerer needs input_ids, not inputs_embeds")
        first = self.trackers is None
        if first:
            self.trackers = [SegmentTracker(self.tokenizer, pad_id=self.tokenizer.pad_token_id) for _ in range(ids.shape[0])]
        labels = [tr.feed_many(row.tolist()) for tr, row in zip(self.trackers, ids)]
        self.last_labels = labels
        if self.prefill_only:
            keep = [[first and lab != "pad" for lab in row] for row in labels]
        else:
            keep = [[(lab != "pad") if self.segments is None else (lab in self.segments) for lab in row] for row in labels]
        self.steered.update(lab for row, krow in zip(labels, keep) for lab, k in zip(row, krow) if k)
        self.mask = torch.tensor(keep, dtype=torch.bool, device=ids.device)

    def _layer_hook(self, _mod, _inp, out):
        h = _hidden(out)
        h = h + self.mask.unsqueeze(-1).to(h.dtype) * self.vector.to(h.device, h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def reset(self):
        """Forget per-row segment state. Call between independent forward passes or generate calls."""
        self.trackers, self.mask = None, None

    def __enter__(self):
        self.reset()
        self.steered.clear()
        self._handles = [
            self.model.register_forward_pre_hook(self._pre_hook, with_kwargs=True),
            get_layer(self.model, self.layer).register_forward_hook(self._layer_hook),
        ]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        self.reset()


@torch.no_grad()
def generate(model, tokenizer, prompts: list[list[int]], max_new_tokens: int = 1024, temperature: float = 1.0, steerer: Steerer | None = None, batch_size: int = 32, seed: int = 0) -> list[list[int]]:
    """Left-padded batched sampling from rendered prompt ids; returns generated ids without pads.

    Pass the Steerer that is already active (inside its `with` block); its segment
    state is reset for every batch.
    """
    pad = tokenizer.pad_token_id
    torch.manual_seed(seed)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    results: list[list[int]] = [[] for _ in prompts]
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        width = max(len(prompts[i]) for i in idx)
        ids = torch.tensor([[pad] * (width - len(prompts[i])) + prompts[i] for i in idx], device=model.device)
        mask = (ids != pad).long()
        if steerer is not None:
            steerer.reset()
        out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=temperature > 0, temperature=temperature if temperature > 0 else None, top_p=1.0, pad_token_id=pad)
        for b, i in enumerate(idx):
            results[i] = [t for t in out[b, width:].tolist() if t != pad]
    return results
