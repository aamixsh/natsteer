"""SAE loading and reconstruction metrics.

andyrdt BatchTopK SAEs: saved with biases and threshold already rescaled, so they
take raw resid_post activations. Encoding uses the learned fixed threshold, which
makes codes independent of the batch. Training excluded token position 0 and,
with p=0.95, the system message, so metrics exclude both by default.
"""

import json

import torch
from huggingface_hub import hf_hub_download
from torch import nn

from natsteer.config import GOODFIRE_SAE_FILE, GOODFIRE_SAE_REPO, SAE_REPO, SAE_TRAINERS


class BatchTopKSAE(nn.Module):
    """Mirror of dictionary_learning.trainers.batch_top_k.BatchTopKSAE (andyrdt/gpt-oss branch)."""

    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_sae)
        self.decoder = nn.Linear(d_sae, d_model, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.register_buffer("k", torch.tensor(k, dtype=torch.int))
        self.register_buffer("threshold", torch.tensor(-1.0))

    @property
    def d_sae(self) -> int:
        return self.encoder.out_features

    def pre_acts(self, x):
        return torch.relu(self.encoder(x.to(self.encoder.weight.dtype) - self.b_dec))

    def encode(self, x, use_threshold: bool = True):
        acts = self.pre_acts(x)
        if use_threshold:
            return acts * (acts > self.threshold)
        flat = acts.flatten()
        top = flat.topk(int(self.k) * x.shape[0], sorted=False)
        return torch.zeros_like(flat).scatter_(-1, top.indices, top.values).reshape(acts.shape)

    def decode(self, f):
        return self.decoder(f) + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f

    @property
    def W_dec(self):
        """Decoder directions as rows, [d_sae, d_model]."""
        return self.decoder.weight.T


class TopKSAE(nn.Module):
    """Per-token TopK SAE, used for the Goodfire layer-15 SAE.

    The checkpoint holds encoder_linear.weight [d_sae, d_model], decoder.weight
    [d_sae, d_model] and decoder_bias; there is no encoder bias and no README
    spec, so whether to subtract decoder_bias before encoding is an option that
    the E0 smoke test settles empirically.
    """

    def __init__(self, W_enc, W_dec, b_dec, k: int, subtract_b_dec: bool = True):
        super().__init__()
        self.W_enc = nn.Parameter(W_enc)  # [d_model, d_sae]
        self.W_dec = nn.Parameter(W_dec)  # [d_sae, d_model]
        self.b_dec = nn.Parameter(b_dec)
        self.k, self.subtract_b_dec = k, subtract_b_dec

    @property
    def d_sae(self) -> int:
        return self.W_enc.shape[1]

    def encode(self, x):
        x = x.to(self.W_enc.dtype)
        pre = ((x - self.b_dec) if self.subtract_b_dec else x) @ self.W_enc
        top = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, top.indices, torch.relu(top.values))

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


def load_andyrdt_sae(layer: int, k: int = 64, device="cuda:0", dtype=torch.float32) -> BatchTopKSAE:
    sub = f"resid_post_layer_{layer}/trainer_{SAE_TRAINERS[k]}"
    cfg = json.load(open(hf_hub_download(SAE_REPO, f"{sub}/config.json")))["trainer"]
    assert cfg["layer"] == layer and cfg["k"] == k, cfg
    state = torch.load(hf_hub_download(SAE_REPO, f"{sub}/ae.pt"), map_location="cpu", weights_only=True)
    sae = BatchTopKSAE(cfg["activation_dim"], cfg["dict_size"], cfg["k"])
    sae.load_state_dict(state)
    if float(sae.threshold) < 0:
        raise ValueError(f"{sub}: threshold was never set, BatchTopK would fall back to per-batch top-k")
    sae.requires_grad_(False)
    return sae.to(device=device, dtype=dtype).eval()


def load_goodfire_sae(device="cuda:0", dtype=torch.float32, k: int = 32, subtract_b_dec: bool = True) -> TopKSAE:
    state = torch.load(hf_hub_download(GOODFIRE_SAE_REPO, GOODFIRE_SAE_FILE), map_location="cpu", weights_only=True)
    sae = TopKSAE(state["encoder_linear.weight"].T.contiguous(), state["decoder.weight"], state["decoder_bias"], k, subtract_b_dec)
    sae.requires_grad_(False)
    return sae.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def recon_stats(sae, x: torch.Tensor, chunk: int = 1024) -> dict:
    """Reconstruction metrics for activations x [n, d_model], matching dictionary_learning's eval."""
    n = x.shape[0]
    if n == 0:
        return {"n": 0}
    dev = next(sae.parameters()).device
    sums = dict(sq_err=0.0, sq_norm=0.0, cos=0.0, l2_ratio=0.0, l0=0.0)
    s_x, s_xx, s_e, s_ee = 0, 0, 0, 0
    for i in range(0, n, chunk):
        xb = x[i : i + chunk].to(dev, torch.float32)
        xh, f = sae(xb)
        xh = xh.float()
        err = xb - xh
        sums["sq_err"] += err.pow(2).sum().item()
        sums["sq_norm"] += xb.pow(2).sum().item()
        sums["cos"] += torch.cosine_similarity(xb, xh, dim=-1).sum().item()
        sums["l2_ratio"] += (xh.norm(dim=-1) / xb.norm(dim=-1)).sum().item()
        sums["l0"] += (f > 0).float().sum().item()
        s_x, s_xx = s_x + xb.sum(0), s_xx + xb.pow(2).sum(0)
        s_e, s_ee = s_e + err.sum(0), s_ee + err.pow(2).sum(0)
    var_x = (s_xx / n - (s_x / n) ** 2).sum().item()
    var_e = (s_ee / n - (s_e / n) ** 2).sum().item()
    return {
        "n": n,
        "frac_variance_explained": 1 - var_e / var_x,
        "rel_mse": sums["sq_err"] / sums["sq_norm"],
        "cossim": sums["cos"] / n,
        "l2_ratio": sums["l2_ratio"] / n,
        "l0": sums["l0"] / n,
        "rms_norm": (sums["sq_norm"] / n) ** 0.5,
    }
