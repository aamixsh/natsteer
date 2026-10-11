import torch

from natsteer.sae import BatchTopKSAE
from natsteer.vllm_steer import CAP_ALPHAS, capped_alpha


def toy_sae(d=8):
    """Identity-like SAE on the first 4 coordinates: reconstruction drops the last 4."""
    sae = BatchTopKSAE(d, 4, k=4)
    with torch.no_grad():
        sae.encoder.weight.zero_()
        sae.encoder.bias.zero_()
        sae.decoder.weight.zero_()
        for i in range(4):
            sae.encoder.weight[i, i] = 1.0
            sae.decoder.weight[i, i] = 1.0
        sae.threshold.fill_(0.0)
    return sae


def test_cap_picks_largest_allowed_alpha():
    sae = toy_sae()
    h0 = torch.zeros(3, 8)
    h0[:, 0] = 1.0  # perfectly reconstructed: error 0
    on = torch.zeros(8)
    on[1] = 1.0  # stays inside the SAE's span: error stays 0
    off = torch.zeros(8)
    off[5] = 1.0  # outside the span: error grows with alpha
    assert torch.allclose(capped_alpha(h0, on, sae, cap=0.0), torch.ones(3))
    assert torch.allclose(capped_alpha(h0, off, sae, cap=0.0), torch.zeros(3))
    # error of h0 + a*off is a^2 / (1 + a^2): 0.36 at a=0.75, 0.2 at a=0.5
    assert torch.allclose(capped_alpha(h0, off, sae, cap=0.25), torch.full((3,), 0.5))
    assert capped_alpha(h0, off, sae, cap=1.0).tolist() == [CAP_ALPHAS[0]] * 3


def test_cap_never_below_own_error():
    sae = toy_sae()
    h0 = torch.zeros(2, 8)
    h0[:, 6] = 1.0  # unsteered error already 1.0, above the cap
    off = torch.zeros(8)
    off[5] = 1.0
    assert torch.allclose(capped_alpha(h0, off, sae, cap=0.1), torch.ones(2))
