import torch

from natsteer.sae import BatchTopKSAE, recon_stats


def _sae():
    torch.manual_seed(0)
    sae = BatchTopKSAE(16, 64, k=4)
    torch.nn.init.normal_(sae.encoder.weight)
    sae.threshold.fill_(0.5)
    return sae


def test_threshold_encoding_is_batch_independent():
    sae, x = _sae(), torch.randn(32, 16)
    full = sae.encode(x)
    # Equal up to matmul rounding across batch shapes.
    torch.testing.assert_close(full[:1], sae.encode(x[:1]))
    torch.testing.assert_close(full[5:9], sae.encode(x[5:9]))


def test_per_batch_topk_is_not():
    sae, x = _sae(), torch.randn(32, 16)
    assert not torch.equal(sae.encode(x, use_threshold=False)[:1], sae.encode(x[:1], use_threshold=False))


def test_recon_stats_identity():
    class Ident(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return x, x.abs()

    s = recon_stats(Ident(), torch.randn(100, 8), chunk=33)
    assert abs(s["frac_variance_explained"] - 1) < 1e-6 and abs(s["cossim"] - 1) < 1e-6 and s["rel_mse"] < 1e-12
