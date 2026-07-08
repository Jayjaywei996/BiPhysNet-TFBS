import torch

from biphysnet.losses import FocalLoss, SafeNTXentLoss, bernoulli_kl


def test_safe_ntxent_loss_is_finite_scalar():
    loss_fn = SafeNTXentLoss(temperature=0.1)
    z_i = torch.randn(4, 8)
    z_j = torch.randn(4, 8)
    loss = loss_fn(z_i, z_j)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_focal_loss_is_finite_scalar():
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    logits = torch.tensor([0.2, -1.0, 1.5, 0.0])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = loss_fn(logits, labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_bernoulli_kl_is_finite():
    p = torch.tensor([0.2, 0.8])
    q = torch.tensor([0.3, 0.7])
    kl = bernoulli_kl(p, q)
    assert kl.shape == p.shape
    assert torch.isfinite(kl).all()
