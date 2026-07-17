"""
Hinge adversarial loss (Lim & Ye 1705.02894; MelGAN/Vocos vocoders).

Two terms for the min-max game between the reused-encoder critic and the
decoder-as-generator:
  - `disc_loss`  trains the critic: push real scores >= +1, fake <= -1.
  - `gen_loss`   trains the decoder: push the critic's fake score up.

Scores are per-frame logits of arbitrary shape (B?, T', 1); everything
reduces with a plain mean, so the reduction is scale-free in T and B.
"""
import torch
import torch.nn.functional as F


def disc_loss(real_score: torch.Tensor, fake_score: torch.Tensor) -> torch.Tensor:
    """Critic hinge loss. The +1 margin stops a sample's gradient once it is
    classified confidently past it, which keeps the critic from outrunning the
    generator. `fake_score` must be DETACHED: this term is the critic's alone and
    must not leak into the decoder.
    """
    return F.relu(1.0 - real_score).mean() + F.relu(1.0 + fake_score).mean()


def gen_loss(fake_score: torch.Tensor) -> torch.Tensor:
    """Non-saturating generator term: simply maximize the critic's fake score.
    No margin here (unlike the critic side) -- the generator always wants more,
    so the gradient never clips. `fake_score` must carry grad into the decoder.
    """
    return -fake_score.mean()


# ---------------------------------------------------------------------------
# Self-test: the hinge's margin behaviour and the min-max direction.
# ---------------------------------------------------------------------------

def _selftest():
    torch.manual_seed(0)
    real = torch.randn(4, 10, 1)
    fake = torch.randn(4, 10, 1)

    # A critic that clears the margin both ways pays nothing.
    assert disc_loss(torch.full_like(real, 5.0), torch.full_like(fake, -5.0)) == 0.0
    # ... and past the margin the gradient is flat (relu's dead side).
    r = torch.full_like(real, 5.0).requires_grad_(True)
    disc_loss(r, torch.full_like(fake, -5.0)).backward()
    assert r.grad.abs().sum() == 0.0

    # Critic loss is lower when real/fake are correctly separated.
    good = disc_loss(torch.full_like(real, 2.0), torch.full_like(fake, -2.0))
    bad = disc_loss(torch.full_like(real, -2.0), torch.full_like(fake, 2.0))
    assert good < bad

    # Generator prefers a higher critic score on its fake.
    assert gen_loss(torch.full_like(fake, 5.0)) < gen_loss(torch.full_like(fake, -5.0))
    # Its gradient pushes the score up (never clips).
    f = fake.clone().requires_grad_(True)
    gen_loss(f).backward()
    assert (f.grad < 0).all()  # d(-mean)/df < 0 -> ascending f lowers the loss

    print("disc_loss selftest OK")


if __name__ == "__main__":
    _selftest()
