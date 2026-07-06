"""
Train the causal TCN autoencoder with an ARD-VAE bottleneck on STFTs of VCTK.

Per step:   wave -> STFT -> model -> summed-MSE recon + kld_scale * KL,
            with the KL taken against the current N(0, beta/alpha) ARD prior.
Per epoch:  closed-form conjugate update of the prior from ~10K pooled latent
            frames (loss/ard_prior.py), then a report of per-axis prior
            variances and the effective dimensionality.

kld_scale ramps linearly over --warmup epochs so the prior can't collapse
axes before the encoder has learned anything (premature dim collapse).

    python train.py --root ../datasets/VCTK-Corpus-0.92
    python train.py --selftest        # synthetic data, no VCTK needed
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from audio.stft import STFT
from datasets.vctk import VCTKDataset
from loss.ard_prior import ARDPrior
from loss.vae_loss import autoencoder_loss, kld_loss
from models.ae import TCNAutoencoder


class RandomCropDataset(Dataset):
    """Fixed-length random crops of a (wave, sr, ...) utterance dataset, so the
    default DataLoader collate can stack them. Short utterances are zero-padded
    at the end."""

    def __init__(self, dataset, crop_len: int, seed: int = 0):
        self._ds = dataset
        self.crop_len = crop_len
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> torch.Tensor:
        wav = np.asarray(self._ds[idx][0], dtype=np.float32)
        if len(wav) > self.crop_len:
            start = self._rng.integers(len(wav) - self.crop_len + 1)
            wav = wav[start:start + self.crop_len]
        elif len(wav) < self.crop_len:
            wav = np.pad(wav, (0, self.crop_len - len(wav)))
        return torch.from_numpy(wav.copy())


def train(model: TCNAutoencoder, prior: ARDPrior, stft: STFT, loader,
          epochs: int, lr: float, kld_scale: float, warmup: int,
          device: torch.device, save: str | None = None) -> list[dict]:
    model.to(device)
    prior.to(device)
    stft.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    for epoch in range(epochs):
        scale = kld_scale * min(1.0, (epoch + 1) / warmup) if warmup else kld_scale
        model.train()
        ae_sum = kld_sum = n = 0
        t0 = time.time()

        for wave in loader:
            wave = wave.to(device, non_blocking=True)
            with torch.no_grad():
                x = stft(wave)

            out = model(x)
            ae = autoencoder_loss(x, out.recon)
            kld = kld_loss(out.mean, out.log_var, prior.alpha, prior.beta)
            loss = ae + scale * kld

            opt.zero_grad()
            loss.backward()
            opt.step()

            prior.collect(out.z)
            ae_sum += ae.item()
            kld_sum += kld.item()
            n += 1

        prior.update()
        rel = prior.relevant_dims()
        stats = {"epoch": epoch, "ae": ae_sum / n, "kld": kld_sum / n,
                 "kld_scale": scale, "relevant": len(rel)}
        history.append(stats)
        var = prior.target_var
        print(f"epoch {epoch:3d} | ae {stats['ae']:10.2f} | kld {stats['kld']:9.2f} "
              f"(x{scale:.3f}) | relevant dims {len(rel)}/{prior.latent_dim} "
              f"| prior var [{var.min():.4f}, {var.max():.4f}] "
              f"| {time.time() - t0:.1f}s")

        if save:
            torch.save({"model": model.state_dict(),
                        "prior": prior.state_dict(),
                        "opt": opt.state_dict(),
                        "stats": stats}, save)
    return history


# ---------------------------------------------------------------------------
# Self-test: overfit a tiny model on synthetic tones, no dataset needed.
# ---------------------------------------------------------------------------

class _SyntheticTones:
    """Utterances = sine sweeps + noise; mirrors VCTKDataset's tuple contract."""

    def __init__(self, n: int, sr: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self._data = []
        for i in range(n):
            t = np.arange(int(sr * rng.uniform(0.5, 1.5))) / sr
            f = rng.uniform(100, 1000)
            wav = (0.5 * np.sin(2 * np.pi * f * t)
                   + 0.01 * rng.standard_normal(len(t))).astype(np.float32)
            self._data.append((wav, sr, f"p{i:03d}", str(i)))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]


def _selftest(device: torch.device):
    torch.manual_seed(0)
    sr = 8000
    # power=1.0 (no compression) keeps the tone dominant over the noise floor,
    # so beating the predict-silence baseline is a clean "it learned" signal.
    stft = STFT(n_fft=256, hop=64, power=1.0)
    ds = RandomCropDataset(_SyntheticTones(48, sr), crop_len=sr)  # 1 s crops
    loader = DataLoader(ds, batch_size=8, shuffle=True)

    model = TCNAutoencoder(in_ch=stft.channels, latent_dim=16, hidden=64, blocks=4)
    prior = ARDPrior(16, max_samples=4000)
    history = train(model, prior, stft, loader, epochs=15, lr=1e-3,
                    kld_scale=1.0, warmup=3, device=device)

    first, last = history[0]["ae"], history[-1]["ae"]
    assert last < first, f"recon did not improve: {first:.1f} -> {last:.1f}"
    with torch.no_grad():
        x = stft(torch.stack([ds[i] for i in range(16)]).to(device))
        silence = autoencoder_loss(x, torch.zeros_like(x)).item()
    assert last < 0.3 * silence, f"no better than silence: {last:.1f} vs {silence:.1f}"
    assert not torch.allclose(prior.target_var,
                              torch.ones_like(prior.target_var)), "prior never updated"
    print(f"train selftest OK: ae {first:.1f} -> {last:.1f} "
          f"(silence baseline {silence:.1f}), "
          f"{history[-1]['relevant']}/16 relevant dims")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--root", type=str, default="../datasets/VCTK-Corpus-0.92")
    ap.add_argument("--mic", type=str, default="mic1")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=float, default=2.0, help="crop length, seconds")
    ap.add_argument("--n-fft", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--kld-scale", type=float, default=1.0,
                    help="weight of the KL term (paper's beta); tune on recon quality")
    ap.add_argument("--warmup", type=int, default=5,
                    help="epochs of linear KL ramp-up (0 = none)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save", type=str, default="../checkpoints/tcn_ardvae.pt")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.selftest:
        _selftest(device)
        return

    vctk = VCTKDataset(args.root, mic_id=args.mic)
    sr = vctk[0][1]  # native rate; VCTK is uniform, so crop 0 is representative
    ds = RandomCropDataset(vctk, crop_len=int(args.crop * sr))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True)

    stft = STFT(n_fft=args.n_fft, hop=args.hop)
    model = TCNAutoencoder(in_ch=stft.channels, latent_dim=args.latent,
                           hidden=args.hidden, blocks=args.blocks)
    prior = ARDPrior(args.latent)

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{len(vctk)} utterances @ {sr} Hz | {n_params/1e6:.1f}M params | "
          f"receptive field {model.enc_tcn.receptive_field} frames "
          f"({model.enc_tcn.receptive_field * args.hop / sr:.2f}s) | {device}")

    train(model, prior, stft, loader, epochs=args.epochs, lr=args.lr,
          kld_scale=args.kld_scale, warmup=args.warmup, device=device,
          save=args.save)


if __name__ == "__main__":
    main()
