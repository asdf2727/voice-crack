"""
Train the causal TCN autoencoder (Vocos-style blocks) with an ARD-VAE
bottleneck on complex STFTs of VCTK.

Pipeline per step, everything (B, T, C):

    wave (B, S) -> STFTEncoder -> x (B, T, 2F) compressed complex STFT
    x -> TCNEncoder.encode -> per-frame (mean, log_var) -> reparameterized z
    z -> TCNDecoder -> recon (B, T - enc.latency - dec.latency, 2F)

Loss: summed-MSE reconstruction against the latency-cropped input (valid
convs eat frames off the front, so recon frame 0 corresponds to input frame
enc.latency + dec.latency) + kld_scale * KL against the EMA ARD prior, which
is updated every step from the sampled latents. kld_scale ramps linearly over
--warmup epochs so axes can't collapse before the encoder learns anything.

    python train.py --root ../datasets/VCTK-Corpus-0.92
    python train.py --selftest        # synthetic tones, no VCTK needed
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.batched import BatchCropDataset
from datasets.vctk import VCTKDataset
from loss.ard_prior import ARDPrior
from loss.vae_loss import autoencoder_loss, kld_loss
from models.STFT import STFTEncoder
from models.ae import TCNDecoder, TCNEncoder


def train(enc: TCNEncoder,
          dec: TCNDecoder,
          prior: ARDPrior,
          stft: STFTEncoder,
          loader,
          epochs: int,
          kld_scale: float,
          device: torch.device,
          lr: float = 1e-3,
          warmup: int | None = None,
          save: str | None = None) -> list[dict]:
    enc.to(device)
    dec.to(device)
    prior.to(device)
    stft.to(device)
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=lr)
    crop = enc.latency + dec.latency

    history = []
    for epoch in range(epochs):
        scale = kld_scale * min(1.0, (epoch + 1) / warmup) if warmup else kld_scale
        enc.train()
        dec.train()
        ae_sum = kld_sum = n = skipped = 0
        t0 = time.time()

        for wave in loader:
            wave = wave.to(device, non_blocking=True)
            x = stft(wave)
            #print(f"Batch {n}: {x.shape[-2]} chunks of size {x.shape[-1]}")
            if x.shape[-2] <= crop:  # sorted-by-length: earliest batches are shortest
                skipped += 1
                continue

            mean, log_var = enc.encode(x)
            z = enc.reparameterize(mean, log_var)
            recon = dec(z)
            ae = autoencoder_loss(x[..., crop:, :], recon)
            kld = kld_loss(mean, log_var, prior.get_var)
            loss = ae + scale * kld

            opt.zero_grad()
            loss.backward()
            opt.step()

            prior.collect_and_update(z)
            ae_sum += ae.item()
            kld_sum += kld.item()
            n += 1

        rel = prior.relevant_dims()
        stats = {"epoch": epoch, "ae": ae_sum / max(n, 1), "kld": kld_sum / max(n, 1),
                 "kld_scale": scale, "relevant": len(rel), "skipped": skipped}
        history.append(stats)
        print(f"epoch {epoch:3d} | ae {stats['ae']:10.2f} | kld {stats['kld']:9.2f} "
              f"(x{scale:.3f}) | relevant dims {len(rel)}/{prior.var.numel()} "
              f"| prior var [{prior.var.min():.4f}, {prior.var.max():.4f}] "
              f"| {time.time() - t0:.1f}s"
              + (f" | {skipped} too-short batches skipped" if skipped else ""))

        if save:
            torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(),
                        "prior": prior.state_dict(), "opt": opt.state_dict(),
                        "stats": stats}, save)
    return history


# ---------------------------------------------------------------------------
# Self-test: overfit a tiny model on synthetic tones, no dataset needed.
# ---------------------------------------------------------------------------

class _SortedTones:
    """Sine utterances of varied length, sorted by length like VCTKDataset."""

    def __init__(self, n: int, sr: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self._data = []
        for i in range(n):
            t = np.arange(int(sr * rng.uniform(0.5, 1.5))) / sr
            f = rng.uniform(100, 1000)
            self._data.append((0.5 * np.sin(2 * np.pi * f * t).astype(np.float32),
                               sr, f"p{i:03d}", str(i)))
        self._data.sort(key=lambda item: len(item[0]))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]


def _selftest(device: torch.device):
    torch.manual_seed(0)
    sr = 8000
    stft = STFTEncoder(hop=64, win_chunks=4)  # n_fft 256
    in_ch = (stft.n_fft // 2 + 1) * 2
    loader = DataLoader(BatchCropDataset(_SortedTones(48, sr), batch_size=8),
                        batch_size=None, shuffle=True)

    enc = TCNEncoder(in_ch, hidden=64, latent_dim=16, blocks=3, kernel=5)
    dec = TCNDecoder(in_ch, latent_dim=16, hidden=64, blocks=3, kernel=5)
    prior = ARDPrior(16, halflife=20_000.0)
    history = train(enc, dec, prior, stft, loader, epochs=10, lr=1e-3,
                    kld_scale=1.0, warmup=3, device=device)

    first, last = history[0]["ae"], history[-1]["ae"]
    assert last < first, f"recon did not improve: {first:.1f} -> {last:.1f}"
    with torch.no_grad():
        x = stft(next(iter(loader)).to(device))
        crop = enc.latency + dec.latency
        silence = autoencoder_loss(x[..., crop:, :],
                                   torch.zeros_like(x[..., crop:, :])).item()
    assert last < 0.7 * silence, f"no better than silence: {last:.1f} vs {silence:.1f}"
    assert not torch.allclose(prior.var, torch.ones_like(prior.var)), \
        "prior never updated"
    print(f"train selftest OK: ae {first:.1f} -> {last:.1f} "
          f"(silence baseline {silence:.1f}), "
          f"{history[-1]['relevant']}/16 relevant dims")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--root", type=str, default="../datasets/VCTK-Corpus-0.92")
    ap.add_argument("--mic", type=str, default="any")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--win-chunks", type=int, default=4, help="n_fft = win_chunks * hop")
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--kernel", type=int, default=7)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--kld-scale", type=float, default=1.0,
                    help="weight of the KL term (paper's beta); tune on recon quality")
    ap.add_argument("--warmup", type=int, default=5,
                    help="epochs of linear KL ramp-up (0 = none)")
    ap.add_argument("--halflife", type=float, default=1e6,
                    help="ARD prior EMA halflife, in latent frames")
    ap.add_argument("--max-seconds", type=float, default=4.0,
                    help="crop cap per batch; peak GPU memory is linear in this "
                         "(~3.4 GiB per 5.3s at the default model size)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save", type=str, default="../checkpoints/tcn_ardvae.pt")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.selftest:
        _selftest(device)
        return

    vctk = VCTKDataset(args.root, mic_id=args.mic)
    sr = vctk[0][1]
    loader = DataLoader(BatchCropDataset(vctk, batch_size=args.batch,
                                         max_samples=int(args.max_seconds * sr)),
                        batch_size=None, shuffle=True,
                        num_workers=args.workers, pin_memory=True)

    stft = STFTEncoder(hop=args.hop, win_chunks=args.win_chunks)
    in_ch = (stft.n_fft // 2 + 1) * 2
    enc = TCNEncoder(in_ch, hidden=args.hidden, latent_dim=args.latent,
                     blocks=args.blocks, kernel=args.kernel)
    dec = TCNDecoder(in_ch, latent_dim=args.latent, hidden=args.hidden,
                     blocks=args.blocks, kernel=args.kernel)
    prior = ARDPrior(args.latent, halflife=args.halflife)

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for m in (enc, dec) for p in m.parameters())
    lat = enc.latency + dec.latency
    print(f"{len(vctk)} utterances @ {sr} Hz | {n_params/1e6:.1f}M params | "
          f"in_ch {in_ch} | latency {lat} frames ({lat * args.hop / sr:.3f}s) | {device}")

    train(enc, dec, prior, stft, loader, epochs=args.epochs, lr=args.lr,
          kld_scale=args.kld_scale, warmup=args.warmup, device=device,
          save=args.save)


if __name__ == "__main__":
    main()
