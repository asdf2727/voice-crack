"""
Train the causal ConvNeXt autoencoder with an ARD-VAE bottleneck on complex
STFTs of VCTK -- generation-based.

A Runner (run.py) owns the model bundle (STFT pair, encoder/decoder, prior,
config); the Trainer owns the optimization: an infinite random-batch loader,
`generations` optimizer steps, per-batch loss rows appended to <save>.csv,
a stats line + target-vs-recon PNG every `viz_every` generations or on Enter,
and checkpoints on 's' + Enter (plus at run end). Epochs survive only as a
fractional progress reference. --compile wraps the encode/decode hot paths.

    python train.py --root ../datasets/VCTK-Corpus-0.92 --compile
    python train.py --selftest        # synthetic tones, no VCTK needed
"""
import argparse
import csv
import select
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from datasets.batched import BatchCropDataset, InfiniteBatchCrops
from datasets.vctk import VCTKDataset
from loss.vae_loss import *
from loss.stft_loss import spectral_loss
from models.STFT import spec_rgb
from models.ae import BlockParams
from run import Runner


class Trainer:
    def __init__(self,
                 runner: Runner,
                 halflife: float = 1e6,
                 compile_model: bool = True,
                 save: str | Path | None = None):
        self.runner = runner
        self.halflife = halflife
        self.save_path = Path(save) if save else None
        # compile the hot paths only; state_dicts stay on the eager modules
        self._encode = (torch.compile(runner.enc.encode, dynamic=True)
                        if compile_model else runner.enc.encode)
        self._decode = (torch.compile(runner.dec, dynamic=True)
                        if compile_model else runner.dec)
        self.crop = runner.enc.latency + runner.dec.latency
        self.history: list[dict] = []
        # run-scoped knobs, refreshed by run(); kept on self for viz/baseline
        self.k_phase = 1.0
        self.k_vae = 1.0
        self.viz_every = 200
        self.batches_per_epoch = None
        self._last = None  # (target, recon) of the newest batch, for the viz PNG
        self._mark = (0, time.time())  # (generation, wall time) of the last viz
        self._stdin_ok = sys.stdin is not None and not sys.stdin.closed
        self._csv = self._writer = None

    def epoch(self, gen: int) -> float:
        """Fractional epochs seen after `gen` generations (progress reference)."""
        return gen / self.batches_per_epoch if self.batches_per_epoch else float("nan")

    def features(self, wave: torch.Tensor) -> torch.Tensor:
        """wave (B, S) -> (B, 2, T, F) in the model band, normalized."""
        return self.runner.wave_to_spec(wave)[1]

    def train_batch(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        mean, log_var = self._encode(x)
        self.runner.prior.update(mean, log_var, self.halflife)
        recon = self._decode(self.runner.enc.reparameterize(mean, log_var))
        target = x[..., self.crop:, :]
        mag, phs = spectral_loss(recon, target)
        kld = self.runner.prior.kld_loss(mean, log_var)
        self._last = (target.detach(), recon.detach())
        return mag, phs, kld

    def run(self,
            generations: int,
            loader: DataLoader,
            lr: float = 1e-3,
            k_phase: float = 1.0,
            k_vae: float = 1.0,
            vae_warmup: int = 0,
            viz_every: int = 200,
            batches_per_epoch: int | None = None) -> list[dict]:
        opt = torch.optim.Adam([*self.runner.enc.parameters(),
                                    *self.runner.dec.parameters()], lr=lr, fused=True)
        self.k_phase, self.k_vae = k_phase, k_vae
        self.viz_every = viz_every
        self.batches_per_epoch = batches_per_epoch
        if self.save_path and self._csv is None:
            self._csv = open(self.save_path.with_suffix(".csv"), "w", newline="")
            self._writer = csv.writer(self._csv)
            self._writer.writerow(["gen", "epoch", "mag_loss", "phs_loss",
                                   "vae_loss", "vae_mult"])

        self.runner.enc.train()
        self.runner.dec.train()
        batches = iter(loader)
        try:
            for gen in range(generations):
                x = self.features(next(batches))
                if x.shape[-2] <= self.crop:
                    continue

                mag_loss, phs_loss, vae_loss = self.train_batch(x)
                vae_mult = gen / vae_warmup if gen < vae_warmup else 1
                total_loss = mag_loss + k_phase * phs_loss + (k_vae * vae_mult) * vae_loss
                opt.zero_grad()
                total_loss.backward()
                opt.step()

                stats = {
                    "gen": gen,
                    "epoch": round(self.epoch(gen), 4),
                    "mag_loss": mag_loss.item(),
                    "phs_loss": phs_loss.item(),
                    "vae_loss": vae_loss.item(),
                    "vae_mult": vae_mult,
                }
                self.history.append(stats)
                if self._writer:
                    self._writer.writerow(stats.values())
                if viz_every and (gen + 1) % viz_every == 0:
                    self.viz(gen + 1)
                self.poll_input(gen + 1)
        finally:
            self.save()
            if self._csv:
                self._csv.close()
                self._csv = self._writer = None
        return self.history

    def poll_input(self, gen: int) -> None:
        """Non-blocking stdin: Enter -> viz now, 's' + Enter -> checkpoint."""
        while self._stdin_ok and select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            if line == "":  # EOF: stdin is not interactive, stop polling
                self._stdin_ok = False
                break
            if line.strip().lower() == "s":
                self.save()
                print(f"gen {gen}: checkpoint saved to {self.save_path}")
            self.viz(gen)

    def viz(self, gen: int) -> None:
        window = self.history[-min(len(self.history), self.viz_every or 100):]
        mean = {k: sum(r[k] for r in window) / len(window)
                for k in ("mag_loss", "phs_loss", "vae_loss")}
        mark_gen, mark_t = self._mark
        rate = (gen - mark_gen) / max(time.time() - mark_t, 1e-9)
        self._mark = (gen, time.time())
        prior = self.runner.prior
        print(f"gen {gen} (epoch {self.epoch(gen):.3f}) "
              f"| mag {mean['mag_loss']:9.2f} "
              f"| phs {mean['phs_loss']:9.2f} (x{self.k_phase:.2f}) "
              f"| vae {mean['vae_loss']:9.2f} "
              f"(x{self.k_vae * self.history[-1]['vae_mult']:.3f}) "
              f"| relevant dims {len(prior.relevant_dims())}/{prior.var.numel()} "
              f"| {rate:.1f} gen/s")
        if self.save_path and self._last:
            target, recon = self._last
            top, bottom = spec_rgb(target[0])[::-1], spec_rgb(recon[0])[::-1]
            sep = np.ones((2, top.shape[1], 3))
            plt.imsave(self.save_path.with_suffix(".png"),
                       np.concatenate([top, sep, bottom]).clip(0.0, 1.0))
        if self._csv:
            self._csv.flush()

    def silence_baseline(self, batches) -> float:
        """Average loss of predicting zeros over a finite batch iterable."""
        tot = n = 0
        with torch.no_grad():
            for wave in batches:
                x = self.features(wave)
                if x.shape[-2] <= self.crop:
                    continue
                xc = x[..., self.crop:, :]
                m, ph = spectral_loss(torch.zeros_like(xc), xc)
                tot += (m + self.k_phase * ph).item()
                n += 1
        return tot / max(n, 1)

    def save(self) -> None:
        if self.save_path:
            self.runner.save_model(self.save_path)


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
    batches = BatchCropDataset(_SortedTones(48, sr), batch_size=8)
    loader = DataLoader(InfiniteBatchCrops(batches, seed=0), batch_size=None)

    # stride=1 first stage = pointwise stem + full-resolution blocks: phase
    # needs fine-frequency capacity (see findings log).
    blocks = [BlockParams(8, depth=2, kernel=5, stride=1),
              BlockParams(16, depth=2, kernel=5)]
    runner = Runner.new_sym_model(blocks, latent_dim=16, stft_hop=64,
                                  stft_win_chunks=4, device=device)
    trainer = Trainer(runner, lr=2e-3, halflife=20_000.0)
    history = trainer.run(360, loader, k_phase=1.0, k_vae=1.0, vae_warmup=18,
                          viz_every=60, batches_per_epoch=len(batches))

    ae = [r["mag_loss"] + r["phs_loss"] for r in history]
    first, last = np.mean(ae[:30]), np.mean(ae[-60:])
    assert last < first, f"recon did not improve: {first:.1f} -> {last:.1f}"
    silence = trainer.silence_baseline(batches[i] for i in range(len(batches)))
    assert last < 0.75 * silence, f"no better than silence: {last:.1f} vs {silence:.1f}"
    prior = runner.prior
    assert not torch.allclose(prior.var, torch.ones_like(prior.var)), \
        "prior never updated"
    print(f"train selftest OK: ae {first:.1f} -> {last:.1f} "
          f"(silence baseline {silence:.1f}), "
          f"{len(prior.relevant_dims())}/16 relevant dims")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--root", type=str, default="../datasets/VCTK-Corpus-0.92")
    ap.add_argument("--mic", type=str, default="any")
    ap.add_argument("--generations", type=int, default=20_000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--win-chunks", type=int, default=4, help="n_fft = win_chunks * hop")
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--channels", type=int, nargs="+", default=[8, 16, 32],
                    help="channels per stage; each stage = Downsample + `depth` blocks")
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 2, 2],
                    help="ConvNeXt blocks per stage (pairs with --channels)")
    ap.add_argument("--strides", type=int, nargs="+", default=None,
                    help="per-stage frequency stride (default 1 for the first "
                         "stage, 2 for the rest: full-res blocks first)")
    ap.add_argument("--kernel", type=int, default=7)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--k-vae", type=float, default=1.0,
                    help="weight of the KL term (paper's beta); tune on recon quality")
    ap.add_argument("--k-phase", type=float, default=1.0,
                    help="weight of the phase-alignment term in spectral_loss")
    ap.add_argument("--warmup", type=int, default=2000,
                    help="generations of linear KL ramp-up (0 = none)")
    ap.add_argument("--halflife", type=float, default=1e6,
                    help="ARD prior EMA halflife, in latent frames")
    ap.add_argument("--max-seconds", type=float, default=2.0,
                    help="crop cap per batch; peak GPU memory is linear in batch * seconds")
    ap.add_argument("--viz-every", type=int, default=200,
                    help="print stats + write the recon PNG every k generations")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the encoder/decoder hot paths")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--init", type=str, default=None,
                    help="checkpoint (.pt) to start from; loads enc/dec/prior "
                         "(fresh optimizer). Architecture flags must match.")
    ap.add_argument("--save", type=str, default="../models/tcn_ardvae.pt")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.selftest:
        _selftest(device)
        return
    strides = args.strides or [1] + [2] * (len(args.channels) - 1)
    if not (len(args.channels) == len(args.depths) == len(strides)):
        ap.error("--channels, --depths and --strides must have the same length")

    vctk = VCTKDataset(args.root, mic_id=args.mic)
    sr = vctk[0][1]
    batches = BatchCropDataset(vctk, batch_size=args.batch,
                               max_samples=int(args.max_seconds * sr))
    loader = DataLoader(InfiniteBatchCrops(batches), batch_size=None,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0)

    blocks = [BlockParams(c, depth=d, kernel=args.kernel, stride=s)
              for c, d, s in zip(args.channels, args.depths, strides)]
    runner = Runner.new_sym_model(blocks, latent_dim=args.latent,
                                  stft_hop=args.hop,
                                  stft_win_chunks=args.win_chunks, device=device)
    if args.init:
        runner.load_data(torch.load(args.init, map_location=device))
        print(f"initialized from {args.init}")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for m in (runner.enc, runner.dec) for p in m.parameters())
    lat = runner.enc.latency + runner.dec.latency
    print(f"{len(vctk)} utterances @ {sr} Hz ({len(batches)} batches/epoch) | "
          f"{n_params/1e6:.2f}M params | latency {lat * args.hop / sr:.3f}s | {device}\n"
          f"controls: Enter = viz now, s + Enter = save checkpoint")

    trainer = Trainer(runner, halflife=args.halflife,
                      compile_model=args.compile, save=args.save)
    trainer.run(args.generations, loader, lr=args.lr,
                k_phase=args.k_phase, k_vae=args.k_vae,
                vae_warmup=args.warmup, viz_every=args.viz_every,
                batches_per_epoch=len(batches))


if __name__ == "__main__":
    main()
