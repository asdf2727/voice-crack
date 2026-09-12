import os
import random
import select
import sys

from datasets.vctk import VCTK_092, batch_wavs
from models import VoiceCrack
from modules.stft import *
from loss import *

VERSION = "v0.1"
if not os.path.exists(f"../../models/{VERSION}"):
    os.mkdir(f"../../models/{VERSION}")

class Trainer:
    def __init__(self,
                 hop: int, win_chunks: int,
                 device: torch.device = torch.device("cpu"),
                 lr: float = 1e-4,
                 vae_w: float = 1.0):
        self.device = device
        self.model = VoiceCrack(hop, win_chunks).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, fused=True)
        self.vae_w = torch.tensor(vae_w)
        self.dataset = VCTK_092("../../datasets")

    @torch.compile(fullgraph=True, dynamic=False)
    def _run_batch(self, x: Tensor):
        spec = self.model.enc(x)
        mean, log_var = self.model.spec_to_latent(spec)
        recon = self.model.latent_to_spec(mean, log_var)

        self.model.vae_prior.update(mean, log_var)

        mag, phs = mag_phs_loss(spec[..., self.model.latency:, :, :], recon)
        vae = self.model.vae_prior.kld_loss(mean, log_var)
        return mag, phs, vae

    def _run_epoch(self,
                   loader: torch.utils.data.DataLoader,
                   step_size: int = 4):
        since_last_step = 0
        self.opt.zero_grad()
        for batch in loader:
            since_last_step += batch.shape[0]
            batch = batch.to(self.device)
            mag, phs, vae = self._run_batch(batch)
            loss = mag + phs + vae * self.vae_w
            loss.backward()
            if since_last_step < step_size:
                continue

            self.opt.step()
            self.opt.zero_grad()
            since_last_step = 0
            print(f"{self.model.step_cnt}: {mag.item():.4f} + {phs.item():.4f} + {vae.item():.4f} = {loss.item():.4f}")
            self.model.step_cnt += 1
            self.check_input()

    def check_input(self) -> bool:
        """Non-blocking stdin: Enter -> viz now, 's' + Enter -> checkpoint."""
        while select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            if line == "":  # EOF: stdin is not interactive, stop polling
                raise OSError("stdin is not interactive")
            if line.strip().lower() == "s":
                self._save(f"step_{self.model.step_cnt}")
            if line.strip().lower() == "show":
                self.show_spec()
        return False

    def show_spec(self):
        wav = random.choice(self.dataset)[0].to(self.device)
        spec = self.model.enc(wav)
        show_hsv(self.model.enc.feats_to_hsv(spec[self.model.latency:, ...].cpu()))
        out_spec = self.model(spec).detach()
        show_hsv(self.model.dec.feats_to_hsv(out_spec.cpu()))

    def run_training(self,
                     epochs: int):
        loader = torch.utils.data.DataLoader(self.dataset, 4, True, collate_fn=batch_wavs, num_workers=4)
        for epoch in range(epochs):
            self._run_epoch(loader)
            self._save(f"epoch_{epoch+1}")

    def _save(self, name: str):
        path = f"../../models/{VERSION}/{name}.pt"
        self.model.save_model(path)
        sym = f"../../models/{VERSION}/latest.pt"
        if os.path.exists(sym): os.remove(sym)
        os.symlink(path, f"../../models/{VERSION}/latest.pt")


def main():
    trainer = Trainer(256, 4, torch.device("cuda"))
    if os.path.exists(f"../../models/{VERSION}/latest.pt"):
        trainer.model.load_checkpoint(f"../../models/{VERSION}/latest.pt")
    else:
        print("No checkpoint found. Training from scratch.")
    trainer.run_training(10)

if __name__ == "__main__":
    main()