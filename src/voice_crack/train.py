import os
import random
import select
import sys
import csv

import torch
import torch.nn.functional as F
from random import randrange

from datasets.vctk import VCTK_092, SampleType
from models import VoiceCrack
from modules.stft import *
from loss import *

VERSION = "v0.2"
if not os.path.exists(f"../../models/{VERSION}"):
    os.mkdir(f"../../models/{VERSION}")

CROP = 3 * 48000

def batch_wavs(inputs: list[SampleType]) -> Tensor:
    wav_list = [inpt[0] for inpt in inputs]
    crops = []
    for wav in wav_list:
        if wav.numel() == CROP:
            crops.append(wav)
        elif wav.numel() < CROP:
            crops.append(F.pad(wav, (0, CROP - wav.numel())))
        else:
            offset = randrange(wav.numel() - CROP)
            crops.append(wav[offset:offset + CROP])
    return torch.stack(crops)

class Trainer:
    def __init__(self,
                 model: VoiceCrack,
                 device: torch.device = torch.device("cpu"),
                 lr: float = 1e-4,
                 vae_w: float = 1.0):
        self.device = device
        self.model = model.to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, fused=True)
        self.vae_w = torch.tensor(vae_w)
        self.dataset = VCTK_092("../../datasets", download=True)
        self.csv_buffer: list[list[str]] = []
        if not os.path.exists(f"../../models/{VERSION}/hist.csv"):
            self.csv_buffer.append(["step", "mag", "phs", "vae", "tot"])

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
                   step_size: int):
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
            step_data = [
                f"{self.model.step_cnt}",
                f"{mag.item():.4f}",
                f"{phs.item():.4f}",
                f"{vae.item():.4f}",
                f"{loss.item():.4f}"]
            self.csv_buffer.append(step_data)
            print(" - ".join(step_data))
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
        loader = torch.utils.data.DataLoader(self.dataset, 16, True, collate_fn=batch_wavs, num_workers=4)
        for epoch in range(epochs):
            self._run_epoch(loader, step_size=16)
            self._save(f"epoch_{epoch+1}")

    def _save(self, name: str):
        path = f"../../models/{VERSION}/{name}.pt"
        self.model.save_model(path)
        sym = f"../../models/{VERSION}/latest.pt"
        if os.path.exists(sym): os.remove(sym)
        os.symlink(path, f"../../models/{VERSION}/latest.pt")
        hist = f"../../models/{VERSION}/hist.csv"
        with open(hist, mode='a', newline='') as file:
            writer = csv.writer(file)
            for row in self.csv_buffer:
                writer.writerow(row)
            self.csv_buffer = []


def main():
    model = VoiceCrack(1024, 4, 512)
    trainer = Trainer(model, torch.device("cuda"))
    if os.path.exists(f"../../models/{VERSION}/latest.pt"):
        trainer.model.load_checkpoint(f"../../models/{VERSION}/latest.pt")
    else:
        print("No checkpoint found. Training from scratch.")
    trainer.run_training(10)

if __name__ == "__main__":
    main()
