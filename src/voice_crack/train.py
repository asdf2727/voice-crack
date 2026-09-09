from datasets.vctk import VCTK_092
from models import VoiceCrack
from modules.stft import *
from loss import *

VERSION = "v0"

class Trainer:
    def __init__(self,
                 enc: STFT,
                 device: torch.device = torch.device("cpu"),
                 lr: float = 1e-4):
        self.device = device
        self.enc = enc.to(device)
        self.dec = ISTFT(self.enc).to(device)
        self.model = VoiceCrack(self.enc.out_bins).to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, fused=True)
        self.vae_w = None
        self._train_batch = self.train_batch

    def train_batch(self, x: torch.Tensor):
        spec = self.enc(x)
        mean, log_var = self.model.spec_to_latent(spec)
        recon = self.model.latent_to_spec(mean, log_var)
        mag, phs = mag_phs_loss(spec[..., self.model.latency:, :, :], recon)
        vae = self.model.vae_prior.kld_loss(mean, log_var)
        loss = mag + phs + vae * self.vae_w
        loss.backward()

    def _run_epoch(self, step_size: int = 4):
        dataset = VCTK_092("../../datasets")
        loader = torch.utils.data.DataLoader(dataset)
        since_last_step = 0
        self.opt.zero_grad()
        for batch in loader:
            batch = batch[0].to(self.device)
            self._train_batch(batch)
            since_last_step += 1
            if since_last_step >= step_size:
                self.opt.step()
                self.opt.zero_grad()
                since_last_step = 0
                print(f"Step {self.model.step_cnt} done")
                self.model.step_cnt += 1

    def run_training(self,
                     batches: int,
                     vae_w: float = 1,
                     compile_kernel: bool = False):
        self.vae_w = vae_w
        self._train_batch = torch.compile(self.train_batch, fullgraph=True)\
                            if compile_kernel else self.train_batch
        for epoch in range(batches):
            self._run_epoch(epoch)
            self.model.save_model(f"../../models/{VERSION}/epoch_{epoch+1}.pt")

def main():
    trainer = Trainer(STFT(256, 4), torch.device("cuda"))

    trainer.run_training(batches=10000, vae_w=0.01)

if __name__ == "__main__":
    main()