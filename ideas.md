### Full end-to-end:
```
stft -> vocos encoder -> VAE bottleneck -> vocos decoder -> ISTFT
```

- vae bottleneck:
  - simple linear to mean and logvar?
  - vocodec like downsampling?
  - see how qincodec does it

- vocos encoder/decoder:
  - 1D conv like original vocos? use 2*freq channels?
  - 2D crazy downscale chain?

- input representation:
  - stft complex, imag?
  - add more representations, linear layer to choose inputs?
  - use 2 * freq hidden dim for encoder?

- output representation:
  - stft complex imag?
  - logmag and R, I for phase?
  - use 3 * freq hidden dim for decoder?
