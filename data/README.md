### Data Access
Download the data with `gdown` from this [GDrive Folder](https://drive.google.com/drive/folders/1vgPSYsqDJyhv1s9aD09GgXqAeGm0Rb1V?usp=sharing)

```bash
cd data

# download FlyWire (~20GB)
gdown https://drive.google.com/uc?id=1Tpf-yfxvSFd6TLwLut8tcx90dWpLmykP
unzip flywire_full_v783.zip

# download MANC (~7GB)
gdown https://drive.google.com/uc?id=1X170_XfqRx44aXJQEDEkOE3heZ6ByfuZ
unzip manc_v1.0.zip

# download Hemibrain (~8GB)
gdown https://drive.google.com/uc?id=10wB5xEebzOGz90PTtkGltqHdUDVpNiZY
unzip hemibrain_v1.2.zip
```

### References
- [Hemibrain Data](https://www.janelia.org/project-team/flyem/hemibrain) via [Neuprint Python](https://connectome-neuprint.github.io/neuprint-python/docs/)
- [MANC Data](https://www.janelia.org/project-team/flyem/manc-connectome) via [Neuprint Python](https://connectome-neuprint.github.io/neuprint-python/docs/)
- [FlyWire Data](https://flywire.ai/) via [Codex](https://codex.flywire.ai/api/download)


### Point Affinity Transformer Checkpoints

```bash
mkdir ckpt && cd ckpt

# download FlyWire Model (~4.5GB)
gdown https://drive.google.com/uc?id=1MEbL3Yrci4ohQPPFPnvI9wDiq8b0wp5E

# download MANC Model (~4.5GB)
gdown https://drive.google.com/uc?id=1VvCxXsna-VZMTAtxotUBtRHlHs8ajyrX

# download Hemibrain Model (~4.5GB)
gdown https://drive.google.com/uc?id=1jFRWQF265Znk9-KK2SQQmHNoP1dTLDHu
```
