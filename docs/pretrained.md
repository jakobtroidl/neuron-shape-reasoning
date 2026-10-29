# Pretrained models

| Name | Training data | Preset | Scale | Translate (µm) |
|---|---|---|---|---|
| `flywire` | FlyWire v783, optic lobe | `ae_d1024_m512`, depth 24 | 659.88367 | 20 |
| `hemibrain` | Hemibrain v1.2 | `ae_d1024_m512`, depth 24 | 659.88367 | 60 |
| `manc` | MANC v1.0 | `ae_d1024_m512`, depth 24 | 659.88367 | 70 |

```python
model = pat.load_pretrained("hemibrain", device="cuda")
```

On first use the training checkpoint (about 4.5 GB including optimizer state) is
downloaded from Google Drive with `gdown`. It is then converted to a 1.5 GB
`model.safetensors` plus `config.json` in `~/.cache/pat/<name>/` and the download is deleted.
Set `PAT_CACHE` to use a different directory.

`load_pretrained` also accepts a model directory (from `pat.save_pretrained`, `pat export`
or a training run) or a research `.pth` checkpoint.
