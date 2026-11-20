# External artifacts

Large research artifacts are stored outside the Git repository. A reproducible
release is complete only after every TODO below has been replaced by a stable
URL and a verified SHA-256 checksum.

| Artifact | External URL | SHA-256 | Notes |
|---|---|---|---|
| BIRDSAI raw dataset | https://lila.science/datasets/conservationdrones | Provided by the dataset publisher | Do not duplicate in Git history |
| Final B2 epoch-1 checkpoint | `TODO` | `TODO` | Selected by validation mAP |
| Final tiled TestReal predictions | `TODO` | `TODO` | 384-pixel tiles, 20% overlap |
| Full Experiment B training logs | `TODO` | `TODO` | B1, B2, and B3 runs |

Generate a checksum with:

```bash
sha256sum /path/to/artifact
```

Verify a downloaded artifact with:

```bash
echo "EXPECTED_SHA256  downloaded-file" | sha256sum --check -
```

Recommended hosts are a Hugging Face model repository for the checkpoint and
Zenodo, OSF, or institutional object storage for prediction files and training
logs. Do not commit the BIRDSAI archive, checkpoints, Hugging Face cache, or
full prediction streams to Git.
