# External artifacts

Large research artifacts are stored outside the Git repository. Replace the placeholders below after uploading each artifact.

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

Recommended hosts are a Hugging Face model repository for the checkpoint and Zenodo, OSF, or institutional object storage for prediction files and training logs.
