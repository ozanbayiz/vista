# IDARVE

Investigating Demographic Attribute Representation in Vision Encoders.

IDARVE trains sparse autoencoders (SAEs) on vision encoder latents and uses linear probes to measure how demographic attributes (age, gender, race) are encoded in the learned sparse features. The project supports multiple modern SAE architectures and provides evaluation tools for feature-demographic alignment, ablation studies, and intersectional analysis.

## Project Structure

```
config/
  config.yaml              # root Hydra config
  data/                    # data module configs
  model/                   # model hyperparameter configs
  module/                  # Lightning module configs
src/
  main.py                  # Hydra + Lightning entrypoint
  data.py                  # FairFaceDataModule (LightningDataModule)
  evaluation.py            # post-training evaluation script
  datasets/
    fairface.py            # FairFaceDataset (HDF5 reader)
  models/
    sparse_autoencoder.py  # BaseSAE + Vanilla, TopK, BatchTopK, JumpReLU
    linear_probes.py       # LinearProbe (multi-task classification head)
  modules/
    sae.py                 # SAEModule (LightningModule)
    probe.py               # ProbeModule (LightningModule)
  setup_datasets/
    sae_latent_dataset.py  # encode VE latents through a trained SAE
datasets/
  patchSAE_style_analysis.ipynb  # analysis notebook
setup/
  download_checkpoints.sh  # fetch pretrained checkpoints
```

## Setup

Requires Python 3.11+. Use [uv](https://docs.astral.sh/uv/) for environment and dependency management.

```bash
uv venv
uv pip install -e ".[dev]"
```

For PyTorch with CUDA:

```bash
uv pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu126
```

## Data Preparation

Download pretrained checkpoints:

```bash
bash setup/download_checkpoints.sh
```

Generate SAE-encoded latent datasets from vision encoder HDF5 files:

```bash
python -m src.setup_datasets.sae_latent_dataset \
    --input data/ve_latent_fairface.hdf5 \
    --checkpoint checkpoints/best.ckpt \
    --output-raw data/sae_latent_fairface.hdf5 \
    --output-agg data/agg_sae_latent_fairface.hdf5
```

## Training

Training uses [Hydra](https://hydra.cc/) for configuration and [PyTorch Lightning](https://lightning.ai/) for the training loop, DDP, mixed precision, checkpointing, and logging.

### Train an SAE

The default config trains a BatchTopK SAE on FairFace vision encoder latents:

```bash
python -m src.main
```

Select an SAE variant by overriding the module config:

```bash
python -m src.main module.variant=topk module.top_k=32
python -m src.main module.variant=jumprelu module.l0_coeff=5e-5
python -m src.main module.variant=vanilla module.l1_coeff=3e-4
```

### Train Linear Probes

Switch the module, model, and data configs:

```bash
python -m src.main module=probe model=linear_probe data=fairface_labels
```

### Common Overrides

```bash
python -m src.main trainer.max_epochs=100 data.batch_size=256 trainer.precision=bf16-mixed
python -m src.main wandb.enabled=false
python -m src.main trainer.devices=4 trainer.strategy=ddp
```

## SAE Variants

All variants share a common architecture: pre-encoder bias subtraction, learned encoder weights, and a unit-norm-constrained decoder. Dead features are tracked and revived via auxiliary losses where applicable.

| Variant     | Sparsity Mechanism        | Key Hyperparameters             |
|-------------|---------------------------|---------------------------------|
| `vanilla`   | L1 penalty on activations | `l1_coeff`                      |
| `topk`      | Per-sample top-k          | `top_k`, `top_k_aux`, `aux_penalty` |
| `batchtopk` | Batch-level top-k         | `top_k`, `top_k_aux`, `aux_penalty` |
| `jumprelu`  | Learnable thresholds      | `bandwidth`, `l0_coeff`         |

## Evaluation

Run post-training evaluation on a trained SAE checkpoint:

```bash
python -m src.evaluation \
    --checkpoint path/to/best.ckpt \
    --hdf5 data/ve_latent_fairface.hdf5 \
    --output-dir eval_output
```

This produces `eval_output/evaluation_results.json` containing:

- **SAE health**: dead feature ratio, L0 statistics, activation frequency distribution
- **Reconstruction quality**: MSE, cosine similarity, explained variance (R^2)
- **Feature-demographic alignment**: per-attribute selectivity, top variant features
- **Ablation studies**: necessity (accuracy drop when top-k features are zeroed) and sufficiency (accuracy using only top-k features) for each demographic attribute
- **Downstream probe comparison**: logistic regression accuracy on raw VE latents versus SAE latent activations

## Configuration

Hydra composes the full config from four groups:

| Group    | Default                | Options                                |
|----------|------------------------|----------------------------------------|
| `module` | `sae`                  | `sae`, `probe`                         |
| `model`  | `sparse_autoencoder`   | `sparse_autoencoder`, `linear_probe`   |
| `data`   | `fairface_no_labels`   | `fairface_no_labels`, `fairface_labels`|

Lightning Trainer parameters live under the `trainer` key in `config/config.yaml`. WandB logging is controlled by the `wandb` key.

## Logging

Training metrics are logged to [Weights & Biases](https://wandb.ai/). SAE training logs reconstruction loss, sparsity loss, auxiliary loss, L0 norm, dead feature count, and validation cosine similarity. Probe training logs per-task loss, accuracy, and macro F1.

Disable WandB with `wandb.enabled=false`.
