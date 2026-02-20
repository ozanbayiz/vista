"""Hydra-driven evaluation entry point.

Wraps the existing evaluation pipeline (src.evaluation) in a Hydra config
so that evaluation runs are fully reproducible with saved configs, seed
management, and W&B logging -- identical to training runs.

Usage:
    python -m src.eval \
        checkpoint=/path/to/sae.ckpt \
        hdf5=/path/to/data.hdf5 \
        output_dir=results/eval_paligemma2

    # Or with an experiment config:
    python -m src.eval +experiment=sae_paligemma2_imagenet \
        checkpoint=/path/to/sae.ckpt \
        hdf5=/path/to/data.hdf5
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from src.evaluation import run_evaluation

log = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="eval", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("Resolved eval config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    run_evaluation(
        checkpoint_path=cfg.checkpoint,
        hdf5_path=cfg.hdf5,
        output_dir=cfg.get("output_dir", "eval_output"),
        batch_size=cfg.get("batch_size", 256),
        ablation_ks=list(cfg.get("ablation_ks", [10, 20, 50, 100])),
        mode=cfg.get("mode", "validation"),
    )


if __name__ == "__main__":
    main()
