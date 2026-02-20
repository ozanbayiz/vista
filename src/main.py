import logging
import subprocess

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _get_git_sha() -> str:
    """Return the current git commit SHA, or 'unknown' if not in a repo."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # ------------------------------------------------------------------
    # Reproducibility: seed all RNGs
    # ------------------------------------------------------------------
    seed = cfg.get("seed", 42)
    L.seed_everything(seed, workers=True)
    log.info("Global seed set to %d", seed)

    if cfg.get("deterministic", False):
        torch.use_deterministic_algorithms(True)
        log.info("Deterministic algorithms enabled (may be slower)")

    # ------------------------------------------------------------------
    # Log run metadata (git SHA, resolved config)
    # ------------------------------------------------------------------
    git_sha = _get_git_sha()
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    resolved_cfg["_git_sha"] = git_sha  # type: ignore[index]
    log.info("Git SHA: %s", git_sha)

    # ------------------------------------------------------------------
    # Instantiate module and data
    # ------------------------------------------------------------------
    module = hydra.utils.instantiate(cfg.module, _recursive_=False)
    datamodule = hydra.utils.instantiate(cfg.data)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    callbacks = [
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            filename="best-{epoch}-{step}",
        ),
        ModelCheckpoint(filename="last-{epoch}-{step}", save_last=True),
    ]

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------
    logger = (
        WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity"),
            config=resolved_cfg,
        )
        if cfg.wandb.get("enabled", True)
        else None
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = L.Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        callbacks=callbacks,
        logger=logger,
        deterministic=cfg.get("deterministic", False),
    )

    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
