"""
Configuration loading and management via OmegaConf / Hydra.

Handles hierarchical config merging: base.yaml → dataset.yaml → model.yaml → CLI overrides.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


# ──── Defaults applied to every config ────

_DEFAULTS = {
    "seed": 42,
    "device": "auto",  # "auto", "cuda", "cpu"
    "precision": "fp32",  # "fp32", "fp16", "bf16"
    "num_workers": 4,
    "pin_memory": True,
    "project_name": "neurograph-conformer-rl",
    "wandb": {
        "enabled": False,
        "project": "neurograph-conformer-rl",
        "entity": None,
        "tags": [],
    },
}


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a YAML config file and apply optional CLI overrides.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.
    overrides : list[str], optional
        Dot-notation overrides, e.g. ["model.d_model=128", "training.lr=1e-4"].

    Returns
    -------
    DictConfig
        Merged configuration object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    cfg = OmegaConf.load(path)

    # Merge with defaults (defaults are lower priority)
    defaults_cfg = OmegaConf.create(_DEFAULTS)
    cfg = OmegaConf.merge(defaults_cfg, cfg)

    # Apply CLI overrides (highest priority)
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    # Resolve interpolations
    OmegaConf.resolve(cfg)

    return cfg


def merge_configs(*configs: DictConfig | dict) -> DictConfig:
    """Merge multiple configs with later ones taking precedence.

    Parameters
    ----------
    *configs : DictConfig or dict
        Configuration objects to merge, in order of increasing priority.

    Returns
    -------
    DictConfig
        Merged configuration.
    """
    result = OmegaConf.create({})
    for cfg in configs:
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        result = OmegaConf.merge(result, cfg)
    return result


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Convert OmegaConf DictConfig to a plain Python dict (for serialization)."""
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Save configuration to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)


def print_config(cfg: DictConfig) -> None:
    """Pretty-print configuration using Rich."""
    try:
        from rich.console import Console
        from rich.syntax import Syntax

        console = Console()
        yaml_str = OmegaConf.to_yaml(cfg, resolve=True)
        syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=True)
        console.print(syntax)
    except ImportError:
        print(OmegaConf.to_yaml(cfg, resolve=True))
