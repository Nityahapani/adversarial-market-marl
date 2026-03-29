"""
Configuration loading and validation.

Supports YAML configs with deep merging for inheritance
(fast_debug.yaml overrides only the keys it specifies).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override into base (override wins on conflicts)."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def load_config(
    config_path: Union[str, Path],
    base_config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load a YAML config file, optionally merging over a base config.

    Args:
        config_path:      Path to the main config file.
        base_config_path: Optional base config to merge on top of.
                          If None and config is not default.yaml,
                          default.yaml is used as the base.
        overrides:        Dict of dot-notation overrides to apply last.
                          E.g., {"agents.execution.lambda_leakage": 0.8}

    Returns:
        Fully merged config dict.
    """
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Load base if this is not default.yaml itself
    if base_config_path is not None:
        with open(Path(base_config_path)) as f:
            base = yaml.safe_load(f)
        cfg = _deep_merge(base, cfg)
    else:
        default_path = config_path_obj.parent / "default.yaml"
        if default_path.exists() and config_path_obj.name != "default.yaml":
            with open(default_path) as f:
                base = yaml.safe_load(f)
            cfg = _deep_merge(base, cfg)

    # Apply dot-notation overrides
    if overrides:
        for dotpath, value in overrides.items():
            _set_nested(cfg, dotpath.split("."), value)

    return cfg


def _set_nested(d: Dict[str, Any], keys: List[str], value: Any) -> None:
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def validate_config(cfg: Dict[str, Any]) -> None:
    """Basic config validation — raises ValueError on invalid settings."""
    required_sections = ["environment", "agents", "networks", "training", "logging"]
    for sec in required_sections:
        if sec not in cfg:
            raise ValueError(f"Missing required config section: '{sec}'")

    lam = cfg["agents"]["execution"]["lambda_leakage"]
    if not 0.0 <= lam <= 10.0:
        raise ValueError(f"lambda_leakage must be in [0, 10], got {lam}")

    rollout = cfg["training"]["rollout_length"]
    mini = cfg["training"]["minibatch_size"]
    if mini > rollout:
        raise ValueError(f"minibatch_size ({mini}) must be <= rollout_length ({rollout})")


def save_config(cfg: Dict[str, Any], path: str) -> None:
    """Serialize config back to YAML."""
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
