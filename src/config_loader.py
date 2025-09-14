from __future__ import annotations
import os
import json
import yaml
from typing import Any, Dict
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML config file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Failed to load YAML: {path}", extra={"context": {"error": str(e)}})
        return {}

def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON config file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON: {path}", extra={"context": {"error": str(e)}})
        return {}

def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_config(paths: list[str]) -> Dict[str, Any]:
    """
    Load and merge multiple config files (YAML or JSON).
    Later files override earlier ones.
    """
    config: Dict[str, Any] = {}
    for path in paths:
        if not os.path.exists(path):
            logger.warning(f"Config file not found: {path}")
            continue
        ext = os.path.splitext(path)[1].lower()
        part = load_yaml(path) if ext in (".yml", ".yaml") else load_json(path)
        config = deep_merge(config, part)
    logger.info("Config loaded", extra={"context": {"files": paths}})
    return config
