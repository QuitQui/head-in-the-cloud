"""Config management for head-in-the-cloud.

Config file: ~/.hitc/config.toml

[default]
platform = "kaggle"
output_dir = "./output"
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w

_CONFIG_DIR = Path.home() / ".hitc"
_CONFIG_FILE = _CONFIG_DIR / "config.toml"

_DEFAULTS: dict = {
    "default": {
        "platform": "kaggle",
        "output_dir": "./output",
    }
}


def load() -> dict:
    if not _CONFIG_FILE.exists():
        return copy.deepcopy(_DEFAULTS)
    with _CONFIG_FILE.open("rb") as f:
        data = tomllib.load(f)
    merged = copy.deepcopy(_DEFAULTS)
    for section, values in data.items():
        merged.setdefault(section, {}).update(values)
    return merged


def save(config: dict) -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with _CONFIG_FILE.open("wb") as f:
        tomli_w.dump(config, f)


def get(key: str, section: str = "default") -> str | None:
    return load().get(section, {}).get(key)


def set_value(key: str, value: str, section: str = "default") -> None:
    config = load()
    config.setdefault(section, {})[key] = value
    save(config)


def show() -> None:
    config = load()
    print(f"Config file: {_CONFIG_FILE}")
    for section, values in config.items():
        print(f"\n[{section}]")
        for k, v in values.items():
            print(f"  {k} = {v!r}")
