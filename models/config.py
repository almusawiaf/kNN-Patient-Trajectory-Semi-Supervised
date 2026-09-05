"""Configuration loading and path resolution."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
)


@dataclass
class Paths:
    """Resolved absolute paths for every directory the project touches."""

    mimic_dir: str
    project_root: str
    data_dir: str
    logs_dir: str
    results_dir: str

    def ensure(self) -> None:
        """Create the writable directories if they do not exist."""
        for d in (self.data_dir, self.logs_dir, self.results_dir):
            os.makedirs(d, exist_ok=True)

    def data(self, *parts: str) -> str:
        return os.path.join(self.data_dir, *parts)

    def results(self, *parts: str) -> str:
        return os.path.join(self.results_dir, *parts)

    def logs(self, *parts: str) -> str:
        return os.path.join(self.logs_dir, *parts)

    def mimic(self, filename: str) -> str:
        return os.path.join(self.mimic_dir, filename)


@dataclass
class Config:
    """Thin wrapper around the parsed YAML with resolved paths attached."""

    raw: dict[str, Any]
    paths: Paths = field(init=False)

    def __post_init__(self) -> None:
        p = self.raw["paths"]
        root = os.path.abspath(os.path.expanduser(p["project_root"]))

        def resolve(value: str) -> str:
            value = os.path.expanduser(value)
            return value if os.path.isabs(value) else os.path.join(root, value)

        self.paths = Paths(
            mimic_dir=os.path.abspath(os.path.expanduser(p["mimic_dir"])),
            project_root=root,
            data_dir=resolve(p["data_dir"]),
            logs_dir=resolve(p["logs_dir"]),
            results_dir=resolve(p["results_dir"]),
        )

    # -- section accessors ---------------------------------------------------
    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def task(self) -> dict[str, Any]:
        return self.raw["task"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def hierarchical(self) -> dict[str, Any]:
        return self.raw["hierarchical"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.raw["evaluation"]

    @property
    def runtime(self) -> dict[str, Any]:
        return self.raw["runtime"]

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.raw)

    def tag(self) -> str:
        """Short identifier used to name result files for this configuration."""
        return "{task}_icd{digits}_{sim}_k{k}".format(
            task=self.task["name"],
            digits=self.data["icd_digits"],
            sim=self.model["similarity"],
            k=self.model["k"],
        )


def load_config(path: str | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load YAML config, apply dotted-key overrides, resolve paths.

    Overrides use dotted keys, e.g. ``{"model.k": 50}``.
    """
    path = path or DEFAULT_CONFIG_PATH
    with open(path) as fh:
        raw = yaml.safe_load(fh)

    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        node = raw
        keys = dotted.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value

    return Config(raw=raw)