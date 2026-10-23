"""Load the YAML config into a nested dict with attribute-style access."""
from pathlib import Path
import yaml


class Cfg(dict):
    def __getattr__(self, key):
        val = self[key]
        return Cfg(val) if isinstance(val, dict) else val


def load_config(path: str = "config.yaml", **overrides) -> Cfg:
    with open(path) as f:
        cfg = Cfg(yaml.safe_load(f))
    for dotted, value in overrides.items():
        if value is None:
            continue
        node = cfg
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = value
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg
