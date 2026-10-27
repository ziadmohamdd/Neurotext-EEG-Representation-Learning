from pathlib import Path
import yaml

def load_config(path):
    path = Path(path)
    config = yaml.safe_load(path.read_text()) or {}
    if "base" in config:
        base = load_config(path.parent / config.pop("base"))
        base.update(config)
        config = base
    return config
