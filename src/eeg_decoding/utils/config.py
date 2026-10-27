from pathlib import Path
import yaml

def load_config(path):
    path = Path(path)
    # Windows PowerShell 5 writes UTF-8 with a BOM. Decode explicitly so
    # Windows' default ANSI code page cannot turn it into part of a YAML key.
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    if "base" in config:
        base = load_config(path.parent / config.pop("base"))
        base.update(config)
        config = base
    return config
