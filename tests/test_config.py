from pathlib import Path

import pytest

from eeg_decoding.utils.config import load_config


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_config_inheritance_under_windows_default_encoding(tmp_path, monkeypatch, encoding):
    (tmp_path / "local.yaml").write_text(
        "embedding_dim: 128\nbatch_size: 2\nlabel: café\n", encoding=encoding)
    overlay = tmp_path / "alignment.yaml"
    overlay.write_text(
        "base: local.yaml\nbatch_size: 8\ngradient_accumulation_steps: 2\n",
        encoding=encoding,
    )
    original = Path.read_text

    def windows_read_text(path, encoding=None, errors=None):
        return original(path, encoding=encoding or "cp1252", errors=errors)

    monkeypatch.setattr(Path, "read_text", windows_read_text)
    assert load_config(overlay) == {
        "embedding_dim": 128,
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "label": "café",
    }
