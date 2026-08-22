from __future__ import annotations

import json
from pathlib import Path

import pytest

from h3_multinode.lora_catalog import CatalogError, apply_request_lora


def _write_catalog(tmp_path: Path) -> None:
    adapter = tmp_path / "turbo4"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"x")
    (tmp_path / "catalog.json").write_text(
        json.dumps(
            {
                "version": 1,
                "adapters": {
                    "turbo4": {
                        "path": "turbo4/",
                        "format": "peft",
                        "profile": "turbo",
                        "default_scale": 1.0,
                        "recommended_steps": 4,
                    }
                },
            }
        )
    )


def test_static_omitted_lora_returns_catalog_request(tmp_path, monkeypatch):
    _write_catalog(tmp_path)
    monkeypatch.setenv("H3_LORA_DIR", str(tmp_path))
    monkeypatch.setenv("H3_LORA_MODE", "static")
    monkeypatch.setenv("H3_LORA_NAME", "turbo4")
    monkeypatch.setenv("H3_LORA_ALLOW_TURBO", "true")
    request, scale = apply_request_lora(None, enforce_eager=False)
    assert request.lora_name == "turbo4"
    assert request.lora_path == str((tmp_path / "turbo4").resolve())
    assert scale == 1.0


def test_request_mode_unknown_name(tmp_path, monkeypatch):
    _write_catalog(tmp_path)
    monkeypatch.setenv("H3_LORA_DIR", str(tmp_path))
    monkeypatch.setenv("H3_LORA_MODE", "request")
    monkeypatch.setenv("H3_LORA_ALLOW_TURBO", "true")
    with pytest.raises(CatalogError, match="unknown LoRA adapter"):
        apply_request_lora({"name": "missing"}, enforce_eager=True)
