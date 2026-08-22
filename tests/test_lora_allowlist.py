from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from h3_multinode.lora_catalog import (
    CatalogError,
    apply_request_lora,
    assert_lora_path_allowed,
    enforce_turbo_schedule,
)


def _write_catalog(tmp_path: Path) -> Path:
    adapter = tmp_path / "turbo4"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"x")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
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
    return catalog


def test_assert_lora_path_requires_env(tmp_path, monkeypatch):
    monkeypatch.delenv("H3_LORA_DIR", raising=False)
    with pytest.raises(CatalogError, match="H3_LORA_DIR is not set"):
        assert_lora_path_allowed(str(tmp_path / "turbo4"))


def test_assert_lora_path_rejects_escape(tmp_path, monkeypatch):
    monkeypatch.setenv("H3_LORA_DIR", str(tmp_path))
    outside = tmp_path.parent / "escape"
    outside.mkdir(exist_ok=True)
    with pytest.raises(CatalogError, match="escapes H3_LORA_DIR"):
        assert_lora_path_allowed(str(outside))


def test_assert_lora_path_requires_catalog_equality(tmp_path, monkeypatch):
    monkeypatch.setenv("H3_LORA_DIR", str(tmp_path))
    allowed = tmp_path / "turbo4"
    other = tmp_path / "other"
    allowed.mkdir()
    other.mkdir()
    with pytest.raises(CatalogError, match="catalog-resolved"):
        assert_lora_path_allowed(str(other), expected=allowed)


def test_off_mode_rejects_client_lora(tmp_path, monkeypatch):
    monkeypatch.setenv("H3_LORA_MODE", "off")
    with pytest.raises(CatalogError, match="LoRA is disabled"):
        apply_request_lora({"name": "turbo4"}, enforce_eager=True)


def test_request_mode_requires_eager_flag(tmp_path, monkeypatch):
    _write_catalog(tmp_path)
    monkeypatch.setenv("H3_LORA_DIR", str(tmp_path))
    monkeypatch.setenv("H3_LORA_MODE", "request")
    monkeypatch.setenv("H3_LORA_ALLOW_TURBO", "true")
    with pytest.raises(CatalogError, match="requires H3_EXECUTION_MODE=eager"):
        apply_request_lora({"name": "turbo4"}, enforce_eager=False)


def test_static_rejects_scale_and_path(tmp_path, monkeypatch):
    _write_catalog(tmp_path)
    monkeypatch.setenv("H3_LORA_DIR", str(tmp_path))
    monkeypatch.setenv("H3_LORA_MODE", "static")
    monkeypatch.setenv("H3_LORA_NAME", "turbo4")
    monkeypatch.setenv("H3_LORA_ALLOW_TURBO", "true")
    with pytest.raises(CatalogError, match="scale is frozen"):
        apply_request_lora({"name": "turbo4", "scale": 0.5}, enforce_eager=True)
    with pytest.raises(CatalogError, match="lora.path is not allowed"):
        apply_request_lora({"name": "turbo4", "path": str(tmp_path / "turbo4")}, enforce_eager=True)


def test_turbo_schedule_rejects_wrong_steps(tmp_path, monkeypatch):
    _write_catalog(tmp_path)
    monkeypatch.setenv("H3_LORA_DIR", str(tmp_path))
    from h3_multinode.lora_catalog import resolve_adapter

    entry = resolve_adapter("turbo4")
    params = SimpleNamespace(num_inference_steps=20)
    with pytest.raises(CatalogError, match="num_inference_steps=4"):
        enforce_turbo_schedule(params, entry)
    params = SimpleNamespace(num_inference_steps=None)
    enforce_turbo_schedule(params, entry)
    assert params.num_inference_steps == 4
