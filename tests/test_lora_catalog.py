from __future__ import annotations

import json
from pathlib import Path

import pytest

from h3_multinode.lora_catalog import (
    CatalogError,
    load_catalog,
    resolve_adapter,
    resolved_scale,
    validate_profile_from_env,
)


def _write_catalog(tmp_path: Path, adapters: dict) -> Path:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"version": 1, "adapters": adapters}))
    return catalog


def _turbo4_body() -> dict:
    return {
        "path": "turbo4/",
        "format": "peft",
        "profile": "turbo",
        "default_scale": 1.0,
        "recommended_steps": 4,
        "recommended_flow_shift": 6,
        "recommended_audio_flow_shift": 3,
        "sha256_manifest": "turbo4.sha256",
    }


def test_load_and_resolve_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("H3_LORA_DIR", str(tmp_path))
    _write_catalog(tmp_path, {"turbo4": _turbo4_body()})
    adapters = load_catalog()
    entry = resolve_adapter("turbo4", catalog=adapters)
    assert entry.profile == "turbo"
    assert entry.recommended_steps == 4
    assert entry.resolved_path == (tmp_path / "turbo4").resolve()
    assert entry.resolved_path.is_relative_to(tmp_path.resolve())


def test_rejects_path_escape(tmp_path, monkeypatch):
    monkeypatch.setenv("H3_LORA_DIR", str(tmp_path))
    _write_catalog(
        tmp_path,
        {"bad": {**_turbo4_body(), "path": "../escape/"}},
    )
    with pytest.raises(CatalogError, match="single relative directory"):
        load_catalog()


def test_resolved_scale_precedence(monkeypatch):
    monkeypatch.delenv("H3_LORA_SCALE", raising=False)
    assert resolved_scale() == 1.0
    assert resolved_scale(catalog_default=0.8) == 0.8
    monkeypatch.setenv("H3_LORA_SCALE", "1.0")
    assert resolved_scale(catalog_default=0.8) == 1.0
    assert resolved_scale(request_scale=0.5, catalog_default=0.8) == 0.5


def test_resolved_scale_rejects_out_of_range(monkeypatch):
    monkeypatch.setenv("H3_LORA_SCALE", "0")
    with pytest.raises(CatalogError, match=r"\(0, 8\]"):
        resolved_scale()
    monkeypatch.setenv("H3_LORA_SCALE", "9")
    with pytest.raises(CatalogError, match=r"\(0, 8\]"):
        resolved_scale()


def test_mode_off_skips_catalog(monkeypatch, tmp_path):
    monkeypatch.setenv("H3_LORA_MODE", "off")
    monkeypatch.delenv("H3_LORA_DIR", raising=False)
    assert validate_profile_from_env() is None


def test_static_turbo_requires_allow_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("H3_LORA_DIR", str(tmp_path))
    monkeypatch.setenv("H3_LORA_MODE", "static")
    monkeypatch.setenv("H3_LORA_NAME", "turbo4")
    monkeypatch.setenv("H3_LORA_ALLOW_TURBO", "false")
    _write_catalog(tmp_path, {"turbo4": _turbo4_body()})
    with pytest.raises(CatalogError, match="H3_LORA_ALLOW_TURBO=true"):
        validate_profile_from_env()


def test_static_turbo_allowed(tmp_path, monkeypatch):
    monkeypatch.setenv("H3_LORA_DIR", str(tmp_path))
    monkeypatch.setenv("H3_LORA_MODE", "static")
    monkeypatch.setenv("H3_LORA_NAME", "turbo4")
    monkeypatch.setenv("H3_LORA_ALLOW_TURBO", "true")
    _write_catalog(tmp_path, {"turbo4": _turbo4_body()})
    adapters = validate_profile_from_env()
    assert adapters is not None
    assert "turbo4" in adapters


def test_request_mode_requires_eager(tmp_path, monkeypatch):
    monkeypatch.setenv("H3_LORA_DIR", str(tmp_path))
    monkeypatch.setenv("H3_LORA_MODE", "request")
    monkeypatch.setenv("H3_EXECUTION_MODE", "compile")
    monkeypatch.setenv("H3_LORA_ALLOW_TURBO", "true")
    _write_catalog(tmp_path, {"turbo4": _turbo4_body()})
    with pytest.raises(CatalogError, match="H3_EXECUTION_MODE=eager"):
        validate_profile_from_env()


def test_unknown_static_name(tmp_path, monkeypatch):
    monkeypatch.setenv("H3_LORA_DIR", str(tmp_path))
    monkeypatch.setenv("H3_LORA_MODE", "static")
    monkeypatch.setenv("H3_LORA_NAME", "missing")
    monkeypatch.setenv("H3_LORA_ALLOW_TURBO", "true")
    _write_catalog(tmp_path, {"turbo4": _turbo4_body()})
    with pytest.raises(CatalogError, match="unknown LoRA adapter"):
        validate_profile_from_env()
