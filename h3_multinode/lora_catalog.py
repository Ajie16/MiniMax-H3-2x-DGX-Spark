"""Allowlisted LoRA catalog for two-Spark MiniMax H3 serving.

Slice 1 parses and validates the on-disk catalog. Request-time serving
hooks (apply_request_lora) land in a later slice.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ADAPTER_NAME_RE = re.compile(r"^[a-z0-9._-]+$")
RELATIVE_ADAPTER_PATH_RE = re.compile(r"^[a-z0-9._-]+/?$")
ALLOWED_FORMATS = frozenset({"peft"})
ALLOWED_PROFILES = frozenset({"style", "turbo"})
PEFT_WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin")
FORBIDDEN_TARGET_MODULES = frozenset(
    {
        "linear",
        "condition_proj",
        "video_patch_proj",
        "audio_patch_proj",
        "proj_in",
        "proj_out",
        "video_out",
        "audio_out",
    }
)


class CatalogError(ValueError):
    """Fail-closed catalog or profile error."""


@dataclass(frozen=True)
class AdapterEntry:
    name: str
    relative_path: str
    resolved_path: Path
    format: str
    profile: str
    default_scale: float
    recommended_steps: int | None
    recommended_flow_shift: float | None
    recommended_audio_flow_shift: float | None
    allowed_steps: tuple[int, ...] | None
    sha256_manifest: str | None
    source: str | None
    source_file: str | None
    trigger: str | None
    notes: str | None

    @property
    def manifest_path(self) -> Path | None:
        if not self.sha256_manifest:
            return None
        return lora_dir() / self.sha256_manifest

    def to_request(self) -> Any:
        from vllm_omni.lora.request import LoRARequest
        from vllm_omni.lora.utils import stable_lora_int_id

        path = str(self.resolved_path)
        return LoRARequest(self.name, int(stable_lora_int_id(path)), path)


def _check_scale(value: Any, *, source: str) -> float:
    try:
        scale = float(value)
    except (TypeError, ValueError) as exc:
        raise CatalogError(f"{source} must be a float in (0, 8]") from exc
    if not (0.0 < scale <= 8.0):
        raise CatalogError(f"{source} must be a float in (0, 8]")
    return scale


def resolved_scale(
    *,
    request_scale: float | None = None,
    catalog_default: float | None = None,
) -> float:
    """Resolve LoRA scale: request > explicit env > catalog default > 1.0."""
    if request_scale is not None:
        return _check_scale(request_scale, source="request scale")
    if "H3_LORA_SCALE" in os.environ:
        return _check_scale(os.environ["H3_LORA_SCALE"], source="H3_LORA_SCALE")
    if catalog_default is not None:
        return _check_scale(catalog_default, source="catalog default_scale")
    return 1.0


def lora_dir() -> Path:
    raw = os.environ.get("H3_LORA_DIR", "")
    if not raw:
        raise CatalogError("H3_LORA_DIR is required when H3_LORA_MODE is not off")
    path = Path(raw)
    if not path.is_absolute():
        raise CatalogError("H3_LORA_DIR must be an absolute path")
    return path


def catalog_path() -> Path:
    raw = os.environ.get("H3_LORA_CATALOG", "")
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            raise CatalogError("H3_LORA_CATALOG must be an absolute path")
        return path
    return lora_dir() / "catalog.json"


def _resolved_under_lora_dir(relative: str) -> Path:
    if not RELATIVE_ADAPTER_PATH_RE.match(relative):
        raise CatalogError(
            "adapter path must be a single relative directory name "
            f"(got {relative!r})"
        )
    root = lora_dir().resolve()
    resolved = (root / relative.rstrip("/")).resolve()
    if not resolved.is_relative_to(root):
        raise CatalogError(f"adapter path escapes H3_LORA_DIR: {relative}")
    return resolved


def _optional_float(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise CatalogError(f"{field} must be a number") from exc


def _optional_int(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise CatalogError(f"{field} must be an integer")
    return value


def _optional_steps(value: Any, *, field: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise CatalogError(f"{field} must be a non-empty list of integers")
    steps = tuple(_optional_int(item, field=field) for item in value)
    if any(step < 1 for step in steps):
        raise CatalogError(f"{field} entries must be >= 1")
    return steps


def _parse_adapter(name: str, body: Mapping[str, Any]) -> AdapterEntry:
    if not ADAPTER_NAME_RE.match(name):
        raise CatalogError(f"invalid adapter name {name!r}")
    if not isinstance(body, Mapping):
        raise CatalogError(f"adapter {name} must be an object")
    relative = body.get("path")
    if not isinstance(relative, str) or not relative:
        raise CatalogError(f"adapter {name} is missing path")
    fmt = body.get("format", "peft")
    if fmt not in ALLOWED_FORMATS:
        raise CatalogError(f"adapter {name} format must be peft")
    profile = body.get("profile")
    if profile not in ALLOWED_PROFILES:
        raise CatalogError(f"adapter {name} profile must be style or turbo")
    default_scale = _check_scale(
        body.get("default_scale", 1.0),
        source=f"adapter {name} default_scale",
    )
    manifest = body.get("sha256_manifest")
    if manifest is not None and not isinstance(manifest, str):
        raise CatalogError(f"adapter {name} sha256_manifest must be a string")
    if isinstance(manifest, str):
        if Path(manifest).is_absolute() or ".." in Path(manifest).parts:
            raise CatalogError(f"adapter {name} sha256_manifest is not a safe relative path")
    return AdapterEntry(
        name=name,
        relative_path=relative,
        resolved_path=_resolved_under_lora_dir(relative),
        format=str(fmt),
        profile=str(profile),
        default_scale=default_scale,
        recommended_steps=_optional_int(
            body.get("recommended_steps"), field=f"adapter {name} recommended_steps"
        ),
        recommended_flow_shift=_optional_float(
            body.get("recommended_flow_shift"),
            field=f"adapter {name} recommended_flow_shift",
        ),
        recommended_audio_flow_shift=_optional_float(
            body.get("recommended_audio_flow_shift"),
            field=f"adapter {name} recommended_audio_flow_shift",
        ),
        allowed_steps=_optional_steps(
            body.get("allowed_steps"), field=f"adapter {name} allowed_steps"
        ),
        sha256_manifest=manifest,
        source=body.get("source") if isinstance(body.get("source"), str) else None,
        source_file=body.get("source_file") if isinstance(body.get("source_file"), str) else None,
        trigger=body.get("trigger") if isinstance(body.get("trigger"), str) else None,
        notes=body.get("notes") if isinstance(body.get("notes"), str) else None,
    )


def load_catalog(path: Path | None = None) -> dict[str, AdapterEntry]:
    catalog_file = path or catalog_path()
    if not catalog_file.is_file():
        raise CatalogError(f"LoRA catalog is missing: {catalog_file}")
    try:
        payload = json.loads(catalog_file.read_text())
    except json.JSONDecodeError as exc:
        raise CatalogError(f"LoRA catalog is not valid JSON: {catalog_file}") from exc
    if not isinstance(payload, dict):
        raise CatalogError("LoRA catalog must be a JSON object")
    adapters = payload.get("adapters")
    if not isinstance(adapters, dict) or not adapters:
        raise CatalogError("LoRA catalog adapters must be a non-empty object")
    parsed = {name: _parse_adapter(name, body) for name, body in adapters.items()}
    return parsed


def resolve_adapter(name: str, *, catalog: Mapping[str, AdapterEntry] | None = None) -> AdapterEntry:
    if not ADAPTER_NAME_RE.match(name):
        raise CatalogError(f"invalid adapter name {name!r}")
    adapters = catalog if catalog is not None else load_catalog()
    try:
        return adapters[name]
    except KeyError as exc:
        raise CatalogError(f"unknown LoRA adapter {name!r}") from exc


def peft_files_present(entry: AdapterEntry) -> bool:
    config = entry.resolved_path / "adapter_config.json"
    if not config.is_file():
        return False
    return any((entry.resolved_path / weight).is_file() for weight in PEFT_WEIGHT_NAMES)


def assert_peft_layout(entry: AdapterEntry) -> None:
    if not entry.resolved_path.is_dir():
        raise CatalogError(f"adapter directory is missing: {entry.resolved_path}")
    config_path = entry.resolved_path / "adapter_config.json"
    if not config_path.is_file():
        raise CatalogError(f"adapter {entry.name} is missing adapter_config.json")
    if not any((entry.resolved_path / weight).is_file() for weight in PEFT_WEIGHT_NAMES):
        raise CatalogError(
            f"adapter {entry.name} is missing adapter_model.safetensors or adapter_model.bin"
        )
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise CatalogError(f"adapter {entry.name} adapter_config.json is not valid JSON") from exc
    if not isinstance(config, dict):
        raise CatalogError(f"adapter {entry.name} adapter_config.json must be an object")
    for required in ("r", "lora_alpha", "target_modules"):
        if required not in config:
            raise CatalogError(f"adapter {entry.name} adapter_config.json missing {required}")
    targets = config.get("target_modules")
    if not isinstance(targets, list) or not all(isinstance(item, str) for item in targets):
        raise CatalogError(f"adapter {entry.name} target_modules must be a list of strings")
    forbidden = FORBIDDEN_TARGET_MODULES.intersection(targets)
    if forbidden:
        raise CatalogError(
            f"adapter {entry.name} target_modules include forbidden names: "
            + ", ".join(sorted(forbidden))
        )


def _allow_turbo() -> bool:
    value = os.environ.get("H3_LORA_ALLOW_TURBO", "false")
    if value not in {"true", "false"}:
        raise CatalogError("H3_LORA_ALLOW_TURBO must be true or false")
    return value == "true"


def selected_adapters() -> list[AdapterEntry]:
    """Adapters preflight must hash: static name, or every catalog entry in request mode."""
    adapters = validate_profile_from_env()
    if adapters is None:
        return []
    mode = os.environ.get("H3_LORA_MODE", "off")
    if mode == "static":
        return [resolve_adapter(os.environ["H3_LORA_NAME"], catalog=adapters)]
    return list(adapters.values())


CLIENT_PATH_KEYS = ("path", "lora_path", "local_path", "lora_local_path", "int_id", "lora_int_id")
CLIENT_NAME_KEYS = ("name", "lora_name", "adapter")
CLIENT_SCALE_KEYS = ("scale", "lora_scale")
LORA_ENV_KEYS = (
    "H3_LORA_MODE",
    "H3_LORA_DIR",
    "H3_LORA_CATALOG",
    "H3_LORA_NAME",
    "H3_LORA_SCALE",
    "H3_MAX_CPU_LORAS",
    "H3_LORA_ALLOW_TURBO",
)


def _has_any_key(body: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    return any(key in body and body[key] is not None for key in keys)


def _name_only(lora_body: Mapping[str, Any]) -> str:
    if _has_any_key(lora_body, CLIENT_PATH_KEYS):
        raise CatalogError("client-supplied lora.path is not allowed")
    name = None
    for key in CLIENT_NAME_KEYS:
        value = lora_body.get(key)
        if value:
            name = str(value)
            break
    if not name:
        raise CatalogError("lora.name is required")
    if not ADAPTER_NAME_RE.match(name):
        raise CatalogError(f"invalid adapter name {name!r}")
    return name


def _optional_request_scale(lora_body: Mapping[str, Any]) -> float | None:
    for key in CLIENT_SCALE_KEYS:
        if key in lora_body and lora_body[key] is not None:
            return _check_scale(lora_body[key], source="request scale")
    return None


def assert_lora_path_allowed(lora_path: str, *, expected: Path | None = None) -> Path:
    raw_dir = os.environ.get("H3_LORA_DIR")
    if not raw_dir:
        raise CatalogError("H3_LORA_DIR is not set; refusing LoRA path")
    root = Path(raw_dir).resolve()
    resolved = Path(lora_path).resolve()
    if not resolved.is_relative_to(root):
        raise CatalogError(f"LoRA path escapes H3_LORA_DIR: {lora_path}")
    if expected is not None and resolved != expected.resolve():
        raise CatalogError("LoRA path is not the catalog-resolved adapter directory")
    return resolved


def enforce_turbo_schedule(gen_params: Any, entry: AdapterEntry) -> None:
    if entry.profile != "turbo" or entry.recommended_steps is None:
        return
    current = getattr(gen_params, "num_inference_steps", None)
    if current is None:
        gen_params.num_inference_steps = entry.recommended_steps
        return
    allowed = entry.allowed_steps or (entry.recommended_steps,)
    if int(current) not in tuple(int(step) for step in allowed):
        raise CatalogError(
            f"turbo adapter requires num_inference_steps in {list(allowed)}"
        )


def apply_request_lora(
    lora_body: Any,
    *,
    enforce_eager: bool,
) -> tuple[Any, float | None]:
    """Resolve a Videos `lora` field to (LoRARequest|None, scale|None)."""
    mode = os.environ.get("H3_LORA_MODE", "off")
    if mode == "off":
        if lora_body:
            raise CatalogError("LoRA is disabled (H3_LORA_MODE=off)")
        return None, None
    if mode == "request" and not enforce_eager:
        raise CatalogError("H3_LORA_MODE=request requires H3_EXECUTION_MODE=eager")
    if lora_body is not None and not isinstance(lora_body, Mapping):
        raise CatalogError("Invalid lora field: expected an object.")

    catalog = load_catalog()
    if mode == "static":
        static_name = os.environ.get("H3_LORA_NAME", "")
        static = resolve_adapter(static_name, catalog=catalog)
        if not lora_body:
            return static.to_request(), resolved_scale(catalog_default=static.default_scale)
        if _optional_request_scale(lora_body) is not None:
            raise CatalogError(
                "static LoRA scale is frozen at init; restart or use H3_LORA_MODE=request"
            )
        name = _name_only(lora_body)
        if name != static.name:
            raise CatalogError(
                f"static LoRA is {static.name!r}; restart with H3_LORA_NAME={name} to switch"
            )
        return static.to_request(), resolved_scale(catalog_default=static.default_scale)

    if not lora_body:
        return None, None
    name = _name_only(lora_body)
    entry = resolve_adapter(name, catalog=catalog)
    if entry.profile == "turbo" and not _allow_turbo():
        raise CatalogError("turbo adapters require H3_LORA_ALLOW_TURBO=true")
    req_scale = _optional_request_scale(lora_body)
    return entry.to_request(), resolved_scale(
        request_scale=req_scale, catalog_default=entry.default_scale
    )


def validate_profile_from_env() -> dict[str, AdapterEntry] | None:
    """Validate H3_LORA_* env + catalog. mode=off returns None and does no I/O."""
    mode = os.environ.get("H3_LORA_MODE", "off")
    if mode not in {"off", "static", "request"}:
        raise CatalogError("H3_LORA_MODE must be off, static, or request")
    allow = os.environ.get("H3_LORA_ALLOW_TURBO", "false")
    if allow not in {"true", "false"}:
        raise CatalogError("H3_LORA_ALLOW_TURBO must be true or false")
    if mode == "off":
        return None
    if mode == "request" and os.environ.get("H3_EXECUTION_MODE", "compile") != "eager":
        raise CatalogError(
            "request mode requires H3_EXECUTION_MODE=eager until compile+switch is measured"
        )
    if "H3_LORA_SCALE" in os.environ:
        _check_scale(os.environ["H3_LORA_SCALE"], source="H3_LORA_SCALE")
    max_cpu = os.environ.get("H3_MAX_CPU_LORAS", "1")
    if not max_cpu.isdigit() or int(max_cpu) < 1:
        raise CatalogError("H3_MAX_CPU_LORAS must be an integer >= 1")
    adapters = load_catalog()
    turbo_names = [name for name, entry in adapters.items() if entry.profile == "turbo"]
    if mode == "static":
        name = os.environ.get("H3_LORA_NAME", "")
        if not name:
            raise CatalogError("H3_LORA_NAME is required when H3_LORA_MODE=static")
        entry = resolve_adapter(name, catalog=adapters)
        if entry.profile == "turbo" and not _allow_turbo():
            raise CatalogError("turbo adapters require H3_LORA_ALLOW_TURBO=true")
    elif turbo_names and not _allow_turbo():
        raise CatalogError("turbo adapters require H3_LORA_ALLOW_TURBO=true")
    return adapters


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else "validate-env"
    try:
        if command == "validate-env":
            validate_profile_from_env()
            return 0
        if command == "preflight-list":
            for entry in selected_adapters():
                manifest = entry.sha256_manifest or ""
                print(f"{entry.name}\t{entry.resolved_path}\t{manifest}")
            return 0
        print(
            "usage: python -m h3_multinode.lora_catalog validate-env|preflight-list",
            file=sys.stderr,
        )
        return 2
    except CatalogError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
