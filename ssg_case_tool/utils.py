from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists. Pass --force to overwrite.")
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def dump_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_spec(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_json(path)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("YAML spec requires PyYAML (`pip install pyyaml`).") from exc
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported spec format: {path.suffix}. Use .json/.yaml/.yml.")


def slug_to_path(slug: str) -> str:
    clean = slug.strip("/")
    return clean if clean else "index"

