"""Stable paths for packaged and writable MediLife knowledge resources."""
from __future__ import annotations

import os
from pathlib import Path
import shutil


PACKAGE_DATA_DIR = Path(__file__).resolve().parent / "data"


def runtime_data_dir() -> Path:
    configured = os.getenv("MEDILIFE_DATA_DIR")
    target = Path(configured).expanduser() if configured else Path.home() / ".medilife" / "data"
    target.mkdir(parents=True, exist_ok=True)
    return target


def bundled_knowledge_db() -> Path:
    return PACKAGE_DATA_DIR / "milvus_lite.db"


def prepare_knowledge_db() -> Path:
    target = runtime_data_dir() / "milvus_lite.db"
    source = bundled_knowledge_db()
    if not target.exists() and source.exists():
        shutil.copy2(source, target)
    return target


def evidence_index_db() -> Path:
    return runtime_data_dir() / "evidence_memory.db"


def evidence_memory_store() -> Path:
    return runtime_data_dir() / "evidence_memory.json"
