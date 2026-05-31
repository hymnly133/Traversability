from __future__ import annotations

from pathlib import Path


def resolve_frontend_index_path() -> Path:
    module_path = Path(__file__).resolve()
    candidates = [
        module_path.parents[2] / "frontend" / "paper-workbench" / "index.html",
        module_path.with_name("frontend") / "paper-workbench" / "index.html",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


FRONTEND_INDEX_PATH = resolve_frontend_index_path()


def load_index_html() -> str:
    return FRONTEND_INDEX_PATH.read_text(encoding="utf-8")


INDEX_HTML = load_index_html()
