"""Helpers for versioned Direct model storage."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from config import Config


def safe_period_label(test_period: Optional[str]) -> str:
    """Return a filesystem-safe label for a month or multi-month test period."""
    if not test_period:
        return "auto"
    return str(test_period).replace(",", "__").replace("/", "-").replace("\\", "-")


def model_run_dir(config: Config, model_type: str, test_period: str) -> Path:
    return config.get_model_path("direct") / model_type / safe_period_label(test_period)


def resolve_model_dir(
    config: Config,
    model_type: str,
    test_months: Optional[Sequence[str]] = None,
) -> Path:
    """Resolve a saved model directory.

    If test_months are supplied, prefer the corresponding versioned directory.
    Without test_months, prefer the latest versioned run. Legacy flat directories
    are accepted as a fallback so older trained artifacts remain usable.
    """
    base_dir = config.get_model_path("direct") / model_type
    legacy_model = base_dir / "model_H00.pkl"

    if test_months:
        period = safe_period_label(",".join(str(month) for month in test_months))
        return base_dir / period

    versioned_runs = [
        path
        for path in base_dir.iterdir()
        if path.is_dir() and (path / "metadata_H00.json").exists()
    ] if base_dir.exists() else []
    if versioned_runs:
        return max(versioned_runs, key=lambda path: path.stat().st_mtime)

    return base_dir
