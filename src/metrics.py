"""Metrics persistence helper for genai_ad_studio."""

import json
from pathlib import Path
from typing import Any, Dict


METRICS_PATH = Path("results/metrics.json")


def _to_serializable(val: Any) -> Any:
    """Recursively convert NumPy and non-standard types to Python native types."""
    if hasattr(val, "item") and callable(val.item):
        return val.item()
    if isinstance(val, dict):
        return {k: _to_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_to_serializable(v) for v in val]
    return val


def load_metrics(path: Path = METRICS_PATH) -> Dict[str, Any]:
    """Load existing metrics dictionary from disk.

    Args:
        path (Path): Path to metrics JSON file.

    Returns:
        dict: Loaded metrics, or empty dict if file does not exist.
    """
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_metrics(key: str, value: Any, path: Path = METRICS_PATH) -> None:
    """Save or update a key in metrics.json without modifying other keys.

    Args:
        key (str): Metric identifier key.
        value (Any): Value or sub-dictionary to store.
        path (Path): Path to metrics JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(path)
    metrics[key] = _to_serializable(value)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    current = load_metrics()
    print(f"Current metrics in {METRICS_PATH}:")
    print(json.dumps(current, indent=2))
