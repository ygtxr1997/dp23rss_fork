"""Shared validation for the explicit TCL task-to-slot mapping."""

from pathlib import Path

from omegaconf import OmegaConf


def load_tcl_task_mapping(path: str) -> dict[str, int]:
    """Load the explicit TCL task directory to ten-slot ID mapping."""
    config = OmegaConf.to_container(OmegaConf.load(Path(path).expanduser()), resolve=True)
    entries = config.get("tasks", [])
    if len(entries) > 10:
        raise ValueError("TCL task mapping may contain at most ten tasks")
    mapping, used_ids = {}, set()
    for entry in entries:
        task_id = int(entry["task_id"])
        task_dir = str(entry["task_dir"])
        if not 0 <= task_id < 10 or task_id in used_ids or task_dir in mapping:
            raise ValueError(f"Invalid or duplicate TCL task mapping entry: {entry}")
        mapping[task_dir] = task_id
        used_ids.add(task_id)
    return mapping
