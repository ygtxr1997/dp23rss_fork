import math
from pathlib import Path
import re
from string import Formatter

from omegaconf import OmegaConf


def select_pretrained_checkpoint(path, mode=None) -> Path:
    """Resolve an explicit checkpoint or select best/latest from its source run."""
    path = Path(path).expanduser()
    if mode is None:
        if not path.is_file():
            raise FileNotFoundError(f"Expected a checkpoint file: {path}")
        print(f"[Pretrained checkpoint] mode=explicit, path={path}")
        return path
    if mode not in ("best", "latest"):
        raise ValueError(f"pretrained_ckpt_mode must be null, best or latest, got {mode!r}")

    # Also accept the legacy .../checkpoints/*.ckpt input, even if latest is absent.
    run_dir = path.parent.parent if path.suffix == ".ckpt" else path
    checkpoint_dir = run_dir / "checkpoints"
    if mode == "latest":
        checkpoint_path = checkpoint_dir / "latest.ckpt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        print(f"[Pretrained checkpoint] mode=latest, path={checkpoint_path}")
        return checkpoint_path

    source_path = run_dir / ".hydra" / "config.yaml"
    source_cfg = OmegaConf.load(source_path)
    topk = OmegaConf.select(source_cfg, "checkpoint.topk")
    if topk is None or int(topk.get("k", 1)) <= 0:
        raise ValueError(f"No enabled checkpoint.topk in {source_path}")
    monitor = topk.get("monitor_key")
    direction = topk.get("mode", "min")
    format_str = topk.get("format_str", "epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt")
    if not monitor or direction not in ("min", "max"):
        raise ValueError(f"Invalid checkpoint.topk monitor_key/mode in {source_path}")

    # Match the saved template; metric names and numeric precision vary between runs.
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    parts, fields = [], {}
    field_count = 0
    for literal, field, _, _ in Formatter().parse(format_str):
        parts.append(re.escape(literal))
        if field is None:
            continue
        field_count += 1
        fields[field] = field_count
        value_pattern = r".+?"
        if field == monitor:
            value_pattern = number
        elif field == "epoch":
            value_pattern = r"\d+"
        parts.append(rf"(\s*{value_pattern}\s*)")
    if monitor not in fields:
        raise ValueError(f"checkpoint.topk.format_str must contain {monitor!r}: {source_path}")
    pattern = re.compile("".join(parts))

    candidates = []
    for checkpoint_path in checkpoint_dir.glob("*.ckpt"):
        if checkpoint_path.name == "latest.ckpt" or not checkpoint_path.is_file():
            continue
        match = pattern.fullmatch(checkpoint_path.name)
        if match is None:
            continue
        score = float(match[fields[monitor]])
        if not math.isfinite(score):
            continue
        epoch = int(match[fields["epoch"]]) if "epoch" in fields else -1
        # Filename scores are rounded; ties prefer later epochs, then lexical order.
        rank = -score if direction == "max" else score
        candidates.append((rank, -epoch, checkpoint_path.name, checkpoint_path, score))
    if not candidates:
        raise FileNotFoundError(
            f"No valid top-k checkpoints in {checkpoint_dir} matching {format_str!r}"
        )
    _, _, _, checkpoint_path, score = min(candidates)
    print(
        f"[Pretrained checkpoint] mode=best, monitor={monitor}, "
        f"direction={direction}, filename_score={score}, path={checkpoint_path}"
    )
    return checkpoint_path
