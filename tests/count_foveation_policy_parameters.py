"""Load a foveation policy config and print its parameter statistics."""

import argparse
import sys
from pathlib import Path
from typing import Union


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hydra
from omegaconf import OmegaConf


DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "pusht256_foveation_dp.yaml"


def print_parameter_statistics_from_config(
    config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
):
    """Instantiate the configured policy and print its real parameter counts."""
    config_path = Path(config_path).expanduser().resolve()
    cfg = OmegaConf.load(config_path)
    policy = hydra.utils.instantiate(cfg.policy)
    return policy.print_parameter_statistics()


def main() -> None:
    """Run parameter counting for the selected foveation policy config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Foveation training config to load.",
    )
    args = parser.parse_args()
    print_parameter_statistics_from_config(args.config)


if __name__ == "__main__":
    main()
