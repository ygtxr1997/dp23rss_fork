#!/usr/bin/env bash
# Pass explicit source/output paths and matching assets to the conversion tool.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
exec python -m diffusion_policy.common.pi05_convert_checkpoint "$@"
