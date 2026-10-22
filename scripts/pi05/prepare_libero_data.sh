#!/usr/bin/env bash
# Prepare LIBERO replay splits and raw policy statistics before training.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
exec python -m diffusion_policy.common.prepare_data --config-name=pi05_libero_data "$@"
