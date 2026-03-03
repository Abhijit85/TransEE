#!/usr/bin/env bash
set -euo pipefail

# Separate runner for vanilla RelatE fork.
# Uses Code_vanilla_relate and defaults to .env.vanilla_relate
ENV_FILE=${ENV_FILE:-.env.vanilla_relate}
CODE_PATH=${CODE_PATH:-Code_vanilla_relate}

exec env ENV_FILE="$ENV_FILE" CODE_PATH="$CODE_PATH" bash run.sh "$@"
