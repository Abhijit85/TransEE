#!/usr/bin/env bash
set -euo pipefail

# Runner for RelateV single-model copy.
ENV_FILE=${ENV_FILE:-.env.relate_v}
CODE_PATH=${CODE_PATH:-Code_relate_v}

exec env ENV_FILE="$ENV_FILE" CODE_PATH="$CODE_PATH" bash run.sh "$@"
