#!/usr/bin/env bash
set -euo pipefail

# Controlled A/B continuation from the same checkpoint.
# A: PLM-only baseline
# B: PLM-only + mild relation balancing for weak WN18RR relations

GPU_DEVICE="${1:-0}"
START_CKPT="${2:-./model_tuned/RelatE_wn18rr_wn18rr_rel_simkgc_legacy_plm_011126/checkpoint}"
START_STEP="${3:-39500}"
MAX_STEPS="${4:-40250}"

if [[ ! -f "$START_CKPT" ]]; then
  echo "Checkpoint not found: $START_CKPT" >&2
  exit 1
fi

if [[ ! -f ".env copy wn18rr" ]]; then
  echo "Missing env file: .env copy wn18rr" >&2
  exit 1
fi

set -a
source ".env copy wn18rr"
set +a

SAVE_A="wn18rr_ab_plm_A_baseline_${MAX_STEPS}"
SAVE_B="wn18rr_ab_plm_B_balanced_${MAX_STEPS}"

run_variant() {
  local save_id="$1"
  shift
  echo "=== Running ${save_id} ==="
  env \
    SAVE_ID="$save_id" \
    DATASET_NAME=wn18rr \
    INIT_CHECKPOINT="$START_CKPT" \
    SKIP_OPTIMIZER_STATE=false \
    MAX_STEPS="$MAX_STEPS" \
    TEACHER_TYPE= \
    TEACHER_CHECKPOINT= \
    PLM_TEACHER=true \
    PLM_ENTITY_EMB_PATH=./data/wn18rr/plm_entity.npy \
    PLM_REL_EMB_PATH=./data/wn18rr/plm_relation.npy \
    "$@" \
    bash run.sh train RelatE wn18rr "$GPU_DEVICE" "$save_id"
}

# A: baseline (no relation-specific reweighting)
run_variant "$SAVE_A" \
  RELATION_SAMPLING_WEIGHTS= \
  RELATION_LOSS_WEIGHTS= \
  EXTRA_HARD_RELATIONS=

# B: balanced boost for weak relations, still global training over all triples
run_variant "$SAVE_B" \
  RELATION_SAMPLING_WEIGHTS="_hypernym:1.8,_has_part:1.6,_member_meronym:1.6,_instance_hypernym:1.4" \
  RELATION_LOSS_WEIGHTS="_hypernym:1.8,_has_part:1.6,_member_meronym:1.6,_instance_hypernym:1.4" \
  EXTRA_HARD_RELATIONS="0 4 6"

LOG_A="model_tuned/RelatE_wn18rr_${SAVE_A}/train.log"
LOG_B="model_tuned/RelatE_wn18rr_${SAVE_B}/train.log"

/mnt/data1/achakr40/TransEE/.venv/bin/python3 - <<PY
import re
from pathlib import Path

start_step = int("${START_STEP}")
max_steps = int("${MAX_STEPS}")
window_lo = start_step + 750
window_hi = max_steps
pat = re.compile(r"Valid MRR at step (\\d+): ([0-9]*\\.?[0-9]+)")

def read_metrics(path: str):
    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if not m:
            continue
        step = int(m.group(1))
        mrr = float(m.group(2))
        out[step] = mrr
    return out

def summarize(name: str, metrics: dict):
    target = metrics.get(window_hi)
    in_window = [(s, v) for s, v in metrics.items() if window_lo <= s <= window_hi]
    best = max(in_window, key=lambda x: x[1]) if in_window else None
    print(f"{name}:")
    print(f"  target_step({window_hi})_mrr = {target}")
    print(f"  best_in_window[{window_lo},{window_hi}] = {best}")

a = read_metrics("${LOG_A}")
b = read_metrics("${LOG_B}")
summarize("A_baseline", a)
summarize("B_balanced", b)

if a.get(window_hi) is not None and b.get(window_hi) is not None:
    print(f"delta_target_step = {b[window_hi] - a[window_hi]:+.6f}")
PY

