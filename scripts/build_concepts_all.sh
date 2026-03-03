#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/mnt/data1/achakr40/TransEE/.venv/bin/python3}"

run_build() {
  local data_path="$1"
  local type_map="$2"
  local out_dir="$3"
  local extra=()
  if [[ -f "$type_map" ]]; then
    extra+=(--entity_type_map "$type_map")
  fi
  echo "[build_concepts_all] Building concepts for ${data_path} -> ${out_dir}"
  "$PYTHON_BIN" scripts/build_concepts.py \
    --data_path "$data_path" \
    "${extra[@]}" \
    --output_dir "$out_dir" \
    --min_component_size 3
}

run_build "data/wn18rr" "wn18rr_entity_type_map.json" "data/wn18rr/concepts"
run_build "data/FB15k-237" "fb15k237_entity_type_map.json" "data/FB15k-237/concepts"
run_build "data/YAGO3-10" "yago3_entity_type_map.json" "data/YAGO3-10/concepts"

cat <<'EOF'

[build_concepts_all] Suggested concept-guided env settings:

# WN18RR
ENTITY_CONCEPT_MAP_PATH=./data/wn18rr/concepts/entity_to_concept.json
CONCEPT_DEPTH_MAP_PATH=./data/wn18rr/concepts/concept_depth.json
CONCEPT_PHASE_WEIGHT=0.02
CONCEPT_MODULUS_WEIGHT=0.03
CONCEPT_RELATION_WEIGHT=0.02
CONCEPT_DEPTH_MARGIN=0.05

# FB15k-237
ENTITY_CONCEPT_MAP_PATH=./data/FB15k-237/concepts/entity_to_concept.json
CONCEPT_DEPTH_MAP_PATH=./data/FB15k-237/concepts/concept_depth.json
CONCEPT_PHASE_WEIGHT=0.01
CONCEPT_MODULUS_WEIGHT=0.015
CONCEPT_RELATION_WEIGHT=0.01
CONCEPT_DEPTH_MARGIN=0.03

# YAGO3-10
ENTITY_CONCEPT_MAP_PATH=./data/YAGO3-10/concepts/entity_to_concept.json
CONCEPT_DEPTH_MAP_PATH=./data/YAGO3-10/concepts/concept_depth.json
CONCEPT_PHASE_WEIGHT=0.008
CONCEPT_MODULUS_WEIGHT=0.012
CONCEPT_RELATION_WEIGHT=0.008
CONCEPT_DEPTH_MARGIN=0.02

EOF
