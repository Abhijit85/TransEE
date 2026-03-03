#!/usr/bin/env python3
"""
Build dataset-agnostic concept files for concept-guided RelatE training.

Outputs:
  - entity_to_concept.json : entity string -> concept string
  - concept_depth.json     : concept string -> average depth (if available)
  - concept_members.json   : concept string -> [entity strings]
"""

from __future__ import annotations

import argparse
import json
import os
import math
from collections import defaultdict


class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def read_entities(path: str):
    entity2id = {}
    with open(path, "r") as fin:
        for line in fin:
            idx, ent = line.rstrip("\n").split("\t")
            entity2id[ent] = int(idx)
    return entity2id


def read_relations(path: str):
    rel2id = {}
    with open(path, "r") as fin:
        for line in fin:
            idx, rel = line.rstrip("\n").split("\t")
            rel2id[rel] = int(idx)
    return rel2id


def read_train(path: str, entity2id, rel2id):
    triples = []
    with open(path, "r") as fin:
        for line in fin:
            h, r, t = line.rstrip("\n").split("\t")
            if h in entity2id and t in entity2id and r in rel2id:
                triples.append((entity2id[h], rel2id[r], entity2id[t]))
    return triples


def _depth_map_stats(depth_map):
    if not depth_map:
        return 0, 0, 0
    vals = list(depth_map.values())
    nonzero = sum(1 for v in vals if float(v) != 0.0)
    min_v = min(vals)
    max_v = max(vals)
    return nonzero, min_v, max_v


def load_depth_map(primary_path: str | None, data_path: str):
    candidates = []
    if primary_path:
        candidates.append(primary_path)
    # Prefer computed entity depths over legacy hierarchy_depth files.
    candidates.append(os.path.join(data_path, "entity_depths.json"))
    candidates.append(os.path.join(data_path, "hierarchy_depth.json"))

    tried = set()
    for path in candidates:
        if not path or path in tried or not os.path.exists(path):
            continue
        tried.add(path)
        with open(path, "r") as fin:
            raw = json.load(fin)
        depth_map = {k: float(v) for k, v in raw.items()}
        nonzero, min_v, max_v = _depth_map_stats(depth_map)
        # Reject degenerate maps (all values identical or all zero).
        if len(depth_map) > 0 and min_v == max_v:
            print(f"[build_concepts] Skipping degenerate depth map: {path} (constant value {min_v})")
            continue
        if len(depth_map) > 0 and nonzero == 0:
            print(f"[build_concepts] Skipping depth map with all-zero values: {path}")
            continue
        print(f"[build_concepts] Using depth map: {path} (entries={len(depth_map)}, min={min_v}, max={max_v})")
        return depth_map
    return {}


def infer_relation_roles(rel2id):
    hier_keys = ("hypernym", "subclass", "isa", "instance", "type")
    mero_keys = ("meronym", "has_part", "part", "member")
    hier = set()
    mero = set()
    for rel, idx in rel2id.items():
        low = rel.lower()
        if any(key in low for key in hier_keys):
            hier.add(idx)
        if any(key in low for key in mero_keys):
            mero.add(idx)
    return hier, mero


def build_structural_components(nentity, triples, structural_rel_ids):
    dsu = DSU(nentity)
    for h, r, t in triples:
        if r in structural_rel_ids:
            dsu.union(h, t)
    comps = defaultdict(list)
    for i in range(nentity):
        comps[dsu.find(i)].append(i)
    return comps


def build_proxy_depth_map(entity2id, triples, rel_ids_for_depth):
    """
    Build a dataset-agnostic proxy depth when ontology depth is unavailable.
    Heuristic: entities with lower structural degree are treated as more specific
    (deeper), using a smooth inverse-log transform.
    """
    degree = [0] * len(entity2id)
    for h, r, t in triples:
        if rel_ids_for_depth and r not in rel_ids_for_depth:
            continue
        degree[h] += 1
        degree[t] += 1
    max_deg = max(degree) if degree else 1
    id2entity = {idx: ent for ent, idx in entity2id.items()}
    depth_map = {}
    denom = math.log1p(max_deg) if max_deg > 0 else 1.0
    for idx, deg in enumerate(degree):
        # Higher degree -> more generic -> shallower depth.
        # Lower degree -> more specific -> deeper depth.
        depth_val = (denom - math.log1p(deg)) if denom > 0 else 0.0
        depth_map[id2entity[idx]] = float(depth_val)
    return depth_map


def main():
    parser = argparse.ArgumentParser(description="Build concept maps for KGE datasets.")
    parser.add_argument("--data_path", required=True, help="Dataset directory with *.dict and train.txt")
    parser.add_argument("--entity_type_map", default=None, help="Optional JSON entity->type map")
    parser.add_argument("--entity_depth_map", default=None, help="Optional JSON entity->depth map")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--min_component_size", type=int, default=2,
                        help="Use structural component concept when component size >= this threshold")
    args = parser.parse_args()

    entities_path = os.path.join(args.data_path, "entities.dict")
    relations_path = os.path.join(args.data_path, "relations.dict")
    train_path = os.path.join(args.data_path, "train.txt")
    os.makedirs(args.output_dir, exist_ok=True)

    entity2id = read_entities(entities_path)
    id2entity = {v: k for k, v in entity2id.items()}
    rel2id = read_relations(relations_path)
    triples = read_train(train_path, entity2id, rel2id)

    type_map = {}
    if args.entity_type_map and os.path.exists(args.entity_type_map):
        with open(args.entity_type_map, "r") as fin:
            type_map = json.load(fin)

    depth_map = load_depth_map(args.entity_depth_map, args.data_path)

    hier_rel_ids, mero_rel_ids = infer_relation_roles(rel2id)
    structural_rels = hier_rel_ids | mero_rel_ids
    components = build_structural_components(len(entity2id), triples, structural_rels)
    if not depth_map:
        rel_ids_for_depth = structural_rels if structural_rels else set(rel2id.values())
        depth_map = build_proxy_depth_map(entity2id, triples, rel_ids_for_depth)
        print(f"[build_concepts] Using proxy structural depth (relations={len(rel_ids_for_depth)}).")

    entity_to_concept = {}
    concept_members = defaultdict(list)
    component_of_entity = {}
    component_size = {}
    for comp_root, members in components.items():
        component_size[comp_root] = len(members)
        for eid in members:
            component_of_entity[eid] = comp_root

    for ent, eid in entity2id.items():
        concept = None
        ent_type = type_map.get(ent)
        if ent_type:
            concept = f"type::{ent_type}"
        comp_root = component_of_entity.get(eid)
        if comp_root is not None and component_size.get(comp_root, 0) >= args.min_component_size and concept is None:
            concept = f"comp::{comp_root}"
        if concept is None:
            # Fallback to lightweight namespace heuristic for portability.
            if ":" in ent:
                concept = f"ns::{ent.split(':', 1)[0]}"
            else:
                concept = f"self::{ent}"
        entity_to_concept[ent] = concept
        concept_members[concept].append(ent)

    concept_depth = {}
    if depth_map:
        for concept, members in concept_members.items():
            vals = [depth_map[m] for m in members if m in depth_map]
            if vals:
                concept_depth[concept] = float(sum(vals) / len(vals))

    with open(os.path.join(args.output_dir, "entity_to_concept.json"), "w") as fout:
        json.dump(entity_to_concept, fout, indent=2, sort_keys=True)
    with open(os.path.join(args.output_dir, "concept_members.json"), "w") as fout:
        json.dump(concept_members, fout, indent=2, sort_keys=True)
    with open(os.path.join(args.output_dir, "concept_depth.json"), "w") as fout:
        json.dump(concept_depth, fout, indent=2, sort_keys=True)

    print(f"entities={len(entity_to_concept)} concepts={len(concept_members)} depth_concepts={len(concept_depth)}")


if __name__ == "__main__":
    main()
