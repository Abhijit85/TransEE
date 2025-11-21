#!/usr/bin/env python3
"""
Utility to convert the heterogeneous ogbl-biokg dataset downloaded via OGB
into the flat triple/text format expected by Code/driver.py.

The script generates:
  - entities.dict / relations.dict
  - train.txt / valid.txt / test.txt
  - entity_type_map.json (optional input for --type_map_path)
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from ogb.linkproppred import LinkPropPredDataset


def patch_torch_load():
    """
    OGB caches graphs with torch.save(..., pickle_protocol=4).
    PyTorch 2.6+ defaults to weights_only=True which refuses to load
    arbitrary pickles, so temporarily force weights_only=False.
    """
    original_load = torch.load

    def patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = patched
    return original_load


def restore_torch_load(original_load):
    if original_load is not None:
        torch.load = original_load


def sanitize_token(token: str) -> str:
    return token.strip().replace("\t", " ").replace("\n", " ")


def load_entity_names(mapping_dir: Path) -> Dict[str, Dict[int, str]]:
    entity_names: Dict[str, Dict[int, str]] = {}
    for csv_path in sorted(mapping_dir.glob("*_entidx2name.csv.gz")):
        etype = csv_path.name.replace("_entidx2name.csv.gz", "")
        names: Dict[int, str] = {}
        with gzip.open(csv_path, "rt") as fin:
            reader = csv.DictReader(fin)
            idx_key = reader.fieldnames[0]
            name_key = reader.fieldnames[1]
            for row in reader:
                idx = int(row[idx_key])
                names[idx] = sanitize_token(row[name_key])
        entity_names[etype] = names
    return entity_names


def write_entities(
    out_dir: Path,
    num_nodes_dict: Dict[str, int],
    entity_names: Dict[str, Dict[int, str]],
) -> Tuple[Dict[Tuple[str, int], str], Dict[str, str]]:
    entity_map: Dict[Tuple[str, int], str] = {}
    type_map: Dict[str, str] = {}
    lines: List[str] = []
    global_idx = 0

    for etype in sorted(num_nodes_dict.keys()):
        total = int(num_nodes_dict[etype])
        names_for_type = entity_names.get(etype, {})
        for local_idx in range(total):
            label = names_for_type.get(local_idx, str(local_idx))
            token = sanitize_token(f"{etype}:{label}")
            entity_map[(etype, local_idx)] = token
            type_map[token] = etype
            lines.append(f"{global_idx}\t{token}\n")
            global_idx += 1

    (out_dir / "entities.dict").write_text("".join(lines))
    with (out_dir / "entity_type_map.json").open("w") as fout:
        json.dump(type_map, fout)
    return entity_map, type_map


def load_relation_tuples(raw_dir: Path) -> List[Tuple[str, str, str]]:
    triplet_file = raw_dir / "triplet-type-list.csv.gz"
    relations: List[Tuple[str, str, str]] = []
    with gzip.open(triplet_file, "rt") as fin:
        reader = csv.reader(fin)
        for head_type, rel_name, tail_type in reader:
            relations.append(
                (sanitize_token(head_type), sanitize_token(rel_name), sanitize_token(tail_type))
            )
    return relations


def write_relations(out_dir: Path, relations: List[Tuple[str, str, str]]) -> None:
    with (out_dir / "relations.dict").open("w") as fout:
        for idx, (_, rel_name, _) in enumerate(relations):
            fout.write(f"{idx}\t{rel_name}\n")


def write_split(
    out_dir: Path,
    split_name: str,
    split_dict,
    entity_map: Dict[Tuple[str, int], str],
    relations: List[Tuple[str, str, str]],
) -> None:
    head = split_dict["head"]
    tail = split_dict["tail"]
    relation = split_dict["relation"]
    head_types = split_dict["head_type"]
    tail_types = split_dict["tail_type"]

    total = head.shape[0]
    out_path = out_dir / f"{split_name}.txt"
    with out_path.open("w") as fout:
        for idx in range(total):
            ht = head_types[idx]
            tt = tail_types[idx]
            rel_id = int(relation[idx])
            rel_tuple = relations[rel_id]
            if rel_tuple[0] != ht or rel_tuple[2] != tt:
                raise ValueError(
                    f"Split relation mismatch at row {idx}: "
                    f"{ht}->{tt} vs {rel_tuple}"
                )

            head_name = entity_map[(ht, int(head[idx]))]
            tail_name = entity_map[(tt, int(tail[idx]))]
            rel_name = rel_tuple[1]
            fout.write(f"{head_name}\t{rel_name}\t{tail_name}\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare ogbl-biokg triples for RelatE.")
    parser.add_argument(
        "--ogb_root",
        default="data/ogb",
        help="Base directory where LinkPropPredDataset stored ogbl_biokg",
    )
    parser.add_argument(
        "--output",
        default="data/ogbl_biokg_kge",
        help="Directory for the exported triples",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    restore_handle = patch_torch_load()
    try:
        dataset = LinkPropPredDataset(name="ogbl-biokg", root=args.ogb_root)
        graph = dataset[0]
        split = dataset.get_edge_split()
    finally:
        restore_torch_load(restore_handle)

    mapping_dir = Path(args.ogb_root) / "ogbl_biokg" / "mapping"
    entity_names = load_entity_names(mapping_dir)

    entity_map, _ = write_entities(out_dir, graph["num_nodes_dict"], entity_names)
    relations = load_relation_tuples(Path(args.ogb_root) / "ogbl_biokg" / "raw")
    write_relations(out_dir, relations)

    for split_name in ("train", "valid", "test"):
        write_split(out_dir, split_name, split[split_name], entity_map, relations)

    print(f"Finished writing ogbl-biokg triples to {out_dir}." )


if __name__ == "__main__":
    main()
