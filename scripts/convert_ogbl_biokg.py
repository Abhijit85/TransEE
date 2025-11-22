#!/usr/bin/env python3
"""Convert ogbl-biokg into the plain-text KGE format used by RelatE."""

import argparse
import csv
import gzip
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def read_num_nodes(path: Path) -> Tuple[List[str], List[int]]:
    """Return node types and counts from num-node-dict.csv.gz."""
    with gzip.open(path, "rt") as fin:
        reader = csv.reader(fin)
        header = next(reader)
        counts = next(reader)
    return [h.strip() for h in header], [int(c.strip()) for c in counts]


def load_entity_names(mapping_dir: Path, node_type: str, expected: int) -> List[str]:
    """Load per-type entity names, falling back to synthetic labels as needed."""
    mapping_file = mapping_dir / f"{node_type}_entidx2name.csv.gz"
    names: List[str] = []
    if mapping_file.exists():
        with gzip.open(mapping_file, "rt") as fin:
            reader = csv.reader(fin)
            next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                names.append(row[1].strip())
    if len(names) < expected:
        names.extend(f"{node_type}_{i}" for i in range(len(names), expected))
    return names[:expected]


def build_entity_lookup(
    mapping_dir: Path, num_node_file: Path
) -> Tuple[List[str], Dict[Tuple[str, int], int]]:
    """Build the global entity list plus a (type, local_id) -> global_id lookup."""
    node_types, node_counts = read_num_nodes(num_node_file)
    entity_names: List[str] = []
    lookup: Dict[Tuple[str, int], int] = {}
    for node_type, count in zip(node_types, node_counts):
        names = load_entity_names(mapping_dir, node_type, count)
        offset = len(entity_names)
        for local_idx in range(count):
            lookup[(node_type, local_idx)] = offset + local_idx
            entity_names.append(f"{node_type}:{names[local_idx]}")
    return entity_names, lookup


def load_relation_names(mapping_file: Path) -> List[str]:
    with gzip.open(mapping_file, "rt") as fin:
        reader = csv.reader(fin)
        next(reader, None)
        return [row[1].strip() for row in reader if len(row) >= 2]


def write_dict_file(path: Path, values: List[str]) -> None:
    with path.open("w") as fout:
        for idx, name in enumerate(values):
            fout.write(f"{idx}\t{name}\n")


def convert_split(
    split_path: Path,
    out_path: Path,
    entity_names: List[str],
    lookup: Dict[Tuple[str, int], int],
    rel_names: List[str],
) -> None:
    data = torch.load(split_path, map_location="cpu", weights_only=False)
    heads = data["head"].tolist()
    tails = data["tail"].tolist()
    head_types = data["head_type"]
    tail_types = data["tail_type"]
    relations = data["relation"].tolist()

    with out_path.open("w") as fout:
        for h_type, h_idx, rel_id, t_type, t_idx in zip(
            head_types, heads, relations, tail_types, tails
        ):
            head_name = entity_names[lookup[(h_type, h_idx)]]
            tail_name = entity_names[lookup[(t_type, t_idx)]]
            rel_name = rel_names[rel_id]
            fout.write(f"{head_name}\t{rel_name}\t{tail_name}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ogbl-biokg into KGE text files"
    )
    parser.add_argument(
        "--ogb-root", default="data/ogb", help="Directory containing ogbl_biokg"
    )
    parser.add_argument(
        "--out-path",
        default="data/ogb/ogbl_biokg_kge",
        help="Destination directory for KGE files",
    )
    parser.add_argument(
        "--split", default="random", help="Which ogb split folder to read"
    )
    args = parser.parse_args()

    ogb_dir = Path(args.ogb_root) / "ogbl_biokg"
    if not ogb_dir.exists():
        raise FileNotFoundError(
            f"Cannot find ogbl_biokg under {args.ogb_root}. Did you download it?"
        )

    mapping_dir = ogb_dir / "mapping"
    raw_dir = ogb_dir / "raw"
    split_dir = ogb_dir / "split" / args.split

    entity_names, lookup = build_entity_lookup(
        mapping_dir, raw_dir / "num-node-dict.csv.gz"
    )
    rel_names = load_relation_names(mapping_dir / "relidx2relname.csv.gz")

    out_dir = Path(args.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_dict_file(out_dir / "entities.dict", entity_names)
    write_dict_file(out_dir / "relations.dict", rel_names)

    for part in ("train", "valid", "test"):
        split_file = split_dir / f"{part}.pt"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")
        convert_split(split_file, out_dir / f"{part}.txt", entity_names, lookup, rel_names)

    print(f"Converted ogbl-biokg splits into {out_dir}")


if __name__ == "__main__":
    main()
