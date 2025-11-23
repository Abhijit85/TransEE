#!/usr/bin/env python3
"""Convert ogbl-ddi into the plain-text KGE format used by RelatE."""

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import List

import torch


def load_entity_names(mapping_file: Path, num_nodes: int) -> List[str]:
    names: List[str] = []
    if mapping_file.exists():
        with gzip.open(mapping_file, "rt") as fin:
            reader = csv.reader(fin)
            next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                names.append(f"drug:{row[1].strip()}")
    if len(names) < num_nodes:
        start = len(names)
        names.extend(f"drug:ID{idx}" for idx in range(start, num_nodes))
    return names[:num_nodes]


def load_split_edges(split_file: Path):
    data = torch.load(split_file, map_location="cpu", weights_only=False)
    edges = data["edge"]
    if isinstance(edges, torch.Tensor):
        edges = edges.cpu().numpy()
    return edges.tolist()


def write_dict(path: Path, values: List[str]) -> None:
    with path.open("w") as fout:
        for idx, name in enumerate(values):
            fout.write(f"{idx}\t{name}\n")


def write_triples(path: Path, edges, entity_names: List[str], relation: str) -> None:
    with path.open("w") as fout:
        for head, tail in edges:
            fout.write(f"{entity_names[head]}\t{relation}\t{entity_names[tail]}\n")


def write_type_map(path: Path, entity_names: List[str]) -> None:
    mapping = {name: "drug" for name in entity_names}
    with path.open("w") as fout:
        json.dump(mapping, fout)


def write_inverse_map(path: Path, relation: str) -> None:
    # ogbl-ddi edges are undirected; map relation to itself for inverse handling.
    with path.open("w") as fout:
        json.dump({relation: relation}, fout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ogbl-ddi into KGE text files")
    parser.add_argument("--ogb-root", default="data/ogb", help="Directory containing ogbl_ddi")
    parser.add_argument("--out-path", default="data/ogb/ogbl_ddi_kge", help="Destination directory")
    parser.add_argument("--split", default="target", help="Split folder name (default: target)")
    args = parser.parse_args()

    ogb_dir = Path(args.ogb_root) / "ogbl_ddi"
    mapping_dir = ogb_dir / "mapping"
    raw_dir = ogb_dir / "raw"
    split_dir = ogb_dir / "split" / args.split

    num_nodes = int(gzip.open(raw_dir / "num-node-list.csv.gz", "rt").read().strip())
    entity_names = load_entity_names(mapping_dir / "nodeidx2drugid.csv.gz", num_nodes)
    relation_name = "ddi-interaction"

    out_dir = Path(args.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_dict(out_dir / "entities.dict", entity_names)
    write_dict(out_dir / "relations.dict", [relation_name])
    write_type_map(out_dir / "entity_type_map.json", entity_names)
    write_inverse_map(out_dir / "relation_inverse_map.json", relation_name)

    for split_name in ("train", "valid", "test"):
        edges = load_split_edges(split_dir / f"{split_name}.pt")
        write_triples(out_dir / f"{split_name}.txt", edges, entity_names, relation_name)

    print(f"Converted ogbl-ddi splits into {out_dir}")


if __name__ == "__main__":
    main()
