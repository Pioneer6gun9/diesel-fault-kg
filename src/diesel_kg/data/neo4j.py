#!/usr/bin/env python3
"""Parse a local Neo4j Browser `p` CSV without contacting a database.

The export contains human-readable path strings, not Neo4j internal IDs.
Node IDs below therefore identify exact (entity type, name) pairs. They are
semantic projection IDs, not a claim that the original database is recovered
losslessly. Every observation keeps its CSV record and original properties.
No relation is converted into a causal gold label and no edge weight is added.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re

PATH_RE = re.compile(r"^\((?P<source>:[\s\S]+?)\)-\[(?P<edge>:[\s\S]+?)\]->\((?P<target>:[\s\S]+)\)$")
COMPONENT_RE = re.compile(r"^:(?P<labels>[^\s{]+)\s*(?:\{(?P<properties>[\s\S]*)\})?$")
PROPERTY_RE = re.compile(r"(?:^|,\s*)(?P<key>[^\s,:{}]+):\s*")
NON_DIESEL_NAMESPACE = "智能系统故障图谱"


def stable_id(prefix: str, identity: object) -> str:
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return prefix + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def parse_properties(raw: str | None) -> dict[str, str]:
    """Handle the export's unquoted property values without eval/exec."""
    if not raw:
        return {}
    fields = list(PROPERTY_RE.finditer(raw))
    if not fields or raw[:fields[0].start()].strip():
        raise ValueError("Unrecognized property syntax")
    output = {}
    for index, field in enumerate(fields):
        end = fields[index + 1].start() if index + 1 < len(fields) else len(raw)
        key, value = field.group("key"), raw[field.end():end].strip()
        if key in output:
            raise ValueError("Duplicate property key")
        if value.startswith('"') and value.endswith('"'):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        output[key] = str(value)
    return output


def parse_component(raw: str) -> tuple[list[str], dict[str, str]]:
    match = COMPONENT_RE.fullmatch(raw.strip())
    if not match:
        raise ValueError("Unrecognized node/relationship syntax")
    return match.group("labels").split(":"), parse_properties(match.group("properties"))


def parse_export(input_path: Path) -> dict:
    nodes: dict[str, dict] = {}
    edges: dict[str, dict] = {}
    paths = []
    unparsed = []
    node_property_keys = collections.Counter()
    edge_property_keys = collections.Counter()
    relationship_observations = collections.Counter()
    namespace_observations = collections.Counter()
    eligible_records = 0

    def add_node(labels: list[str], properties: dict[str, str], source: dict) -> str:
        entity_type = properties.get("type") or properties.get("entity_type")
        if not entity_type:
            entity_type = next((label for label in labels if label != "Entity"), "Entity")
        name = properties.get("name")
        if name is None:
            raise ValueError("Node has no name property")
        node_id = stable_id("n_", [entity_type, name])
        eligible = NON_DIESEL_NAMESPACE not in labels
        node = nodes.setdefault(node_id, {
            "id": node_id, "name": name, "entity_type": entity_type,
            "labels": [], "sources": [], "analysis_eligible": eligible,
            "identity_policy": "exact_entity_type_and_name_no_alias_merging",
        })
        node["labels"] = sorted(set(node["labels"]) | set(labels))
        node["analysis_eligible"] = node["analysis_eligible"] and eligible
        node["sources"].append({**source, "properties": properties, "labels": labels})
        node_property_keys.update(properties.keys())
        namespace_observations.update(labels)
        return node_id

    with input_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames or "p" not in fieldnames:
            raise ValueError("The CSV must have a p column")
        row_count = 0
        for record_number, row in enumerate(reader, 1):
            row_count += 1
            raw = row.get("p") or ""
            source = {"csv_record_number": record_number, "csv_line_end": reader.line_num}
            match = PATH_RE.fullmatch(raw.strip())
            if not match:
                unparsed.append({**source, "reason": "unrecognized_path_shape", "p": raw})
                continue
            try:
                slabels, sprops = parse_component(match.group("source"))
                tlabels, tprops = parse_component(match.group("target"))
                relations, rprops = parse_component(match.group("edge"))
                if len(relations) != 1:
                    raise ValueError("Expected exactly one relationship type")
                if "name" not in sprops or "name" not in tprops:
                    raise ValueError("Node has no name property")
            except ValueError as error:
                unparsed.append({**source, "reason": str(error), "p": raw})
                continue
            sid = add_node(slabels, sprops, source)
            tid = add_node(tlabels, tprops, source)
            relation = relations[0]
            edge_id = stable_id("e_", [sid, relation, tid])
            eligible = NON_DIESEL_NAMESPACE not in slabels and NON_DIESEL_NAMESPACE not in tlabels
            edge = edges.setdefault(edge_id, {
                "id": edge_id, "source": sid, "target": tid,
                "relation_type": relation, "sources": [],
                "analysis_eligible": eligible, "weight": None,
                "causal_gold_label": False, "orientation": "as_exported",
            })
            edge["analysis_eligible"] = edge["analysis_eligible"] and eligible
            edge["sources"].append({**source, "properties": rprops})
            paths.append({**source, "p": raw, "source": sid, "target": tid,
                          "edge": edge_id, "analysis_eligible": eligible})
            eligible_records += int(eligible)
            edge_property_keys.update(rprops.keys())
            relationship_observations.update([relation])

    export_ids = collections.defaultdict(set)
    for node in nodes.values():
        for observation in node["sources"]:
            if "id" in observation["properties"]:
                export_ids[observation["properties"]["id"]].add(node["id"])
    reused_export_ids = {key: len(value) for key, value in export_ids.items() if len(value) > 1}
    schema = {
        "source_csv": str(input_path.resolve()),
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "csv_columns": fieldnames, "row_count": row_count,
        "parsed_records": len(paths), "unparsed_records": len(unparsed),
        "unique_semantic_nodes": len(nodes), "unique_directed_typed_edges": len(edges),
        "eligible_diesel_records": eligible_records,
        "excluded_namespace_records": len(paths) - eligible_records,
        "eligible_diesel_nodes": sum(n["analysis_eligible"] for n in nodes.values()),
        "eligible_diesel_edges": sum(e["analysis_eligible"] for e in edges.values()),
        "self_loops": sum(e["source"] == e["target"] for e in edges.values()),
        "repeated_edge_observations": len(paths) - len(edges),
        "node_property_keys": dict(node_property_keys),
        "edge_property_keys": dict(edge_property_keys),
        "relationship_observation_counts": dict(relationship_observations),
        "node_label_observation_counts": dict(namespace_observations),
        "reused_original_id_values_across_semantic_nodes": reused_export_ids,
        "weight_policy": "No numeric propagation weights are present or inferred. All weight fields are null.",
        "eligibility_policy": "Retain all rows; flag rows with the exact 智能系统故障图谱 label as off-domain. This does not certify all other rows as correct.",
        "identity_limitation": "The p strings lack Neo4j internal node/edge IDs. Exact entity-type/name merging produces a semantic projection, not a lossless database restore.",
        "causality_limitation": "Preserve original relationship names and directions. Metadata, symptoms, repair and application links are not causal gold labels.",
        "role": "original_graph_semantic_projection_for_descriptive_analysis",
    }
    return {"schema": schema, "nodes": list(nodes.values()), "edges": list(edges.values()),
            "source_paths": paths, "unparsed": unparsed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON export path; default prints schema only.")
    parser.add_argument("--flat-output-dir", type=Path, help="Optional directory for diesel-eligible entities.jsonl and triples.jsonl; case_id stays empty.")
    args = parser.parse_args()
    result = parse_export(args.csv_path)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.flat_output_dir:
        args.flat_output_dir.mkdir(parents=True, exist_ok=True)
        lookup = {node["id"]: node for node in result["nodes"]}
        entities = [{"case_id": "", "entity": node["name"], "entity_type": node["entity_type"]}
                    for node in result["nodes"] if node["analysis_eligible"]]
        triples = []
        for edge in result["edges"]:
            head, tail = lookup[edge["source"]], lookup[edge["target"]]
            if not (edge["analysis_eligible"] and head["analysis_eligible"] and tail["analysis_eligible"]):
                continue
            triples.append({"case_id": "", "head": head["name"], "head_type": head["entity_type"],
                            "relation": edge["relation_type"], "tail": tail["name"],
                            "tail_type": tail["entity_type"], "status": "historical_export_unverified"})
        for filename, records in (("entities.jsonl", entities), ("triples.jsonl", triples)):
            (args.flat_output_dir / filename).write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
        result["schema"]["flat_exports"] = {"entities": len(entities), "triples": len(triples),
            "case_binding": "missing_case_id_empty_string", "directory": str(args.flat_output_dir.resolve()),
            "limitation": "Any substring matching against cases is a new fallback association, not recovered historical provenance."}
        if args.output:
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["schema"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
