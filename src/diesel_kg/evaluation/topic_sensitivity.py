#!/usr/bin/env python3
"""Independently compare fixed-topology K7/K10 graph analysis and old K7 baseline."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from diesel_kg.graph.analysis import dump, rows, sha


def numeric_differences(a, b, prefix=""):
    differences = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in a:
            if key in {"elapsed_seconds", "topic_sources"}: continue
            if key not in b:
                differences.append(prefix + "/" + key + ": missing")
            else: differences.extend(numeric_differences(a[key], b[key], prefix + "/" + key))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): differences.append(prefix + ": list lengths differ")
        else:
            for i, (x, y) in enumerate(zip(a, b)): differences.extend(numeric_differences(x, y, prefix + f"/{i}"))
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        if not isinstance(b, (int, float)) or not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
            differences.append(f"{prefix}: {a} != {b}")
    return differences


def graph_signature(data):
    nodes = {n["id"]: {k: n.get(k) for k in ["name", "entity_type", "layer", "case_support", "member_entity_ids", "dictionary_terms"]}
             for n in data["nodes"]}
    edges = {e["id"]: {k: e.get(k) for k in ["source", "target", "relation_type", "kind", "layer", "basis_base_edge_ids", "case_support", "cooccurrence", "physical"]}
             for e in data["analysis_edges"]}
    return {"nodes": nodes, "edges": edges}


def path_signature(records):
    return {tuple(p["node_ids"]): {k: p[k] for k in ["case_ids", "literal_case_frequency", "unique_body_support_frequency"]} for p in records}


def run(k7, k10, reference, output):
    k7, k10 = Path(k7), Path(k10)
    a = json.loads((k7 / "analysis_summary.json").read_text())
    b = json.loads((k10 / "analysis_summary.json").read_text())
    ga = json.loads((k7 / "enhanced_graph.json").read_text())
    gb = json.loads((k10 / "enhanced_graph.json").read_text())
    checks = {"same_node_edge_topology_case_support_bridge_provenance_cooccurrence_physical": graph_signature(ga) == graph_signature(gb)}
    rank_analysis = {}
    for layer in ["base", "enhanced"]:
        for suffix in ["case_grounded_paths", "propagation_candidate_paths"]:
            name = f"{layer}_{suffix}"
            pa, pb = rows(k7 / f"{name}.jsonl"), rows(k10 / f"{name}.jsonl")
            checks[name + "_same_sequences_support_frequencies"] = path_signature(pa) == path_signature(pb)
            if suffix == "propagation_candidate_paths":
                weights_a = {tuple(p["node_ids"]): p["average_edge_weight"] for p in pa}
                weights_b = {tuple(p["node_ids"]): p["average_edge_weight"] for p in pb}
                keys = sorted(weights_a.keys() & weights_b.keys())
                rho = float(spearmanr([weights_a[k] for k in keys], [weights_b[k] for k in keys]).statistic) if len(keys) > 1 else None
                top_a = [p["node_ids"] for p in pa if p["n_nodes"] >= 3][:10]
                top_b = [p["node_ids"] for p in pb if p["n_nodes"] >= 3][:10]
                rank_analysis[layer] = {"n_shared_candidate_paths": len(keys), "average_weight_rank_spearman": rho,
                                        "top10_at_least_three_node_path_overlap": len({tuple(p) for p in top_a} & {tuple(p) for p in top_b}),
                                        "K7_top10_node_sequences": top_a, "K10_top10_node_sequences": top_b}
    differences = None
    if reference:
        old = json.loads((Path(reference) / "analysis_summary.json").read_text())
        differences = numeric_differences(old, a)
        checks["default_K7_all_previous_summary_numeric_values_unchanged"] = not differences
    edge_a = {e["id"]: e for e in ga["analysis_edges"]}
    edge_b = {e["id"]: e for e in gb["analysis_edges"]}
    deltas = [edge_b[n]["weight"] - edge_a[n]["weight"] for n in edge_a.keys() & edge_b.keys()]
    summary = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
               "K7_default_numeric_regression_differences": differences,
               "K7_topic_dimension": a["topic_sources"]["topic_dimension"], "K10_topic_dimension": b["topic_sources"]["topic_dimension"],
               "engineering_system_count": len(a["seven_system_coupling"]["system_order"]),
               "semantic_weight_changes": {"n_typed_analysis_edges": len(deltas), "n_changed_weight": sum(abs(d) > 1e-12 for d in deltas),
                                            "mean_delta_K10_minus_K7": float(np.mean(deltas)), "max_absolute_delta": float(np.max(np.abs(deltas)))},
               "automatic_communities": {"K7": a["automatic_communities"], "K10": b["automatic_communities"]},
               "candidate_ranking_changes": rank_analysis,
               "interpretation": "Topic dimension changes semantic weights, system probabilities and communities. With identical topology, case supports and eligibility, total/context/strict path counts cannot improve merely from K10.",
               "input_sha256": {"K7_graph": sha(k7 / "enhanced_graph.json"), "K10_graph": sha(k10 / "enhanced_graph.json"), "script": sha(__file__)}}
    dump(output, summary)
    if not all(checks.values()): raise AssertionError("Graph/topic invariance or K7 regression check failed; see saved JSON")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k7-dir", type=Path, required=True); parser.add_argument("--k10-dir", type=Path, required=True)
    parser.add_argument("--reference-k7-dir", type=Path); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.k7_dir, args.k10_dir, args.reference_k7_dir, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
