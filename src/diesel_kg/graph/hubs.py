#!/usr/bin/env python3
"""Compute node/system hubs and inspect the paper's three static scope levels.

Seven engineering-system mappings are retained from the topic pipeline. Hub
scores are computed, not chosen to make the fuel system dominant. Explicit
node/path rule evidence is exported. No propagation-speed or severity values
are fabricated without event timestamps and clinical/engineering annotations.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import time
import sys

import networkx as nx
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diesel_kg.graph.analysis import SYSTEMS, PHYSICAL_TYPES, dump, sha, rows, csv_write, projection

ENGINE_TERMS = ["整机", "停机", "无法启动", "不能启动", "发动机损坏", "动力完全", "动力完全丧失", "动力丧失", "功率完全丧失"]
ENGINE_PHYSICAL_NAMES = {"发动机", "柴油机", "整机", "柴油发动机"}
LEVELS = ["component", "subsystem", "whole_engine", "unknown"]
PATH_SCOPE_LEVELS = ["component", "subsystem", "cross_system_candidate", "explicit_engine_consequence", "unknown"]


def mapped_system(node):
    vector = np.asarray(node["system_distribution"])
    return SYSTEMS[int(np.argmax(vector))] if vector.sum() > 0 else None


def calculate_hubs(nodes, graph):
    cost = graph.copy()
    for _, _, attrs in cost.edges(data=True):
        # A disclosed graph-analysis distance, not a calibrated risk measure.
        attrs["distance"] = 1.0 / max(float(attrs["weight"]), 1e-12)
    betweenness = nx.betweenness_centrality(cost, normalized=True, weight="distance")
    pagerank = nx.pagerank(graph, alpha=.85, weight="weight", max_iter=1000, tol=1e-12)
    systems = {n: mapped_system(nodes[n]) for n in graph}
    cross_in, cross_out = collections.Counter(), collections.Counter()
    unknown_incident = collections.Counter()
    for s, t, attrs in graph.edges(data=True):
        ss, ts = systems[s], systems[t]
        if ss is None or ts is None:
            unknown_incident[s] += attrs["weight"]; unknown_incident[t] += attrs["weight"]
        elif ss != ts:
            cross_out[s] += attrs["weight"]; cross_in[t] += attrs["weight"]
    denominator = max(len(graph) - 1, 1)
    records = []
    for n in graph:
        records.append({"node_id": n, "name": nodes[n]["name"], "entity_type": nodes[n]["entity_type"],
                        "mapped_system": systems[n] or "unknown", "mapping_rule": "argmax_nonzero_keyword_topic_system_vector",
                        "weighted_in_degree": graph.in_degree(n, weight="weight"), "weighted_out_degree": graph.out_degree(n, weight="weight"),
                        "cross_system_weighted_in_degree": cross_in[n], "cross_system_weighted_out_degree": cross_out[n],
                        "cross_system_weighted_total_degree": cross_in[n] + cross_out[n],
                        "cross_system_normalized_total_degree": (cross_in[n] + cross_out[n]) / (2 * denominator),
                        "unknown_system_incident_weight": unknown_incident[n],
                        "betweenness_reciprocal_weight_distance": betweenness[n], "pagerank": pagerank[n],
                        "n_literal_case_support": len(nodes[n]["case_support"]), "clinical_hub_gold": False})
    records.sort(key=lambda r: (-r["cross_system_weighted_total_degree"], -r["betweenness_reciprocal_weight_distance"], r["node_id"]))
    for rank, record in enumerate(records, 1): record["cross_system_degree_rank"] = rank
    all_cross_weight = sum(cross_out.values())
    system_records = []
    for system in SYSTEMS:
        members = [r for r in records if r["mapped_system"] == system]
        outgoing = sum(r["cross_system_weighted_out_degree"] for r in members)
        incoming = sum(r["cross_system_weighted_in_degree"] for r in members)
        system_records.append({"system": system, "n_mapped_nodes": len(members), "cross_system_weighted_in": incoming,
                               "cross_system_weighted_out": outgoing, "cross_system_weighted_total": incoming + outgoing,
                               "cross_system_incident_weight_share": (incoming + outgoing) / (2 * all_cross_weight) if all_cross_weight else None,
                               "mean_normalized_cross_system_degree": np.mean([r["cross_system_normalized_total_degree"] for r in members]) if members else 0.,
                               "sum_betweenness": sum(r["betweenness_reciprocal_weight_distance"] for r in members),
                               "mean_betweenness": np.mean([r["betweenness_reciprocal_weight_distance"] for r in members]) if members else 0.,
                               "pagerank_mass": sum(r["pagerank"] for r in members)})
    system_records.sort(key=lambda r: (-r["cross_system_weighted_total"], r["system"]))
    for rank, record in enumerate(system_records, 1): record["cross_system_weight_rank"] = rank
    summary = {"n_nodes": len(graph), "n_directed_pairs": graph.number_of_edges(), "n_mapped_nodes": sum(s is not None for s in systems.values()),
               "hub_evaluation_status": "not_evaluable_zero_analysis_edges" if not graph.number_of_edges() else "computed_graph_centrality_without_hub_gold",
               "pagerank_interpretation": "dangling-node/uniform baseline without graph-supported hub ranking" if not graph.number_of_edges() else "weighted graph centrality; not hub identification accuracy",
               "n_unknown_system_nodes": sum(s is None for s in systems.values()), "total_cross_system_directed_weight": all_cross_weight,
               "leading_system_by_cross_system_weight": system_records[0]["system"] if all_cross_weight else None,
               "pagerank_sum": sum(pagerank.values()), "betweenness_mode": "exact directed weighted; distance=1/max(weight,1e-12)",
               "normalized_cross_system_total_degree": "cross-in+cross-out divided by 2*(all_graph_nodes-1)",
               "system_incident_weight_share": "system cross-in+cross-out divided by twice all mapped cross-system directed weight; sums to 1"}
    return records, system_records, summary


def hierarchy_nodes(nodes, edges):
    physical_context = collections.defaultdict(set)
    for edge in edges:
        if edge["kind"] in {"structural_context", "spatial_context", "physical_mechanism_context"}:
            for physical, fault in [(edge["source"], edge["target"]), (edge["target"], edge["source"])]:
                if nodes[physical]["entity_type"] in PHYSICAL_TYPES:
                    physical_context[fault].add(physical)
    records = {}
    for n, node in nodes.items():
        system = mapped_system(node)
        engine_terms = [w for w in ENGINE_TERMS if w in node["name"]]
        contextual = sorted(physical_context[n])
        components = [p for p in contextual if nodes[p]["entity_type"] != "系统" and nodes[p]["name"] not in ENGINE_PHYSICAL_NAMES]
        if engine_terms and node["entity_type"] in {"故障现象", "影响后果", "故障状态"}:
            level, rule = "whole_engine", "explicit_engine_consequence_phrase"
        elif node["entity_type"] == "系统":
            level, rule = "subsystem", "exported_entity_type_system"
        elif node["entity_type"] in PHYSICAL_TYPES and node["name"] not in ENGINE_PHYSICAL_NAMES:
            level, rule = "component", "exported_physical_entity_except_generic_engine"
        elif len(components) == 1:
            level, rule = "component", "one_component_in_exported_structural_spatial_or_physical_context"
        elif len(components) > 1 and system is not None:
            level, rule = "subsystem", "multiple_context_components_and_one_weak_system_argmax"
        else:
            level, rule = "unknown", "insufficient_explicit_component_or_engine_scope_evidence"
        records[n] = {"node_id": n, "name": node["name"], "entity_type": node["entity_type"], "scope_level_candidate": level,
                      "assignment_rule": rule, "physical_context_node_ids": contextual, "component_context_node_ids": components,
                      "mapped_system": system, "matched_engine_terms": engine_terms,
                      "evolution_stage_verified": False, "severity_verified": False}
    return records


def hierarchy_paths(paths, nodes, node_levels, case_text_by_id, label, output, support_mode="literal"):
    counts, incidences, signatures = collections.Counter(), collections.Counter(), collections.Counter()
    transitions = collections.Counter()
    export = []
    for path in paths:
        ids = path["node_ids"]
        systems = sorted({mapped_system(nodes[n]) for n in ids} - {None})
        components = {n for n in ids if nodes[n]["entity_type"] in PHYSICAL_TYPES and nodes[n]["entity_type"] != "系统"
                      and nodes[n]["name"] not in ENGINE_PHYSICAL_NAMES}
        for n in ids: components.update(node_levels[n]["component_context_node_ids"])
        explicit_engine = any(node_levels[n]["assignment_rule"] == "explicit_engine_consequence_phrase" for n in ids)
        if explicit_engine:
            level, rule = "explicit_engine_consequence", "path_contains_explicit_engine_consequence_phrase"
        elif len(systems) > 1:
            level, rule = "cross_system_candidate", "cross_system_scope_candidate_not_verified_whole_engine_failure"
        elif len(components) == 1:
            level, rule = "component", "path_context_references_exactly_one_component_and_at_most_one_system"
        elif len(systems) == 1 and len(components) > 1:
            level, rule = "subsystem", "multiple_components_within_one_weakly_mapped_system"
        else:
            level, rule = "unknown", "insufficient_path_scope_evidence"
        node_sequence = [node_levels[n]["scope_level_candidate"] for n in ids]
        for a, b in zip(node_sequence, node_sequence[1:]): transitions[a, b] += 1
        signature = " -> ".join(node_sequence)
        counts[level] += 1; incidences[level] += path["literal_case_frequency"]; signatures[signature] += 1
        export.append({"layer": label, "support_mode": support_mode, "node_ids": ids, "names": [nodes[n]["name"] for n in ids],
                       "scope_level_candidate": level, "scope_assignment_rule": rule, "node_level_sequence": node_sequence,
                       "component_context_node_ids": sorted(components), "mapped_systems": systems,
                       "contains_explicit_engine_consequence_phrase": explicit_engine,
                       "case_ids": path["case_ids"], "literal_case_frequency": path["literal_case_frequency"],
                       "unique_body_frequency": len({case_text_by_id[c] for c in path["case_ids"]}),
                       "average_edge_weight": path["average_edge_weight"],
                       "observed_temporal_progression": False, "propagation_speed": None, "hazard_rank": None,
                       "evidence_status": ("input_model_qualified_occurrence_case_support_and_static_scope_rules_without_temporal_or_causal_gold"
                                           if support_mode == "model_occurrences" else "case_literal_support_and_static_graph_scope_rules_without_temporal_or_clinical_gold"),
                       "frequency_field_interpretation": ("literal_case_frequency retains the interface name but counts input-model-qualified cases"
                                                          if support_mode == "model_occurrences" else "literal endpoint co-mention cases")})
    with (output / f"{label}_path_hierarchy.jsonl").open("w", encoding="utf-8") as handle:
        for record in export: handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    csv_write(output / f"{label}_hierarchy_transitions.csv", [{"source_level": a, "target_level": b,
              "static_path_transition_count": n, "temporal_transition_count": "unavailable"} for (a, b), n in sorted(transitions.items())])
    return {"n_paths": len(paths), "unique_path_scope_counts": {l: counts[l] for l in PATH_SCOPE_LEVELS},
            "static_scope_status": "not_evaluable_no_candidate_paths" if not paths else "static_scope_candidates_without_temporal_gold",
            "path_case_incidences_by_scope": {l: incidences[l] for l in PATH_SCOPE_LEVELS},
            "node_level_sequence_counts": dict(signatures),
            "explicit_engine_consequence_paths": sum(r["contains_explicit_engine_consequence_phrase"] for r in export),
            "cross_system_scope_paths_without_explicit_engine_consequence": sum(r["scope_assignment_rule"].startswith("cross_system") for r in export),
            "no_event_timestamp_data": True, "measured_speed": None, "measured_severity": None}


def run(graph_dir, cases_file, output):
    started = time.perf_counter(); graph_dir, output = Path(graph_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cases = rows(cases_file); case_text = {c["case_id"]: c["text"] for c in cases}
    if len(case_text) != len(cases): raise ValueError("Duplicate case IDs")
    layers, hierarchy_stats, hub_stats = {}, {}, {}
    support_mode = None
    for label in ["base", "enhanced"]:
        data = json.loads((graph_dir / f"{label}_graph.json").read_text())
        layer_support_mode = data.get("support_mode", "literal")
        if support_mode is not None and support_mode != layer_support_mode:
            raise ValueError("Base/enhanced graphs must use the same case-support provenance")
        support_mode = layer_support_mode
        nodes = {n["id"]: n for n in data["nodes"]}; graph = projection(nodes, data["analysis_edges"])
        hubs, systems, summary = calculate_hubs(nodes, graph)
        csv_write(output / f"{label}_node_hub_metrics.csv", hubs)
        csv_write(output / f"{label}_system_hub_metrics.csv", systems)
        hub_stats[label] = summary; layers[label] = {"nodes": nodes, "system_hubs": systems}
        levels = hierarchy_nodes(nodes, data["analysis_edges"])
        csv_write(output / f"{label}_node_hierarchy.csv", levels.values())
        paths = rows(graph_dir / f"{label}_propagation_candidate_paths.jsonl")
        hierarchy_stats[label] = hierarchy_paths(paths, nodes, levels, case_text, label, output, support_mode)
        hierarchy_stats[label]["node_scope_counts"] = dict(collections.Counter(r["scope_level_candidate"] for r in levels.values()))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    system_by_name = {r["system"]: r for r in layers["enhanced"]["system_hubs"]}
    axes[0].barh(SYSTEMS, [system_by_name[s]["cross_system_weighted_total"] for s in SYSTEMS])
    axes[0].set_xlabel("Cross-system weighted in+out degree")
    axes[0].set_title("Computed hub systems (7 mapped systems)")
    x = list(range(len(PATH_SCOPE_LEVELS)))
    for shift, label in [(-.2, "base"), (.2, "enhanced")]:
        axes[1].bar([i + shift for i in x], [hierarchy_stats[label]["unique_path_scope_counts"][l] for l in PATH_SCOPE_LEVELS], .4, label=label)
    axes[1].set_xticks(x, ["component", "subsystem", "cross-system\ncandidate", "explicit engine\nconsequence", "unknown"], rotation=20)
    axes[1].set_ylabel("Static qualified-path scope candidates")
    axes[1].set_title("Cross-system scope separated from engine consequences\nThree-level chronological evolution unverified"); axes[1].legend()
    fig.tight_layout()
    for suffix in ["png", "svg"]: fig.savefig(output / f"hub_and_hierarchy.{suffix}")
    plt.close(fig)
    dump(output / "hierarchy_rules.json", {"node_scope_levels": LEVELS, "path_scope_levels": PATH_SCOPE_LEVELS, "engine_consequence_phrases": ENGINE_TERMS,
        "generic_engine_physical_names": sorted(ENGINE_PHYSICAL_NAMES), "mapping_dimension": len(SYSTEMS),
        "interpretation": "static candidate scope, not severity or chronological evolution",
        "node_rules_priority": ["explicit engine consequence phrase -> whole_engine", "exported system type -> subsystem",
                                "physical component excluding generic engine -> component", "exactly one physical context component -> component",
                                "multiple context components and nonzero system mapping -> subsystem", "otherwise unknown"],
        "path_rules_priority": ["explicit engine consequence -> explicit_engine_consequence", "multiple mapped systems -> cross_system_candidate; do not label whole-engine failure",
                                "exactly one context component -> component", "multiple context components and one mapped system -> subsystem", "otherwise unknown"],
        "time_rules": "No slow/medium/fast category without time observations and prespecified speed thresholds"})
    summary = {"status": "completed_hub_and_static_hierarchy_analysis", "support_mode": support_mode, "n_original_cases": len(cases), "systems": SYSTEMS,
               "legacy_literal_support_count_interpretation": ("n_literal_case_support/literal_case_frequency count input-model-qualified occurrence cases in this branch"
                                                               if support_mode == "model_occurrences" else "literal endpoint co-mention cases"),
               "hub_metrics": hub_stats, "hierarchy_metrics": hierarchy_stats,
               "temporal_evolution": {"status": "not_identifiable_from_available_static_cases", "event_times_available": False,
                                      "speed_categories": None, "hazard_categories": None},
               "limitations": ["Topic-anchored seven-system mapping is not expert system gold; original eight-bin figure mapping was not provided.",
                               "Weighted centrality is a computed graph property, not core-node identification accuracy.",
                               "Cross-system scope does not establish whole-engine failure, chronology, simultaneity, severity or propagation velocity.",
                               "Positive/recovery-phase candidates are retained for separate adjudication; frozen graphs are unchanged."],
               "input_sha256": {"cases": sha(cases_file), "base_graph": sha(graph_dir / "base_graph.json"),
                                "enhanced_graph": sha(graph_dir / "enhanced_graph.json"), "script": sha(__file__)},
               "command": sys.argv, "libraries": {"networkx": nx.__version__, "matplotlib": matplotlib.__version__},
               "elapsed_seconds": time.perf_counter() - started}
    dump(output / "analysis_summary.json", summary); dump(output / "run_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True); parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.graph_dir, args.cases, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
