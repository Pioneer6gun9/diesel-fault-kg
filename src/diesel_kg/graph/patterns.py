#!/usr/bin/env python3
"""Case-grounded linear, branching and convergent fault-influence candidates.

The original paper specifies an unbranched/unmerged linear chain. Unlike the
older global nonexclusive walk count, this module checks each case projection.
All occurrences retain literal text evidence; causality and simultaneity are
not inferred from co-mention. The existing graphs are never filtered/modified.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
from pathlib import Path
import re
import time
import sys

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diesel_kg.graph.analysis import dump, csv_write, sha, rows, projection, count_patterns

INFLUENTIAL_KINDS = {"causal_candidate", "functional_influence", "mechanism_chain_inherited_candidate"}
RECOVERY_CUES = ["恢复", "正常", "消除", "改善", "良好", "稳定", "顺畅", "平稳", "灵活", "精准", "精确"]


def candidate_projection(nodes, analysis_edges):
    """Only exported causal/functional and their inherited mechanism edges.

    Membership, spatial, structural and manifestation edges do not assert a
    fault influence. Excluding them is a mode-analysis projection, not editing
    the previously frozen context-path or propagation-candidate outputs.
    """
    return projection(nodes, [e for e in analysis_edges if e["kind"] in INFLUENTIAL_KINDS])


def local_modes(graph):
    """Return reproducible topology candidates in one case-supported graph."""
    found = []
    for component in sorted(nx.weakly_connected_components(graph), key=lambda c: sorted(c)):
        sub = graph.subgraph(component)
        starts = [n for n in sub if sub.in_degree(n) == 0 and sub.out_degree(n) == 1]
        ends = [n for n in sub if sub.in_degree(n) == 1 and sub.out_degree(n) == 0]
        if (len(sub) >= 3 and sub.number_of_edges() == len(sub) - 1 and len(starts) == len(ends) == 1
                and all(sub.in_degree(n) <= 1 and sub.out_degree(n) <= 1 for n in sub)
                and nx.is_directed_acyclic_graph(sub)):
            chain = list(nx.topological_sort(sub))
            found.append({"pattern": "linear", "node_ids": chain,
                          "edge_pairs": list(zip(chain, chain[1:])),
                          "definition": "maximal weak component is one directed chain; >=3 nodes; no branching/convergence/cycle"})
    reach = {n: nx.descendants(graph, n) for n in graph}
    for center in sorted(graph):
        for successors in itertools.combinations(sorted(graph.successors(center)), 2):
            a, b = successors
            # Dependent downstream nodes form a chain/shortcut rather than two
            # independent branches. This check is stronger than out-degree >=2.
            if a not in reach[b] and b not in reach[a]:
                found.append({"pattern": "branching", "node_ids": [center, a, b],
                              "edge_pairs": [(center, a), (center, b)],
                              "definition": "one source with two mutually unreachable immediate downstream nodes; simultaneous occurrence unverified"})
        for predecessors in itertools.combinations(sorted(graph.predecessors(center)), 2):
            a, b = predecessors
            if a not in reach[b] and b not in reach[a]:
                found.append({"pattern": "convergent", "node_ids": [a, b, center],
                              "edge_pairs": [(a, center), (b, center)],
                              "definition": "two mutually unreachable immediate predecessor sources share one target; physical independence/superposition unverified"})
    return found


def literal_spans(text, word):
    return [{"start": m.start(), "end": m.end(), "text": word} for m in re.finditer(re.escape(word), text)] if word else []


def node_evidence(node, text):
    if node.get("layer") == "mechanism":
        spans = [{**s, "dictionary_term": term} for term in node.get("dictionary_terms", []) for s in literal_spans(text, term)]
        kind = "mechanism_dictionary_occurrence_not_observed_entity_or_event"
    else:
        spans = literal_spans(text, node["name"])
        kind = "exact_literal_name_occurrence"
    return {"node_id": node["id"], "name": node["name"], "entity_type": node["entity_type"],
            "evidence_kind": kind, "literal_spans": spans,
            "positive_or_recovery_cues_in_node": [w for w in RECOVERY_CUES if w in node["name"]]}


def analyze_layer(label, nodes, edges, cases, output, support_mode="literal"):
    graph = candidate_projection(nodes, edges)
    influential_by_id = {edge["id"]: edge for edge in edges if edge["kind"] in INFLUENTIAL_KINDS}
    edge_by_case = collections.defaultdict(list)
    for s, t, attrs in graph.edges(data=True):
        for cid in attrs["case_support"]:
            edge_by_case[cid].append((s, t))
    cache, frequency, case_rows = {}, collections.defaultdict(set), []
    pattern_instances = collections.Counter()
    path = output / f"{label}_pattern_instances.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            cid, text = case["case_id"], case["text"]
            signature = tuple(sorted(set(edge_by_case[cid])))
            if signature not in cache:
                local = nx.DiGraph()
                local.add_edges_from(signature)
                cache[signature] = local_modes(local)
            counts = collections.Counter()
            for mode in cache[signature]:
                identity = (mode["pattern"], tuple(mode["node_ids"]))
                frequency[identity].add(cid)
                evidence = []
                for s, t in mode["edge_pairs"]:
                    attrs = graph[s][t]
                    edge_ids, kinds, basis_ids, weight = attrs["edge_ids"], attrs["kinds"], attrs["basis_base_edge_ids"], attrs["weight"]
                    if support_mode == "model_occurrences":
                        supporting = [influential_by_id[eid] for eid in attrs["edge_ids"]
                                      if cid in influential_by_id[eid]["case_support"]]
                        if not supporting: raise ValueError("Case projection has no supporting typed influence relation")
                        edge_ids = [edge["id"] for edge in supporting]
                        kinds = sorted({edge["kind"] for edge in supporting})
                        basis_ids = sorted({bid for edge in supporting for bid in edge["basis_base_edge_ids"]})
                        # The same typed-edge feature formula is preserved; the
                        # mean now uses only relations supporting this case.
                        weight = sum(sum(edge[key] for key in ["semantic", "cooccurrence", "physical"]) / 3
                                     for edge in supporting) / len(supporting)
                    edge_record = {"source": s, "target": t, "weight": weight,
                                     "kinds": kinds, "edge_ids": edge_ids,
                                     "basis_base_edge_ids": basis_ids,
                                     "support_kind": ("input_model_qualified_occurrence_case_with_model_or_inherited_relation_not_causal_gold"
                                                      if support_mode == "model_occurrences" else "literal_endpoint_comention_with_exported_or_inherited_relation")}
                    if support_mode == "model_occurrences":
                        edge_record["weight_scope"] = "mean_of_existing_typed_graph_features_for_relations_supporting_this_case"
                        edge_record["whole_graph_pair_aggregate_weight"] = attrs["weight"]
                    evidence.append(edge_record)
                record = {**mode, "layer": label, "case_id": cid, "support_mode": support_mode, "source_case_text": text,
                          "source_case_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                          "node_evidence": [node_evidence(nodes[n], text) for n in mode["node_ids"]],
                          "edge_evidence": evidence, "average_edge_weight": sum(e["weight"] for e in evidence) / len(evidence),
                          "causal_gold": False, "temporally_observed": False,
                          "recovery_review_required": any(any(w in nodes[n]["name"] for w in RECOVERY_CUES) for n in mode["node_ids"])}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[mode["pattern"]] += 1
            pattern_instances.update(counts)
            case_rows.append({"case_id": cid, "n_case_influence_edges": len(signature),
                              **{f"n_{p}_instances": counts[p] for p in ["linear", "branching", "convergent"]}})
    body_by_id = {c["case_id"]: c["text"] for c in cases}
    aggregate = [{"pattern": p, "node_ids": list(ids), "names": [nodes[n]["name"] for n in ids],
                  "original_case_frequency": len(cids), "unique_body_frequency": len({body_by_id[c] for c in cids}),
                  "case_ids": sorted(cids), "causal_gold": False}
                 for (p, ids), cids in frequency.items()]
    aggregate.sort(key=lambda r: (r["pattern"], -r["original_case_frequency"], r["node_ids"]))
    csv_write(output / f"{label}_pattern_frequency.csv", aggregate,
              ["pattern", "node_ids", "names", "original_case_frequency", "unique_body_frequency", "case_ids", "causal_gold"])
    csv_write(output / f"{label}_case_pattern_counts.csv", case_rows)
    unique_counts = collections.Counter(r["pattern"] for r in aggregate)
    return {"analysis_nodes": len(graph), "influence_directed_pairs": graph.number_of_edges(),
            "pattern_evaluation_status": "not_evaluable_no_fault_influence_edges" if not graph.number_of_edges() else "computed_topology_candidates_without_causal_gold",
            "pattern_unique_sequences": {p: unique_counts[p] for p in ["linear", "branching", "convergent"]},
            "pattern_case_incidences": {p: pattern_instances[p] for p in ["linear", "branching", "convergent"]},
            "cases_with_each_pattern": {p: sum(r[f"n_{p}_instances"] > 0 for r in case_rows) for p in ["linear", "branching", "convergent"]},
            "n_unique_case_projections": len(cache), "all_case_projections_completed": len(case_rows) == len(cases)}


def run(graph_dir, cases_file, output):
    started = time.perf_counter()
    graph_dir, output = Path(graph_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cases = rows(cases_file)
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate original case IDs")
    basic = json.loads((graph_dir / "base_graph.json").read_text())
    enhanced = json.loads((graph_dir / "enhanced_graph.json").read_text())
    support_mode = basic.get("support_mode", "literal")
    if support_mode != enhanced.get("support_mode", "literal"):
        raise ValueError("Base/enhanced graphs must use the same case-support provenance")
    result = {}
    for label, data in [("base", basic), ("enhanced", enhanced)]:
        nodes = {n["id"]: n for n in data["nodes"]}
        result[label] = analyze_layer(label, nodes, data["analysis_edges"], cases, output, support_mode)
    all_nodes = {n["id"]: n for n in enhanced["nodes"]}
    global_reference = count_patterns(projection(all_nodes, enhanced["analysis_edges"]))
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ["linear", "branching", "convergent"]
    x = list(range(3))
    ax.bar([i - .2 for i in x], [result["base"]["pattern_unique_sequences"][p] for p in labels], .4, label="Basic")
    ax.bar([i + .2 for i in x], [result["enhanced"]["pattern_unique_sequences"][p] for p in labels], .4, label="Enhanced")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Unique case-supported topology candidates")
    ax.set_title("Case fault-influence modes; causal/temporal confirmation pending")
    ax.legend(); fig.tight_layout()
    for suffix in ["png", "svg"]: fig.savefig(output / f"case_pattern_comparison.{suffix}")
    plt.close(fig)
    summary = {"status": "completed_case_supported_topology_analysis", "support_mode": support_mode, "n_original_cases": len(cases),
               "n_unique_bodies": len({c["text"] for c in cases}), "layers": result,
               "older_global_context_motifs_reference": global_reference,
               "projection_kinds": sorted(INFLUENTIAL_KINDS),
               "linear_definition": "a complete weak component is a directed chain of >=3 nodes with no branching, convergence or cycle",
               "branching_definition": "one source and a pair of mutually unreachable immediate downstream nodes",
               "convergent_definition": "a pair of mutually unreachable immediate predecessor nodes with one shared target",
               "pattern_counting": "unordered branch/predecessor pairs; duplicate numbered cases retained; unique-body counts reported separately",
               "recovery_review_policy": "flags are exported without deleting any frozen graph or result",
               "limitations": [("Case support is declared input-model-qualified occurrence case IDs; model/inherited directions are not independent causal gold."
                                 if support_mode == "model_occurrences" else "Case support is literal co-mention; exported/inherited directions are not independent causal gold."),
                               "Branch simultaneity, convergent source physical independence and hazard levels cannot be inferred from static text.",
                               "These case-qualified modes use causal/functional/inherited edges; older global context walks are a different statistic."],
               "input_sha256": {"cases": sha(cases_file), "base_graph": sha(graph_dir / "base_graph.json"),
                                "enhanced_graph": sha(graph_dir / "enhanced_graph.json"), "script": sha(__file__)},
               "command": sys.argv, "libraries": {"networkx": nx.__version__, "matplotlib": matplotlib.__version__},
               "elapsed_seconds": time.perf_counter() - started}
    dump(output / "analysis_summary.json", summary); dump(output / "run_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.graph_dir, args.cases, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
