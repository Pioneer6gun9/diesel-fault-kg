#!/usr/bin/env python3
"""Verify saved mode text offsets, edge provenance, topology and hub/scope totals."""
from __future__ import annotations
import argparse
import collections
import csv
import json
from pathlib import Path
import networkx as nx
from diesel_kg.graph.analysis import dump, rows, sha


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle: return list(csv.DictReader(handle))


def run(graph_dir, cases, patterns_dir, hub_dir, output):
    graph_dir, patterns_dir, hub_dir = Path(graph_dir), Path(patterns_dir), Path(hub_dir)
    source = {r["case_id"]: r["text"] for r in rows(cases)}
    patterns = json.loads((patterns_dir / "analysis_summary.json").read_text())
    hub = json.loads((hub_dir / "analysis_summary.json").read_text())
    checks, details = {}, {}
    for label in ["base", "enhanced"]:
        graph = json.loads((graph_dir / f"{label}_graph.json").read_text())
        edges = {e["id"]: e for e in graph["analysis_edges"]}
        by_case = collections.defaultdict(nx.DiGraph)
        for edge in graph["analysis_edges"]:
            if edge["kind"] in {"causal_candidate", "functional_influence", "mechanism_chain_inherited_candidate"} and edge["source"] != edge["target"]:
                for cid in edge["case_support"]: by_case[cid].add_edge(edge["source"], edge["target"])
        records = rows(patterns_dir / f"{label}_pattern_instances.jsonl")
        spans = 0
        for r in records:
            text = source[r["case_id"]]
            assert r["source_case_text"] == text
            assert r["causal_gold"] is False and r["temporally_observed"] is False
            for n in r["node_evidence"]:
                assert n["literal_spans"], "Missing node literal/dictionary evidence"
                for span in n["literal_spans"]:
                    assert text[span["start"]:span["end"]] == span["text"]
                    spans += 1
            g = by_case[r["case_id"]]
            for evidence in r["edge_evidence"]:
                assert g.has_edge(evidence["source"], evidence["target"])
                for eid in evidence["edge_ids"]:
                    edge = edges[eid]
                    assert r["case_id"] in edge["case_support"]
                    assert (edge["source"], edge["target"]) == (evidence["source"], evidence["target"])
                if graph.get("support_mode") == "model_occurrences":
                    supporting = [edge for edge in edges.values()
                                  if (edge["source"], edge["target"]) == (evidence["source"], evidence["target"])
                                  and edge["kind"] in {"causal_candidate", "functional_influence", "mechanism_chain_inherited_candidate"}
                                  and r["case_id"] in edge["case_support"]]
                    assert set(evidence["edge_ids"]) == {edge["id"] for edge in supporting}
                    assert evidence["kinds"] == sorted({edge["kind"] for edge in supporting})
                    assert evidence["basis_base_edge_ids"] == sorted({bid for edge in supporting for bid in edge["basis_base_edge_ids"]})
                    expected = sum(sum(edge[key] for key in ["semantic", "cooccurrence", "physical"])/3 for edge in supporting)/len(supporting)
                    assert abs(evidence["weight"]-expected) < 1e-12
            if graph.get("support_mode") == "model_occurrences":
                assert abs(r["average_edge_weight"]-sum(e["weight"] for e in r["edge_evidence"])/len(r["edge_evidence"])) < 1e-12
            ids = r["node_ids"]
            if r["pattern"] == "linear":
                assert len(ids) >= 3
                assert g.in_degree(ids[0]) == 0 and g.out_degree(ids[-1]) == 0
                assert all(g.out_degree(n) == 1 for n in ids[:-1])
                assert all(g.in_degree(n) == 1 for n in ids[1:])
                assert all(g.has_edge(s, t) for s, t in zip(ids, ids[1:]))
                assert len(set(ids)) == len(ids)
            elif r["pattern"] == "branching":
                center, a, b = ids
                assert g.has_edge(center, a) and g.has_edge(center, b)
                assert not nx.has_path(g, a, b) and not nx.has_path(g, b, a)
            elif r["pattern"] == "convergent":
                a, b, center = ids
                assert g.has_edge(a, center) and g.has_edge(b, center)
                assert not nx.has_path(g, a, b) and not nx.has_path(g, b, a)
            else: raise AssertionError("Unknown mode")
        counted = collections.Counter(r["pattern"] for r in records)
        unique = collections.Counter(p for p, _ in {(r["pattern"], tuple(r["node_ids"])) for r in records})
        assert {p: counted[p] for p in ["linear", "branching", "convergent"]} == patterns["layers"][label]["pattern_case_incidences"]
        assert {p: unique[p] for p in ["linear", "branching", "convergent"]} == patterns["layers"][label]["pattern_unique_sequences"]
        assert len(read_csv(patterns_dir / f"{label}_case_pattern_counts.csv")) == len(source)
        checks[label + "_case_patterns_source_spans_topology_support_counts"] = True
        details[label + "_patterns"] = {"instances": len(records), "verified_text_spans": spans}
        hubs = read_csv(hub_dir / f"{label}_node_hub_metrics.csv")
        assert len(hubs) == len(graph["nodes"])
        assert abs(sum(float(n["pagerank"]) for n in hubs) - 1.) < 1e-9
        assert abs(sum(float(n["cross_system_weighted_out_degree"]) for n in hubs)
                   - hub["hub_metrics"][label]["total_cross_system_directed_weight"]) < 1e-9
        hierarchy = rows(hub_dir / f"{label}_path_hierarchy.jsonl")
        paths = {tuple(r["node_ids"]): r for r in rows(graph_dir / f"{label}_propagation_candidate_paths.jsonl")}
        assert len(hierarchy) == len(paths)
        counts = collections.Counter()
        for r in hierarchy:
            assert r["case_ids"] == paths[tuple(r["node_ids"])]["case_ids"]
            assert r["propagation_speed"] is None and r["hazard_rank"] is None and r["observed_temporal_progression"] is False
            assert r["scope_level_candidate"] != "whole_engine", "Path cross-system scope must not be pooled into whole-engine failure"
            if r["contains_explicit_engine_consequence_phrase"]:
                assert r["scope_level_candidate"] == "explicit_engine_consequence"
            elif len(r["mapped_systems"]) > 1:
                assert r["scope_level_candidate"] == "cross_system_candidate"
            counts[r["scope_level_candidate"]] += 1
        for level, total in hub["hierarchy_metrics"][label]["unique_path_scope_counts"].items():
            assert counts[level] == total
        checks[label + "_hub_mass_scope_counts_no_invented_speed"] = True
        details[label + "_hierarchy"] = {"qualified_paths_checked": len(hierarchy), "hub_nodes_checked": len(hubs)}
    result = {"status": "passed", "checks": checks, "details": details,
              "scope": "artifact provenance/internal consistency; not independent expert causal or temporal validation",
              "input_sha256": {"cases": sha(cases), "script": sha(__file__)}}
    dump(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["graph-dir", "cases", "patterns-dir", "hub-dir", "output"]: parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.graph_dir, args.cases, args.patterns_dir, args.hub_dir, args.output), indent=2))


if __name__ == "__main__": main()
