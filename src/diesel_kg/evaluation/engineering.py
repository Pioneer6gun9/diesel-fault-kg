#!/usr/bin/env python3
"""Executable engineering comparisons for paper section 6.

Execution is a retrospective query replay against the declared graph source:
historical literal grounding or model-qualified occurrence cases. Neither is
automatically an independent industrial diagnostic accuracy experiment.
Gold and expert scores are optional real inputs; absent labels never become zeros
or inferred labels. All candidate scores are association scores, not probabilities.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import heapq
import json
import math
from pathlib import Path
import platform
import re
import time

import networkx as nx
import numpy as np

INFLUENCE = {"causal_candidate", "functional_influence", "mechanism_chain_inherited_candidate"}
FIRST_CONTEXT = {"structural_context", "spatial_context", "physical_mechanism_context"}
PHYSICAL = {"故障部位", "部件", "组件", "系统"}
ROOT_TYPES = PHYSICAL | {"故障原因", "故障状态"}
SYMPTOM_TYPES = {"故障现象", "影响后果"}
RECOVERY = re.compile(r"恢复|(?<!不)(?<!未)正常|消除|解决|修复|排除|改善|精准|(?<!不)平稳|顺畅|稳定运行")
INSPECTION = re.compile(r"检查发现|经检查|检查|拆检|排查|更换|清洗|调整|维修|修复|处理|修理|检修")
SIS_WEIGHTS = {"S_SC": .30, "S_PC": .25, "S_CC": .25, "S_DA": .20}


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def csv_write(path, records, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: json.dumps(r[k], ensure_ascii=False) if isinstance(r.get(k), (list, dict)) else r.get(k, "") for k in fields})


def write_jsonl(path, records):
    with Path(path).open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")


def load_graph(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in d["nodes"]}
    grouped = defaultdict(list)
    for e in d["analysis_edges"]:
        if e["source"] != e["target"]:
            grouped[e["source"], e["target"]].append(e)
    g = nx.DiGraph()
    g.graph["support_mode"] = d.get("support_mode")
    g.add_nodes_from(nodes)
    for (u, v), rows in grouped.items():
        weight = float(np.mean([r["weight"] for r in rows]))
        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError(f"Invalid association weight on {u}->{v}: {weight}")
        g.add_edge(u, v, weight=weight, cost=-math.log(max(weight, 1e-15)),
                   kinds=sorted({r["kind"] for r in rows}), edge_ids=[r["id"] for r in rows],
                   basis_base_edge_ids=sorted({b for r in rows for b in r.get("basis_base_edge_ids", [])}),
                   case_support=sorted({c for r in rows for c in r.get("case_support", [])}))
    return nodes, g


def graph_source_scope(graph_dir, graph_files, loaded):
    """Read source provenance, including zero-edge graphs; never infer from size."""
    declarations = {}
    for name, path in graph_files.items():
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("support_mode"):
            declarations[name + ".support_mode"] = raw["support_mode"]
    for filename in ["analysis_summary.json", "run_config.json"]:
        path = Path(graph_dir) / filename
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            mode = data.get("support_mode") or data.get("arguments", {}).get("support_mode")
            if mode:
                declarations[filename] = mode
    modes = set(declarations.values())
    if len(modes) > 1:
        raise ValueError(f"Graph provenance support-mode declarations conflict: {declarations}")
    if modes:
        mode = next(iter(modes))
    else:
        # Older historical exports lacked the top-level field but carried an
        # explicit node/edge support-kind. Missing/empty provenance stays unknown.
        kinds = {node.get("case_support_kind", "") for nodes, _ in loaded.values() for node in nodes.values()}
        if any("input_model_qualified" in kind for kind in kinds):
            mode = "model_occurrences"
        elif any("literal" in kind for kind in kinds):
            mode = "literal"
        else:
            mode = "unknown"
    if mode not in {"literal", "model_occurrences", "unknown"}:
        raise ValueError(f"Unsupported declared support mode: {mode}")
    case_ids = sorted({cid for nodes, graph in loaded.values() for node in nodes.values() for cid in node.get("case_support", [])}
                      | {cid for _, graph in loaded.values() for _, _, edge in graph.edges(data=True) for cid in edge["case_support"]})
    if mode == "model_occurrences":
        scope = "model_qualified_occurrence_graph_retrospective_symptom_query_replay"
        kind = "model_qualified_occurrence_relation_candidates"
        support = "actual input model-qualified occurrence case IDs; no full-corpus same-name expansion; model-qualified candidates are not observed causal chains"
        limitation = "graph-mode metadata alone does not prove held-out separation; queries use model relation candidates with declared occurrence case IDs, and causal/diagnostic human gold must be separately verified"
    elif mode == "literal":
        scope = "historical_graph_retrospective_symptom_query_replay"
        kind = "historical_export_literal_grounded_candidates"
        support = "new exact-name literal case grounding of historical-export nodes/links; not recovered historical causal case provenance"
        limitation = "historical graph and topic/grounding construction used the full case source; this replay proves executable query behavior, not unseen-case diagnosis"
    else:
        scope = "unspecified_graph_source_symptom_query_replay"
        kind = "unknown_graph_source"
        support = "input case-support IDs without a declared source mode; source validity is not inferred"
        limitation = "graph provenance has not been declared; query computation alone cannot establish graph origin or independent diagnostic validity"
    for _, graph in loaded.values():
        graph.graph["support_mode"] = mode
    return {"support_mode": mode, "graph_source_kind": kind, "evaluation_scope": scope,
            "support_interpretation": support, "source_leakage_warning": limitation,
            "support_mode_declarations": declarations, "actual_input_case_support_ids": case_ids,
            "n_actual_input_case_support_ids": len(case_ids), "case_support_is_observed_causal_gold": False,
            "graph_source_is_historical_export": mode == "literal"}


def symptom_query(case, nodes):
    """Only query the presentation prefix; never feed the later cause/repair text."""
    m = INSPECTION.search(case["text"])
    prefix = case["text"][:m.start()] if m else case["text"]
    matches = sorted(n for n, d in nodes.items() if d["entity_type"] in SYMPTOM_TYPES
                     and d["name"] and d["name"] in prefix and not RECOVERY.search(d["name"]))
    # Different typed nodes with the same spelling remain distinct canonical IDs.
    return {"query_symptom_node_ids": matches, "query_text": prefix,
            "presentation_cutoff_offset": m.start() if m else None,
            "query_extraction": "exact input-graph symptom name in prefix before first inspection/intervention marker",
            "presentation_boundary_verified_by_human": False}


def path_qualified(ids, nodes, graph):
    if len(ids) < 2 or len(ids) != len(set(ids)) or nodes[ids[-1]]["entity_type"] not in SYMPTOM_TYPES:
        return False
    entity_ids = set(ids)
    if any(nodes[n].get("layer") == "mechanism" and entity_ids.intersection(nodes[n].get("member_entity_ids", [])) for n in ids):
        return False
    influence = False
    for i, (u, v) in enumerate(zip(ids, ids[1:])):
        kinds = set(graph[u][v]["kinds"])
        influence |= bool(kinds & INFLUENCE)
        allowed = kinds & INFLUENCE
        if i == 0 and nodes[u]["entity_type"] in PHYSICAL:
            allowed |= kinds & FIRST_CONTEXT
        if i == len(ids) - 2:
            allowed |= kinds & {"manifestation_context"}
        if not allowed:
            return False
    return influence


def explain_path(ids, graph, nodes):
    links = [{"source": u, "target": v, **graph[u][v]} for u, v in zip(ids, ids[1:])]
    common = sorted(set.intersection(*(set(e["case_support"]) for e in links)))
    result = {"node_ids": list(ids), "node_names": [nodes[n]["name"] for n in ids],
            "node_types": [nodes[n]["entity_type"] for n in ids], "edges": links,
            "product_association_score": float(math.prod(e["weight"] for e in links)),
            "mean_association_weight": float(np.mean([e["weight"] for e in links])),
            "common_case_support_ids": common, "case_support_mode": graph.graph.get("support_mode", "unknown"),
            "has_verified_causal_gold": False}
    if graph.graph.get("support_mode") == "literal":
        result["common_literal_case_ids"] = common
    return result


def infer_query(symptoms, nodes, graph, root_universe, core_k=5):
    """Reverse maximum-product reasoning, with explicitly constrained edge roles.

    One maximum-product suffix per root/symptom is obtained by reverse Dijkstra.
    A physical first context hop may precede it. Invalid member/class redundant
    paths are discarded; alternative longer paths are not exhaustively enumerated.
    The separate saved case path collection is used for gold path completeness.
    """
    best = {}
    rejected_redundancy = 0
    for symptom in symptoms:
        if symptom not in graph:
            continue
        # Dijkstra state tracks whether the suffix contains an influence edge.
        # This avoids allowing a high-weight manifestation-only path to hide a
        # valid causal/influence alternative from the same root.
        distances = {(symptom, False): 0.}
        state_paths = {(symptom, False): (symptom,)}
        heap = [(0., symptom, False)]
        while heap:
            distance, current, seen_influence = heapq.heappop(heap)
            if distance > distances[current, seen_influence]:
                continue
            for previous in sorted(graph.predecessors(current)):
                if previous == symptom:
                    continue
                edge = graph[previous][current]
                kinds = set(edge["kinds"])
                influence = bool(kinds & INFLUENCE)
                if edge["weight"] <= 0 or not (influence or (current == symptom and "manifestation_context" in kinds)):
                    continue
                flag = seen_influence or influence
                cost = distance + edge["cost"]
                key = (previous, flag)
                if cost < distances.get(key, math.inf):
                    distances[key] = cost
                    state_paths[key] = state_paths[current, seen_influence] + (previous,)
                    heapq.heappush(heap, (cost, previous, flag))
        reverse_paths = {n: ids for (n, flag), ids in state_paths.items() if flag}
        for root in sorted(root_universe):
            candidate_paths = []
            if root in reverse_paths and len(reverse_paths[root]) > 1:
                candidate_paths.append(tuple(reversed(reverse_paths[root])))
            if nodes[root]["entity_type"] in PHYSICAL:
                for nxt in graph.successors(root):
                    e = graph[root][nxt]
                    if e["weight"] > 0 and set(e["kinds"]) & FIRST_CONTEXT and nxt in reverse_paths:
                        suffix = tuple(reversed(reverse_paths[nxt]))
                        if root not in suffix:
                            candidate_paths.append((root,) + suffix)
            for ids in candidate_paths:
                if not path_qualified(ids, nodes, graph):
                    rejected_redundancy += 1
                    continue
                score = math.prod(graph[u][v]["weight"] for u, v in zip(ids, ids[1:]))
                key = (score, -len(ids), tuple(ids))
                if root not in best or key > best[root][0]:
                    best[root] = (key, ids)
    ranked = sorted(best, key=lambda n: (-best[n][0][0], len(best[n][1]), n))
    roots = [{"rank": rank + 1, "node_id": n, "name": nodes[n]["name"], "entity_type": nodes[n]["entity_type"],
              "best_explanation": explain_path(best[n][1], graph, nodes)} for rank, n in enumerate(ranked)]
    core_scores, participation = defaultdict(float), Counter()
    for root in roots:
        p = root["best_explanation"]
        for n in p["node_ids"][:-1]:
            if n not in symptoms:
                core_scores[n] += p["product_association_score"]
                participation[n] += 1
    core_order = sorted(core_scores, key=lambda n: (-core_scores[n], -participation[n], n))
    core = [{"rank": i + 1, "node_id": n, "name": nodes[n]["name"], "summed_explanation_association_score": core_scores[n],
             "n_candidate_root_explanations": participation[n]} for i, n in enumerate(core_order[:core_k])]
    return {"candidate_roots": roots, "core_nodes": core, "query_has_symptom": bool(symptoms),
            "query_has_candidate_root": bool(roots), "n_root_candidates": len(roots),
            "n_rejected_best_or_first_context_paths": rejected_redundancy,
            "ranking_is_causal_probability": False, "exhaustive_path_search": False}


def stats(values):
    a = np.asarray(values, dtype=float)
    return {"n": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "p05": float(np.quantile(a, .05)), "p95": float(np.quantile(a, .95))} if len(a) else {
                "n": 0, "mean": None, "median": None, "p05": None, "p95": None}


def paired_bootstrap(rows, fields, seed=42, n_boot=2000):
    """Resample case body groups, maintaining base/enhanced pairing and duplicates."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["group_id"]].append(r)
    group_ids = sorted(groups)
    if not group_ids:
        return {"status": "not_evaluable", "n_case_groups": 0, "metrics": {}}
    rng = np.random.default_rng(seed)
    # Each unique body group has equal weight; repeated numbered cases do not add weight.
    group_rows = [{f: float(np.mean([r[f] for r in groups[g]])) for f in fields} for g in group_ids]
    a = np.array([[r[f] for f in fields] for r in group_rows])
    samples = np.empty((n_boot, len(fields)))
    for i in range(n_boot):
        samples[i] = a[rng.integers(0, len(a), len(a))].mean(axis=0)
    return {"status": "computed", "n_cases": len(rows), "n_case_groups": len(a),
            "resampling_unit": "unique case body or supplied source group; base/enhanced pairing retained",
            "n_bootstrap_samples": n_boot, "seed": seed,
            "metrics": {f: {"mean": float(a[:, j].mean()), "ci95": [float(x) for x in np.quantile(samples[:, j], [.025, .975])]} for j, f in enumerate(fields)}}


def set_scores(pred, gold):
    p, g = set(pred), set(gold)
    if not g:
        return {"precision": None, "recall": None, "f1": None}
    precision = len(p & g) / len(p) if p else 0.
    recall = len(p & g) / len(g)
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.}


def independent_audit(split_path, gold, cases, graph_files):
    failures = ["input_graph_scope_not_proven_train_only"] if not split_path else []
    if not split_path:
        return {"independent_holdout_verified": False, "reasons": failures}
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    train = set(split.get("graph_training_case_ids", []))
    heldout = {g["case_id"] for g in gold}
    declared_heldout = set(split.get("heldout_case_ids", []))
    if not declared_heldout:
        failures.append("missing_heldout_case_ids")
    if not heldout <= declared_heldout:
        failures.append("gold_cases_not_all_in_declared_holdout")
    lookup = {c["case_id"]: c for c in cases}
    if split.get("graph_source_kind") != "train_only":
        failures.append("graph_source_kind_not_train_only")
    if not train:
        failures.append("missing_graph_training_case_ids")
    if train & heldout:
        failures.append("case_id_train_test_overlap")
    if any(c not in lookup for c in train | heldout):
        failures.append("split_contains_unknown_case_ids")
    train_body = {lookup[c]["text"] for c in train if c in lookup}
    test_body = {lookup[c]["text"] for c in heldout if c in lookup}
    if train_body & test_body:
        failures.append("duplicate_body_train_test_overlap")
    groups = split.get("group_id_by_case", {})
    if any(c not in groups for c in train | heldout):
        failures.append("missing_near_template_or_device_group_ids")
    if {groups[c] for c in train if c in groups} & {groups[c] for c in heldout if c in groups}:
        failures.append("source_group_train_test_overlap")
    topic_train = set(split.get("topic_training_case_ids", []))
    if not topic_train:
        failures.append("missing_topic_training_case_ids")
    if topic_train & heldout:
        failures.append("topic_training_includes_evaluation_cases")
    if {lookup[c]["text"] for c in topic_train if c in lookup} & test_body:
        failures.append("topic_training_duplicate_body_test_overlap")
    if {groups[c] for c in topic_train if c in groups} & {groups[c] for c in heldout if c in groups}:
        failures.append("topic_training_source_group_test_overlap")
    for graph_name, file in graph_files.items():
        if split.get("graph_sha256", {}).get(graph_name) != sha(file):
            failures.append(f"{graph_name}_graph_sha256_not_proven")
    # Any input graph may contain held-out case support despite a train-only declaration.
    for name, file in graph_files.items():
        d = json.loads(Path(file).read_text())
        supports = {c for e in d["analysis_edges"] for c in e.get("case_support", [])}
        if supports & heldout:
            failures.append(f"{name}_edge_support_includes_evaluation_cases")
        node_supports = {c for node in d["nodes"] for c in node.get("case_support", [])}
        if node_supports & heldout:
            failures.append(f"{name}_node_or_topic_support_includes_evaluation_cases")
    return {"independent_holdout_verified": not failures, "reasons": sorted(set(failures)),
            "split_manifest": str(split_path), "split_manifest_sha256": sha(split_path)}


def gold_metrics(gold, outputs, path_records, cases, independent):
    by_case = {c["case_id"]: c for c in cases}
    prediction = {r["case_id"]: r for r in outputs}
    path_by_end = {name: defaultdict(set) for name in ["base", "enhanced"]}
    for name, records in path_records.items():
        for p in records:
            path_by_end[name][p["node_ids"][-1]].add(tuple(p["node_ids"]))
    raw, errors = [], []
    for g in gold:
        cid = g.get("case_id")
        if cid not in by_case or cid not in prediction:
            errors.append({"case_id": cid, "reason": "unknown_case_id"})
            continue
        if not g.get("adjudicated") or not g.get("annotator_ids"):
            errors.append({"case_id": cid, "reason": "missing_adjudicated_human_annotation"})
            continue
        if any(not isinstance(g.get(field, []), list) or any(not isinstance(n, str) for n in g.get(field, []))
               for field in ["query_symptom_node_ids", "root_cause_node_ids", "core_node_ids"]):
            errors.append({"case_id": cid, "reason": "node_id fields must be lists of canonical ID strings"})
            continue
        if not isinstance(g.get("propagation_paths", []), list) or any(not isinstance(p, list) or len(p) < 2 or any(not isinstance(n, str) for n in p) for p in g.get("propagation_paths", [])):
            errors.append({"case_id": cid, "reason": "propagation_paths must contain complete ordered lists of >=2 canonical node IDs"})
            continue
        r = {"case_id": cid, "group_id": g.get("source_group_id") or hashlib.sha256(by_case[cid]["text"].encode()).hexdigest()}
        symptoms = g.get("query_symptom_node_ids", [])
        for name in ["base", "enhanced"]:
            result = prediction[cid][name]
            roots = [p["node_id"] for p in result["candidate_roots"]]
            root_gold = g.get("root_cause_node_ids", [])
            if root_gold:
                r[f"{name}_root_top1"] = float(bool(set(roots[:1]) & set(root_gold)))
                r[f"{name}_root_top3"] = float(bool(set(roots[:3]) & set(root_gold)))
                hits = [i + 1 for i, n in enumerate(roots) if n in root_gold]
                r[f"{name}_root_mrr"] = 1 / min(hits) if hits else 0.
            core_gold = g.get("core_node_ids", [])
            if core_gold:
                scores = set_scores([n["node_id"] for n in result["core_nodes"]], core_gold)
                r.update({f"{name}_core_{k}": v for k, v in scores.items()})
            paths_gold = [tuple(p) for p in g.get("propagation_paths", [])]
            if paths_gold:
                inventory = set().union(*(path_by_end[name][s] for s in symptoms)) if symptoms else set()
                returned = {tuple(root.get("best_explanation", {}).get("node_ids", [])) for root in result["candidate_roots"]}
                returned.discard(())
                scores = set_scores(returned, paths_gold)
                r.update({f"{name}_returned_explanation_path_{k}": v for k, v in scores.items()})
                r[f"{name}_candidate_inventory_path_recall"] = set_scores(inventory, paths_gold)["recall"]
                r[f"{name}_path_completeness_rate"] = scores["recall"]
                r[f"{name}_whole_case_chain_complete"] = float(set(paths_gold) <= returned)
            r[f"{name}_query_ms"] = result["query_time_median_ms"]
        raw.append(r)
    summaries = {}
    suffixes = ["root_top1", "root_top3", "root_mrr", "core_precision", "core_recall", "core_f1",
                "returned_explanation_path_precision", "returned_explanation_path_recall", "returned_explanation_path_f1",
                "candidate_inventory_path_recall", "path_completeness_rate", "whole_case_chain_complete", "query_ms"]
    for metric in suffixes:
        valid = [r for r in raw if f"base_{metric}" in r and f"enhanced_{metric}" in r]
        for r in valid:
            r[f"delta_{metric}"] = r[f"enhanced_{metric}"] - r[f"base_{metric}"]
        summaries[metric] = paired_bootstrap(valid, [f"base_{metric}", f"enhanced_{metric}", f"delta_{metric}"])
    available = [summaries[m]["status"] == "computed" for m in ["root_top1", "core_f1", "path_completeness_rate"]]
    overall = "computed" if all(available) else ("computed_partial" if any(available) else "not_evaluable")
    return {"status": overall, "evaluation_scope": "independent_holdout" if independent else "human_gold_retrospective_replay",
            "is_independent_industrial_accuracy": independent, "n_gold_records": len(gold), "n_adjudicated_cases": len(raw),
            "invalid_records": errors, "metrics": summaries,
            "path_metric_definition": "path_completeness_rate and returned_explanation_path P/R/F1 compare the case's actual returned best root explanations with complete ordered human-gold paths",
            "candidate_inventory_path_recall_definition": "separate oracle-style retrieval-availability description against the entire saved strict path collection ending at the symptoms; not a returned diagnostic chain completeness score",
            "core_metric_definition": "top core-k nodes by summed best-root-explanation association scores compared with human core-node sets"}, raw


def icc_absolute_agreement(matrix):
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or min(a.shape) < 2:
        return {"status": "not_evaluable", "reason": "requires >=2 shared raters and >=2 complete rating targets", "ICC_2_1": None}
    n, k = a.shape
    mean = a.mean()
    row, col = a.mean(1), a.mean(0)
    ms_row = k * np.sum((row - mean) ** 2) / (n - 1)
    ms_col = n * np.sum((col - mean) ** 2) / (k - 1)
    residual = a - row[:, None] - col[None, :] + mean
    ms_err = np.sum(residual ** 2) / ((n - 1) * (k - 1))
    denominator = ms_row + (k - 1) * ms_err + k * (ms_col - ms_err) / n
    return {"status": "computed" if denominator else "not_evaluable", "ICC_2_1": float((ms_row - ms_err) / denominator) if denominator else None,
            "n_targets": n, "n_raters": k, "definition": "two-way random-effects absolute-agreement single-rater ICC(2,1)"}


def sis_evaluate(path, cases):
    if not path:
        return {"status": "not_evaluable", "SIS": None, "missing_input": "real expert ratings with case_id, graph, rater_id, S_SC, S_PC, S_CC, S_DA and evidence references",
                "original_weights": SIS_WEIGHTS}, []
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    by_case = {c["case_id"]: c for c in cases}
    targets = defaultdict(dict)
    accepted, errors = [], []
    for row_no, r in enumerate(records, 2):
        try:
            if r.get("graph") not in {"base", "enhanced"} or r.get("case_id") not in by_case or not r.get("rater_id"):
                raise ValueError("unknown graph/case or missing rater_id")
            ratings = {k: float(r[k]) for k in SIS_WEIGHTS}
            if any(not math.isfinite(v) or not 0 <= v <= 10 for v in ratings.values()):
                raise ValueError("all four expert scores must be finite and in [0,10]")
            if any(not r.get("evidence_" + k[2:]) for k in SIS_WEIGHTS):
                raise ValueError("all four scores require a real evidence reference")
            if ratings["S_DA"] > 8 and not r.get("trend_evidence_ref"):
                raise ValueError("paper S_DA >8 requires evidence of trend prediction as well as diagnosis")
            key = (r["case_id"], r["graph"])
            if r["rater_id"] in targets[key]:
                raise ValueError("duplicate case/graph/rater rating")
            value = sum(SIS_WEIGHTS[k] * ratings[k] for k in ratings)
            scored = {**r, **ratings, "SIS": value}
            targets[key][r["rater_id"]] = scored
            accepted.append(scored)
        except (KeyError, ValueError) as e:
            errors.append({"csv_row": row_no, "reason": str(e)})
    paired = []
    for cid in sorted(by_case):
        shared = set(targets[cid, "base"]) & set(targets[cid, "enhanced"])
        if not shared:
            continue
        base = np.mean([targets[cid, "base"][r]["SIS"] for r in sorted(shared)])
        enhanced = np.mean([targets[cid, "enhanced"][r]["SIS"] for r in sorted(shared)])
        paired.append({"case_id": cid, "group_id": hashlib.sha256(by_case[cid]["text"].encode()).hexdigest(),
                       "n_paired_raters": len(shared), "rater_ids": sorted(shared), "base_SIS": float(base),
                       "enhanced_SIS": float(enhanced), "delta_SIS": float(enhanced - base)})
    complete_targets = [v for v in targets.values() if v]
    common_raters = set.intersection(*(set(v) for v in complete_targets)) if complete_targets else set()
    reliability = {"status": "not_evaluable", "reason": "no complete common-rater rating panel", "ICC_2_1": None}
    if len(common_raters) >= 2:
        reliability = icc_absolute_agreement([[v[r]["SIS"] for r in sorted(common_raters)] for v in complete_targets])
        reliability["rater_ids"] = sorted(common_raters)
    return {"status": "computed" if paired else "not_evaluable", "original_weights": SIS_WEIGHTS,
            "rating_scale": [0, 10], "input": str(path), "input_sha256": sha(path),
            "n_valid_rating_rows": len(accepted), "invalid_rows": errors,
            "paired_SIS": paired_bootstrap(paired, ["base_SIS", "enhanced_SIS", "delta_SIS"]),
            "inter_rater_reliability": reliability, "expert_status_and_blinding_externally_verified": False,
            "SIS_is_not_inferred_from_graph_node_edge_or_path_counts": True}, accepted


def structural_evidence_metrics(loaded, path_records):
    """Raw structural descriptions, deliberately distinct from expert SIS scores."""
    measurements = {}
    mode = loaded["enhanced"][1].graph.get("support_mode", "unknown")
    for name, (nodes, graph) in loaded.items():
        mapped, occupied_systems = [], set()
        for nid, node in nodes.items():
            distribution = node.get("system_distribution", [])
            if distribution and sum(distribution) > 0:
                mapped.append(nid)
                occupied_systems.add(int(np.argmax(distribution)))
        mechanism_ids = {n for n, d in nodes.items() if d.get("layer") == "mechanism"}
        mechanism_with_case_support = {n for n in mechanism_ids if nodes[n].get("case_support")}
        complete_structure, cross_system, traceable = [], [], []
        for path in path_records[name]:
            ids = path["node_ids"]
            if nodes[ids[0]]["entity_type"] in PHYSICAL and nodes[ids[-1]]["entity_type"] in SYMPTOM_TYPES and set(ids[1:-1]) & mechanism_ids:
                complete_structure.append(ids)
            systems = {int(np.argmax(nodes[n]["system_distribution"])) for n in ids
                       if nodes[n].get("system_distribution") and sum(nodes[n]["system_distribution"]) > 0}
            if len(systems) > 1:
                cross_system.append(ids)
                links = [graph[u][v] for u, v in zip(ids, ids[1:])]
                provenance = all(e["basis_base_edge_ids"] for e in links)
                common_cases = set.intersection(*(set(e["case_support"]) for e in links)) if links else set()
                if provenance and common_cases:
                    traceable.append(ids)
        measurements[name] = {"n_nodes": len(nodes), "n_nodes_with_weak_system_mapping": len(mapped),
                              "fraction_nodes_with_weak_system_mapping": len(mapped) / len(nodes) if nodes else None,
                              "n_occupied_weak_system_bins": len(occupied_systems), "n_system_bins": 7,
                              "occupied_weak_system_fraction": len(occupied_systems) / 7,
                              "n_mechanism_classes": len(mechanism_ids),
                              "n_mechanism_classes_with_case_support": len(mechanism_with_case_support),
                              "n_strict_candidate_paths": len(path_records[name]),
                              "n_physical_mechanism_symptom_three_segment_structures": len(complete_structure),
                              "physical_mechanism_symptom_structure_fraction": len(complete_structure) / len(path_records[name]) if path_records[name] else None,
                              "n_weak_cross_system_paths": len(cross_system), "n_cross_system_paths_with_base_edge_and_common_case_provenance": len(traceable),
                              "cross_system_provenance_traceable_fraction": len(traceable) / len(cross_system) if cross_system else None}
        if mode == "literal":
            measurements[name]["n_mechanism_classes_with_literal_case_support"] = len(mechanism_with_case_support)
            measurements[name]["n_cross_system_paths_with_base_edge_and_common_literal_case_provenance"] = len(traceable)
    return {"status": "computed", "measurements": measurements,
            "interpretation": "structural descriptive metrics only; none are expert SIS sub-scores or independent diagnostic accuracy",
            "system_mapping": "existing topic keyword mapping and argmax, not engineering system gold",
            "complete_structure": "physical-type source, intermediate explicit mechanism-class node, symptom/consequence endpoint among existing strict candidate paths",
            "support_mode": mode,
            "traceability": "every link references base edge IDs and all links share at least one declared model occurrence case ID; this does not verify causal sequencing" if mode == "model_occurrences" else "every link references base edge IDs and all links share at least one declared case-support ID; this does not verify sentence-level causality",
            "semantic_mechanism_coverage_against_gold": None,
            "expert_S_SC_S_PC_S_CC_S_DA_inferred": False}


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cases = jsonl(args.cases)
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("case_id must be unique")
    graph_files = {n: Path(args.graph_dir) / f"{n}_graph.json" for n in ["base", "enhanced"]}
    loaded = {n: load_graph(p) for n, p in graph_files.items()}
    source_scope = graph_source_scope(args.graph_dir, graph_files, loaded)
    query_case_ids = {c["case_id"] for c in cases}
    source_scope["case_support_ids_outside_query_corpus"] = sorted(set(source_scope["actual_input_case_support_ids"]) - query_case_ids)
    roots = {n for n, d in loaded["base"][0].items() if d["entity_type"] in ROOT_TYPES}
    if not roots <= set(loaded["enhanced"][0]):
        raise ValueError("The two methods must share the basic physical/fault root candidate universe")
    gold = jsonl(args.gold) if args.gold else []
    if len({g["case_id"] for g in gold}) != len(gold):
        raise ValueError("human-gold JSONL requires one adjudicated object per case_id")
    gold_by_case = {g["case_id"]: g for g in gold}
    # Warm up both methods equally; no result cache is used in timing measurements.
    for name in ["base", "enhanced"]:
        infer_query([], *loaded[name], roots, args.core_k)
    outputs, paired, query_rows = [], [], []
    start = time.perf_counter()
    with (output / "case_queries_and_inference.jsonl").open("w", encoding="utf-8") as stream:
        for case in cases:
            query = symptom_query(case, loaded["base"][0])
            if case["case_id"] in gold_by_case:
                query = {"query_symptom_node_ids": gold_by_case[case["case_id"]].get("query_symptom_node_ids", []),
                         "query_text": gold_by_case[case["case_id"]].get("query_text", ""),
                         "query_extraction": "external human-gold symptom IDs", "presentation_boundary_verified_by_human": bool(gold_by_case[case["case_id"]].get("adjudicated"))}
            result = {"case_id": case["case_id"], "case_body_sha256": hashlib.sha256(case["text"].encode()).hexdigest(), **query}
            elapsed = defaultdict(list)
            for repeat in range(args.repeats):
                # Alternate order to avoid always favoring the second warmed method.
                order = ["base", "enhanced"] if repeat % 2 == 0 else ["enhanced", "base"]
                for name in order:
                    tic = time.perf_counter_ns()
                    inferred = infer_query(query["query_symptom_node_ids"], *loaded[name], roots, args.core_k)
                    elapsed[name].append((time.perf_counter_ns() - tic) / 1e6)
                    if repeat == 0:
                        result[name] = inferred
            for name in ["base", "enhanced"]:
                result[name]["query_times_ms"] = elapsed[name]
                result[name]["query_time_median_ms"] = float(np.median(elapsed[name]))
                result[name]["query_timing_scope"] = "in-memory constrained graph reasoning + candidate ranking + explanation materialization; excludes disk loading and human diagnosis"
            # Traceable full ranked roots/core nodes and all explanation links are retained.
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            outputs.append(result)
            p = {"case_id": case["case_id"], "group_id": result["case_body_sha256"]}
            for name in ["base", "enhanced"]:
                p[f"{name}_root_coverage"] = float(result[name]["query_has_candidate_root"])
                p[f"{name}_query_ms"] = result[name]["query_time_median_ms"]
                p[f"{name}_n_candidate_roots"] = result[name]["n_root_candidates"]
            for field in ["root_coverage", "query_ms", "n_candidate_roots"]:
                p[f"delta_{field}"] = p[f"enhanced_{field}"] - p[f"base_{field}"]
            paired.append(p)
            query_rows.append({"case_id": case["case_id"], "n_symptoms": len(query["query_symptom_node_ids"]),
                               "symptom_node_ids": query["query_symptom_node_ids"], **{k: v for k, v in p.items() if k != "group_id"}})
    csv_write(output / "paired_query_metrics.csv", query_rows, list(query_rows[0]) if query_rows else ["case_id"])
    fields = [f"{name}_{metric}" for metric in ["root_coverage", "query_ms", "n_candidate_roots"] for name in ["base", "enhanced", "delta"]]
    replay = paired_bootstrap(paired, fields)
    conditional = [p for p, result in zip(paired, outputs) if result["query_symptom_node_ids"]]
    replay["presentation_symptom_queries_only"] = paired_bootstrap(conditional, fields)
    replay.update({"evaluation_scope": source_scope["evaluation_scope"], "source_scope": source_scope, "n_numbered_cases": len(cases),
                   "n_unique_bodies": len({c["text"] for c in cases}), "n_cases_with_presentation_symptoms": sum(bool(o["query_symptom_node_ids"]) for o in outputs),
                   "n_queries_without_symptoms": sum(not o["query_symptom_node_ids"] for o in outputs),
                   "root_coverage_is_accuracy": False, "query_runtime_is_engineer_root_cause_localization_time": False,
                   "source_leakage_warning": source_scope["source_leakage_warning"],
                   "runtime_seconds": time.perf_counter() - start})
    audit = independent_audit(args.split_manifest, gold, cases, graph_files)
    path_records = {}
    for name in ["base", "enhanced"]:
        path_file = Path(args.graph_dir) / f"{name}_propagation_candidate_paths.jsonl"
        path_records[name] = jsonl(path_file) if path_file.exists() else []
    structural = structural_evidence_metrics(loaded, path_records)
    dump(output / "structural_descriptive_metrics.json", structural)
    if gold:
        gold_summary, gold_raw = gold_metrics(gold, outputs, path_records, cases, audit["independent_holdout_verified"])
    else:
        gold_summary, gold_raw = {"status": "not_evaluable", "root_cause_accuracy": None, "core_node_identification_accuracy": None,
                                 "path_completeness_rate": None, "missing_input": "adjudicated case-level human root causes, core nodes and complete ordered propagation paths"}, []
    write_jsonl(output / "human_gold_paired_case_metrics.jsonl", gold_raw)
    sis_summary, sis_raw = sis_evaluate(args.expert_scores, cases)
    write_jsonl(output / "expert_scores_calculated.jsonl", sis_raw)
    template_fields = ["case_id", "graph", "rater_id", *SIS_WEIGHTS, "evidence_SC", "evidence_PC", "evidence_CC", "evidence_DA", "trend_evidence_ref"]
    csv_write(output / "expert_scores_input_template.csv", [], template_fields)
    dump(output / "human_gold_schema.json", {"format": "JSONL, one object per case; this is a schema description, not measured gold", "required": {
        "case_id": "existing case ID", "query_symptom_node_ids": "canonical symptom IDs; the same query is used for both graphs",
        "root_cause_node_ids": "human-adjudicated root IDs", "core_node_ids": "human-adjudicated key IDs",
        "propagation_paths": "list of complete ordered canonical-node paths", "adjudicated": True,
        "annotator_ids": "list of real annotator identifiers", "source_group_id": "duplicate/near-template/device grouping identifier"},
        "independent_holdout_additional_input": "--split-manifest with train-only graph SHA256, graph_training_case_ids, topic_training_case_ids, heldout_case_ids and group_id_by_case"})
    dump(output / "split_manifest_schema.json", {"format": "JSON object; schema description only, not a claim that the input graph meets this split",
        "required_fields": {"graph_source_kind": "train_only", "graph_training_case_ids": "nonempty list of graph-training case IDs",
            "topic_training_case_ids": "nonempty list of LDA fitting case IDs", "heldout_case_ids": "held-out test case IDs",
            "group_id_by_case": "mapping covering every train/test case; duplicate/near-template/device groups kept together",
            "graph_sha256": {"base": "SHA256 of the actual base_graph.json", "enhanced": "SHA256 of enhanced_graph.json"}},
        "checks": "case IDs, exact bodies, source groups, node/edge grounding and topic training must not overlap held-out cases; graph bytes must match the manifest"})
    config = {"graph_inputs": {n: {"path": str(p), "sha256": sha(p)} for n, p in graph_files.items()},
              "source_scope": source_scope,
              "implementation": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__), "python": platform.python_version(), "platform": platform.platform()},
              "cases": {"path": str(args.cases), "sha256": sha(args.cases)}, "repeats": args.repeats, "core_k": args.core_k,
              "root_candidate_universe": "shared base graph nodes with physical, cause or state types", "n_root_candidates_in_universe": len(roots),
              "root_reasoning": "constrained reverse Dijkstra maximum-product association path with optional first physical context hop",
              "path_search_limit": "one maximum-product suffix per root/symptom; no exhaustive-global-path claim", "query_result_cache": False,
              "score_interpretation": "uncalibrated association; not causal or root-cause probability", "device": "cpu"}
    dump(output / "run_config.json", config)
    summary = {"status": "completed_computational_replay", "retrospective_query_comparison": replay,
               "source_scope": source_scope,
               "human_gold_diagnosis_evaluation": gold_summary, "independence_audit": audit, "expert_SIS": sis_summary,
               "structural_descriptive_metrics": structural,
               "paper_validation_claims": {"path_completeness": gold_summary.get("metrics", {}).get("path_completeness_rate", {}).get("status", "not_evaluable"),
                    "core_node_identification_accuracy": gold_summary.get("metrics", {}).get("core_f1", {}).get("status", "not_evaluable"),
                    "root_cause_accuracy": gold_summary.get("metrics", {}).get("root_top1", {}).get("status", "not_evaluable"),
                    "machine_query_runtime": "measured", "engineer_localization_time": "not_evaluable_without_timed_engineer_trials", "SIS": sis_summary["status"]}}
    dump(output / "analysis_summary.json", summary)
    print(json.dumps({"output": str(output), "status": summary["status"], "n_cases": len(cases),
                      "n_cases_with_symptoms": replay["n_cases_with_presentation_symptoms"], "gold_status": gold_summary["status"], "SIS_status": sis_summary["status"]}, ensure_ascii=False))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-dir", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gold")
    p.add_argument("--expert-scores")
    p.add_argument("--split-manifest")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--core-k", type=int, default=5)
    args = p.parse_args()
    if args.repeats < 1 or args.core_k < 1:
        p.error("--repeats and --core-k must be positive")
    run(args)


if __name__ == "__main__":
    main()
