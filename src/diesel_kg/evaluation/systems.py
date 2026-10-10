#!/usr/bin/env python3
"""Validate Louvain/weak topic mapping against external physical-system gold.

No gold is inferred from the same LDA dictionary. Multi-label expert nodes have
no implicit primary class: one explicit is_primary label is required for the
single-class evaluation; a separate set evaluation retains all gold labels.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

PHYSICAL_TYPES = {"故障部位", "部件", "组件", "系统"}
GOLD_FIELDS = ["node_id", "system_id", "adjudicated", "is_primary", "evidence_ref"]


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_write(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(r[k], ensure_ascii=False) if isinstance(r.get(k), (list, dict)) else r.get(k, "") for k in fields})


def true(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y", "是"}


def load_gold(records, nodes):
    grouped, invalid, duplicate_rows = defaultdict(dict), [], 0
    for row, record in enumerate(records, 2):
        node_id, system_id = str(record.get("node_id") or "").strip(), str(record.get("system_id") or "").strip()
        if node_id not in nodes:
            invalid.append({"csv_row": row, "node_id": node_id, "reason": "node_id missing from the supplied graph"})
            continue
        if not system_id or not true(record.get("adjudicated")) or not str(record.get("evidence_ref") or "").strip():
            invalid.append({"csv_row": row, "node_id": node_id, "reason": "requires nonempty system_id, adjudicated=true and a real evidence_ref"})
            continue
        label = {"system_id": system_id, "is_primary": true(record.get("is_primary")),
                 "evidence_refs": [str(record["evidence_ref"]).strip()]}
        if system_id in grouped[node_id]:
            duplicate_rows += 1
            grouped[node_id][system_id]["evidence_refs"] = sorted(set(grouped[node_id][system_id]["evidence_refs"] + label["evidence_refs"]))
            grouped[node_id][system_id]["is_primary"] |= label["is_primary"]
        else:
            grouped[node_id][system_id] = label
    gold, multiple_primary = {}, []
    for node_id, labels in grouped.items():
        primary = [k for k, v in labels.items() if v["is_primary"]]
        if len(primary) > 1:
            multiple_primary.append(node_id)
            chosen, primary_status = None, "invalid_multiple_primary_labels_single_class_evaluation_excluded"
        elif len(labels) == 1:
            chosen, primary_status = next(iter(labels)), "single_gold_label"
        elif len(primary) == 1:
            chosen, primary_status = primary[0], "explicit_expert_primary_label"
        else:
            chosen, primary_status = None, "multi_label_without_primary_single_class_evaluation_excluded"
        gold[node_id] = {"labels": sorted(labels), "primary_label": chosen, "primary_status": primary_status,
                         "label_evidence": list(labels.values())}
    return gold, {"n_input_rows": len(records), "n_invalid_rows": len(invalid), "invalid_rows": invalid,
                  "n_duplicate_node_system_rows_collapsed": duplicate_rows,
                  "n_adjudicated_graph_nodes": len(gold), "n_multilabel_nodes": sum(len(g["labels"]) > 1 for g in gold.values()),
                  "n_single_or_explicit_primary_nodes": sum(g["primary_label"] is not None for g in gold.values()),
                  "multiple_primary_node_ids": sorted(multiple_primary)}


def weak_prediction(node, systems):
    a = np.array(node.get("system_distribution", []), dtype=float)
    if a.shape != (len(systems),) or not np.isfinite(a).all() or (a < 0).any():
        return None, "invalid_or_missing_system_vector"
    if a.sum() <= 0:
        return None, "zero_support_vector_unassigned"
    highest = np.flatnonzero(np.isclose(a, a.max(), rtol=1e-10, atol=1e-12))
    if len(highest) != 1:
        return None, "tied_maximum_unassigned"
    return systems[int(highest[0])], "unique_argmax_of_weak_topic_system_vector"


def clustering_metrics(node_ids, gold, assignments):
    eligible = [n for n in node_ids if gold[n]["primary_label"] is not None]
    assigned = [n for n in eligible if n in assignments]
    counts = {"n_primary_gold_nodes": len(eligible), "n_community_assigned_primary_nodes": len(assigned),
              "n_community_unassigned_primary_nodes": len(eligible) - len(assigned),
              "community_assignment_coverage": len(assigned) / len(eligible) if eligible else None}
    if len(assigned) < 2:
        return {"status": "not_evaluable", **counts, "ARI": None, "NMI": None, "purity": None,
                "reason": "requires at least two externally labeled community-assigned nodes"}
    target = [gold[n]["primary_label"] for n in assigned]
    predicted = [str(assignments[n]) for n in assigned]
    table = defaultdict(Counter)
    for cluster, label in zip(predicted, target):
        table[cluster][label] += 1
    purity = sum(max(row.values()) for row in table.values()) / len(assigned)
    return {"status": "computed", **counts, "ARI": float(adjusted_rand_score(target, predicted)),
            "NMI": float(normalized_mutual_info_score(target, predicted, average_method="arithmetic")), "purity": purity,
            "n_observed_gold_systems": len(set(target)), "n_observed_Louvain_communities": len(table),
            "degenerate_gold_partition": len(set(target)) < 2,
            "purity_warning": "many small communities can yield high purity; purity is not subsystem classification accuracy",
            "community_unassigned_nodes_are_not_made_into_a_single_fake_community": True}


def mapping_metrics(node_ids, gold, predictions, systems):
    primary = [n for n in node_ids if gold[n]["primary_label"] is not None]
    eligible = [n for n in primary if gold[n]["primary_label"] in systems]
    covered = [n for n in eligible if predictions[n] is not None]
    correct = sum(predictions[n] == gold[n]["primary_label"] for n in covered)
    return {"status": "computed" if eligible else "not_evaluable", "n_primary_gold_nodes": len(primary),
            "n_gold_nodes_in_weak_seven_system_taxonomy": len(eligible), "n_gold_nodes_outside_weak_taxonomy": len(primary) - len(eligible),
            "n_mapping_assigned": len(covered), "n_mapping_unassigned": len(eligible) - len(covered), "n_correct": correct,
            "classification_coverage": len(covered) / len(eligible) if eligible else None,
            "conditional_accuracy_on_assigned_nodes": correct / len(covered) if covered else None,
            "accuracy_abstentions_count_as_failure": correct / len(eligible) if eligible else None,
            "gold_class_is_never_topic_argmax": True}


def multi_label_metrics(node_ids, gold, predictions, systems):
    multi = [n for n in node_ids if len(gold[n]["labels"]) > 1]
    eligible = [n for n in multi if set(gold[n]["labels"]) <= set(systems)]
    assigned = [n for n in eligible if predictions[n] is not None]
    hits = sum(predictions[n] in gold[n]["labels"] for n in assigned)
    jaccard = [1 / len(gold[n]["labels"]) if predictions[n] in gold[n]["labels"] else 0. for n in assigned]
    return {"status": "computed" if eligible else "not_evaluable", "n_multilabel_gold_nodes": len(multi),
            "n_multilabel_nodes_in_weak_taxonomy": len(eligible), "n_excluded_outside_weak_taxonomy": len(multi) - len(eligible),
            "n_mapping_assigned": len(assigned), "mapping_coverage": len(assigned) / len(eligible) if eligible else None,
            "any_gold_label_hit_on_assigned": hits / len(assigned) if assigned else None,
            "micro_precision_assigned_singleton_predictions": hits / len(assigned) if assigned else None,
            "micro_recall_all_gold_labels_including_abstentions": hits / sum(len(gold[n]["labels"]) for n in eligible) if eligible else None,
            "mean_set_Jaccard_on_assigned": float(np.mean(jaccard)) if jaccard else None,
            "protocol": "the frozen weak model predicts a singleton; all expert labels are retained, never silently reduced by argmax"}


def evaluate(nodes, assignments, systems, gold_records):
    gold, audit = load_gold(gold_records, nodes)
    predictions, reasons = {}, {}
    for n, node in nodes.items():
        predictions[n], reasons[n] = weak_prediction(node, systems)
    subsets = {"all_gold_nodes": sorted(gold), "physical_entity_types": sorted(n for n in gold if nodes[n]["entity_type"] in PHYSICAL_TYPES)}
    for kind in sorted({d["entity_type"] for d in nodes.values()}):
        subsets["entity_type:" + kind] = sorted(n for n in gold if nodes[n]["entity_type"] == kind)
    for layer in sorted({d.get("layer", "unspecified") for d in nodes.values()}):
        subsets["layer:" + layer] = sorted(n for n in gold if nodes[n].get("layer", "unspecified") == layer)
    metrics = {subset: {"Louvain_vs_physical_gold": clustering_metrics(ids, gold, assignments),
                        "weak_mapping_vs_physical_gold": mapping_metrics(ids, gold, predictions, systems),
                        "multilabel_weak_mapping": multi_label_metrics(ids, gold, predictions, systems)} for subset, ids in subsets.items()}
    type_coverage = []
    for scope, values, field in [("entity_type", sorted({d["entity_type"] for d in nodes.values()}), "entity_type"),
                                ("layer", sorted({d.get("layer", "unspecified") for d in nodes.values()}), "layer")]:
        for value in values:
            ids = {n for n in nodes if nodes[n].get(field, "unspecified") == value}
            annotated = ids & set(gold)
            type_coverage.append({"scope": scope, "value": value, "n_graph_nodes": len(ids), "n_adjudicated_gold_nodes": len(annotated),
                                  "gold_coverage": len(annotated) / len(ids) if ids else None,
                                  "n_single_or_primary_gold_nodes": sum(gold[n]["primary_label"] is not None for n in annotated),
                                  "n_multilabel_gold_nodes": sum(len(gold[n]["labels"]) > 1 for n in annotated),
                                  "n_community_assigned_gold_nodes": sum(n in assignments for n in annotated),
                                  "n_weak_mapping_assigned_gold_nodes": sum(predictions[n] is not None for n in annotated)})
    comparison = [{"node_id": n, "node_name": nodes[n]["name"], "entity_type": nodes[n]["entity_type"], "layer": nodes[n].get("layer", "unspecified"),
                   "gold_labels": gold[n]["labels"], "primary_gold_label": gold[n]["primary_label"], "gold_primary_status": gold[n]["primary_status"],
                   "community_id": assignments.get(n), "community_assigned": n in assignments, "weak_predicted_system": predictions[n],
                   "weak_mapping_status": reasons[n], "evidence": gold[n]["label_evidence"]} for n in sorted(gold)]
    confusion = Counter((str(assignments[n]), gold[n]["primary_label"]) for n in gold if n in assignments and gold[n]["primary_label"] is not None)
    confusion_rows = [{"community_id": c, "physical_gold_system_id": label, "n_nodes": count} for (c, label), count in sorted(confusion.items())]
    computed = any(value["status"] == "computed" for group in metrics.values() for value in group.values())
    summary = {"status": "computed" if computed else "not_evaluable", "gold_audit": audit,
               "n_graph_nodes": len(nodes), "n_graph_community_assigned": sum(n in assignments for n in nodes),
               "n_graph_community_unassigned": sum(n not in assignments for n in nodes),
               "n_graph_weak_mapping_assigned": sum(predictions[n] is not None for n in nodes),
               "n_graph_weak_mapping_unassigned": sum(predictions[n] is None for n in nodes),
               "gold_node_coverage": len(gold) / len(nodes) if nodes else None,
               "metrics_by_subset": metrics, "coverage_by_node_type_and_layer": type_coverage,
               "weak_system_taxonomy": systems, "gold_inferred_by_this_program_from_LDA_dictionary": False,
               "annotation_source": "external supplied CSV with adjudicated labels and evidence references",
               "external_annotation_independence_and_expert_status_verified": False,
               "multilabel_protocol": "multiple expert labels preserved; single-class ARI/NMI/purity need exactly one explicit primary or a sole label",
               "claim_limit": "computed consistency on covered, externally labeled nodes does not establish industrial diagnosis or seven physical systems without adequate gold coverage"}
    return summary, comparison, confusion_rows


def run(args):
    output, graph = Path(args.output), Path(args.graph_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = {"graph": graph / "enhanced_graph.json", "communities": graph / "automatic_communities.json", "systems": graph / "system_coupling.json"}
    nodes = {n["id"]: n for n in json.loads(files["graph"].read_text())["nodes"]}
    assignments = json.loads(files["communities"].read_text())["node_community"]
    if not set(assignments) <= set(nodes):
        raise ValueError("community assignments contain unknown graph node IDs")
    systems = json.loads(files["systems"].read_text())["system_order"]
    records = []
    if args.labels:
        with Path(args.labels).open(encoding="utf-8-sig", newline="") as f:
            records = list(csv.DictReader(f))
    summary, compared, confusion = evaluate(nodes, assignments, systems, records)
    if not args.labels:
        summary.update(status="not_evaluable", missing_input="real independent adjudicated physical-system labels with node_id/system_id/evidence_ref",
                       Louvain_physical_consistency_confirmed=False, annotation_source="no external physical-system gold supplied")
    summary["input_provenance"] = {name: {"path": str(path), "sha256": sha(path)} for name, path in files.items()}
    summary["labels_provenance"] = {"path": str(args.labels), "sha256": sha(args.labels)} if args.labels else None
    csv_write(output / "physical_system_gold_template.csv", [], GOLD_FIELDS)
    annotation_fields = ["node_id", "node_name", "entity_type", "layer", "system_id", "adjudicated", "is_primary", "evidence_ref"]
    csv_write(output / "node_annotation_inventory.csv", [{"node_id": n, "node_name": d["name"], "entity_type": d["entity_type"],
              "layer": d.get("layer", "unspecified")} for n, d in sorted(nodes.items())], annotation_fields)
    csv_write(output / "adjudicated_node_comparison.csv", compared, list(compared[0]) if compared else ["node_id"])
    csv_write(output / "community_physical_gold_contingency.csv", confusion, ["community_id", "physical_gold_system_id", "n_nodes"])
    dump(output / "input_schema.json", {"CSV_required": ["node_id", "system_id", "adjudicated", "evidence_ref"],
        "optional": ["is_primary"], "independence": "expert labels should be assigned from external engineering evidence before inspecting LDA mapping or Louvain assignments; inventory reveals neither prediction",
        "multi_label": "one row per node/system pair; if a node has multiple labels, exactly one is_primary=true is needed for primary-class metrics; no primary means only multilabel evaluation",
        "system_id": "any expert physical-system ID for clustering metrics; use a canonical weak-system ID below for directly comparable mapping classification",
        "weak_system_ids": systems, "unknown_weak_taxonomy_labels": "excluded from weak mapping scoring and explicitly counted; retained for community consistency",
        "adjudicated": "true/1/yes after real adjudication; default blank records are not scored", "no_gold_behavior": "not_evaluable; ARI/NMI/purity/accuracy are null, not estimated from the same LDA dictionary"})
    dump(output / "analysis_summary.json", summary)
    dump(output / "run_config.json", {"implementation_path": str(Path(__file__).resolve()), "implementation_sha256": sha(__file__),
         "graph_dir": str(graph.resolve()), "labels": str(args.labels) if args.labels else None,
         "independent_gold_required": True, "no_expert_primary_class_argmax": True})
    print(json.dumps({"output": str(output), "status": summary["status"], "n_graph_nodes": len(nodes), "n_gold_nodes": summary["gold_audit"]["n_adjudicated_graph_nodes"]}, ensure_ascii=False))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--labels")
    run(p.parse_args())


if __name__ == "__main__":
    main()
