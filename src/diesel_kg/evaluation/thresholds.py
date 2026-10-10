"""CPU postprocessing of complete saved RE argmax predictions.

Threshold sensitivity describes model candidates, not relation-existence,
precision, diagnosis or causal validation. The paper did not specify the added
0.8 confidence gate. The main 0.8 run is preserved; this independent audit never
invokes a model, changes argmax classes, or replaces its outputs.
"""
from __future__ import annotations
from diesel_kg.paths import PROJECT_ROOT, resolve_script

import argparse
from array import array
import ast
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from diesel_kg.data.neo4j import stable_id

HERE = PROJECT_ROOT
PREDICTIONS = "predictions_ij_relation_probability_token_window"
GATE_ORDER = ["predicted_no_relation", "short_entity", "uncalibrated_confidence_filter",
              "type_qualification", "canonical_self_loop"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip(): yield json.loads(line)


def qualification_rules():
    """Compile only existing rule declarations; no Torch module is imported."""
    path = resolve_script('diesel_kg.models.loading', HERE)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {"FAULT_TYPES", "DEFAULT_ONTOLOGY", "relation_allowed"}
    selected = [node for node in tree.body
                if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in wanted for t in node.targets))
                or (isinstance(node, ast.FunctionDef) and node.name == "relation_allowed")]
    if len(selected) != 3: raise ValueError("Unexpected ontology declarations")
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["DEFAULT_ONTOLOGY"], namespace["relation_allowed"], {
        "path": str(path), "sha256": sha(path),
        "selected_rule_sha256": hashlib.sha256("\n".join(ast.get_source_segment(source, n) for n in selected).encode()).hexdigest()}


def rejection_reason(head, tail, relation, probability, threshold, minimum, ontology, allowed):
    """Same ordered, first-failure gates as author_occurrence_re.py."""
    if relation == "no_relation": return "predicted_no_relation"
    if min(len(head["text"]), len(tail["text"])) < minimum: return "short_entity"
    if probability < threshold: return "uncalibrated_confidence_filter"
    if not allowed(ontology, relation, head["type"], tail["type"]): return "type_qualification"
    if (head["type"], head["text"]) == (tail["type"], tail["text"]): return "canonical_self_loop"
    return None


def quantiles(values):
    levels = [0, .01, .05, .25, .5, .75, .9, .95, .99, 1]
    if not values: return {"count": 0, "quantiles": dict.fromkeys(map(str, levels), None), "mean": None}
    view = np.frombuffer(values, dtype=np.float64)
    return {"count": len(values), "quantiles": {str(q): float(v) for q, v in zip(levels, np.quantile(view, levels))},
            "mean": float(view.mean()), "method": "numpy linear interpolation over all saved max probabilities"}


def validate_entities(row):
    seen = set()
    for entity in row["entities"]:
        start, end = entity.get("start"), entity.get("end")
        if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int)
                or not 0 <= start < end <= len(row["text"]) or row["text"][start:end] != entity.get("text")):
            raise ValueError(f"Invalid saved occurrence offsets/text in {row['case_id']}")
        if not entity.get("type") or not entity.get("occurrence_id"): raise ValueError("Missing occurrence type/ID")
        if entity["occurrence_id"] in seen: raise ValueError("Repeated occurrence ID in a case")
        seen.add(entity["occurrence_id"])


def run(args):
    started = time.perf_counter()
    thresholds = sorted(set(map(float, args.thresholds)))
    if not thresholds or any(not math.isfinite(t) or not 0 <= t <= 1 for t in thresholds): raise ValueError("Thresholds must lie in [0,1]")
    if args.min_entity_length < 1: raise ValueError("Minimum entity length must be positive")
    if args.export_argmax_graph and 0 not in thresholds: raise ValueError("Argmax graph export requires threshold 0")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    input_paths = {"raw_predictions": Path(args.raw_predictions), "entities": Path(args.entities), "rel2id": Path(args.rel2id)}
    before_hashes = {name: {"path": str(path.resolve()), "sha256": sha(path)} for name, path in input_paths.items()}
    if output.resolve() == input_paths["raw_predictions"].parent.resolve(): raise ValueError("Use an independent output directory")
    source_summary_path = input_paths["raw_predictions"].parent / "run_summary.json"
    source_summary = None
    if source_summary_path.is_file():
        source_summary = json.loads(source_summary_path.read_text())
        if not source_summary.get("status", "").startswith("completed"): raise ValueError("Source RE run is not completed")
        input_paths["source_RE_run_summary"] = source_summary_path
        before_hashes["source_RE_run_summary"] = {"path": str(source_summary_path.resolve()), "sha256": sha(source_summary_path)}
        expected_entities = source_summary.get("input_sha256", {}).get(str(input_paths["entities"]))
        if expected_entities is None:
            # Exact copies may live in the audit's frozen input directory.
            # The RE producer records one JSONL input: its NER occurrences.
            recorded_jsonl_inputs = [digest for name, digest in source_summary.get("input_sha256", {}).items()
                                     if Path(name).suffix == ".jsonl"]
            if len(recorded_jsonl_inputs) == 1: expected_entities = recorded_jsonl_inputs[0]
        if expected_entities is not None and expected_entities != before_hashes["entities"]["sha256"]:
            raise ValueError("NER entities do not match source RE input SHA")
    rel2id = json.loads(input_paths["rel2id"].read_text())
    if any(isinstance(v, bool) or not isinstance(v, int) for v in rel2id.values()) or sorted(rel2id.values()) != list(range(len(rel2id))):
        raise ValueError("rel2id must contain unique contiguous integer IDs")
    id2rel = {v: k for k, v in rel2id.items()}
    rows = list(read_jsonl(input_paths["entities"]))
    cases = {row["case_id"]: row for row in rows}
    if len(cases) != len(rows): raise ValueError("Repeated case IDs in entities")
    for row in rows: validate_entities(row)
    ontology, allowed, rule_source = qualification_rules()
    exists_head = "no_relation" in rel2id
    totals = Counter(processed_cases=0, ordered_pairs=0, classified_pairs=0, unclassified_pairs=0,
                     mentions=sum(len(r["entities"]) for r in rows),
                     one_character_mentions=sum(len(e["text"]) == 1 for r in rows for e in r["entities"]))
    states = {t: {"retained_pairs": 0, "retained_cases": set(), "retained_bodies": set(), "typed_edges": set(),
                  "relation_counts": Counter(), "rejection_counts": Counter()} for t in thresholds}
    probability_groups = {name: array("d") for name in ["all_classified_pairs", "after_no_relation_and_min_length", "after_all_nonconfidence_gates"]}
    by_relation = {relation: array("d") for relation in rel2id}
    raw_relations, independent_failures, skipped_reasons = Counter(), Counter(), Counter()
    seen_cases, nodes, edges = set(), {}, {}
    case_records = []
    triples_path = output / "threshold0_qualified_triples.jsonl"
    triples = triples_path.open("w", encoding="utf-8") if args.export_argmax_graph else None
    try:
        with gzip.open(input_paths["raw_predictions"], "rt", encoding="utf-8") as raw_handle, \
             gzip.open(output / "pair_filter_audit.jsonl.gz", "wt", encoding="utf-8") as audit:
            for line in raw_handle:
                if not line.strip(): continue
                raw = json.loads(line); cid = raw["case_id"]
                if cid in seen_cases or cid not in cases: raise ValueError("Repeated/unknown case ID in raw predictions")
                seen_cases.add(cid); row = cases[cid]; entities = row["entities"]
                if raw["entities"] != entities: raise ValueError(f"Raw and NER occurrence records differ: {cid}")
                unclassified = raw.get("unclassified_reasons", {})
                if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in unclassified.values()): raise ValueError("Invalid unclassified count")
                predicted = raw[PREDICTIONS]; ordered = len(entities) * (len(entities) - 1)
                if len(predicted) + sum(unclassified.values()) != ordered: raise ValueError(f"Incomplete pair accounting: {cid}")
                totals.update(processed_cases=1, ordered_pairs=ordered, classified_pairs=len(predicted), unclassified_pairs=sum(unclassified.values()))
                skipped_reasons.update(unclassified)
                case_counts = {t: 0 for t in thresholds}; case_edges = {t: set() for t in thresholds}
                case_rejects = {t: Counter() for t in thresholds}; seen_pairs = set(); audit_pairs = []
                for prediction in predicted:
                    if len(prediction) != 6: raise ValueError("Expected six saved argmax/window fields")
                    i, j, label, probability, left, right = prediction
                    if any(isinstance(v, bool) or not isinstance(v, int) for v in [i, j, label, left, right]): raise ValueError("Pair indices/class/window must be integers")
                    if i == j or not 0 <= i < len(entities) or not 0 <= j < len(entities) or (i, j) in seen_pairs: raise ValueError("Invalid/duplicate ordered occurrence pair")
                    if label not in id2rel or not isinstance(probability, (int, float)) or isinstance(probability, bool) or not math.isfinite(probability) or not 0 <= probability <= 1 or not 0 <= left < right:
                        raise ValueError("Invalid saved probability, label, or token window")
                    seen_pairs.add((i, j)); h, t = entities[i], entities[j]; relation = id2rel[label]
                    raw_relations[relation] += 1; probability_groups["all_classified_pairs"].append(probability); by_relation[relation].append(probability)
                    one_char_pair = len(h["text"]) == 1 or len(t["text"]) == 1
                    totals["one_character_endpoint_classified_pairs"] += one_char_pair
                    failures = {
                        "predicted_no_relation": relation == "no_relation",
                        "short_entity": min(len(h["text"]), len(t["text"])) < args.min_entity_length,
                        "type_qualification": not allowed(ontology, relation, h["type"], t["type"]),
                        "canonical_self_loop": (h["type"], h["text"]) == (t["type"], t["text"])}
                    independent_failures.update({name: 1 for name, failed in failures.items() if failed})
                    if not failures["predicted_no_relation"] and not failures["short_entity"]:
                        probability_groups["after_no_relation_and_min_length"].append(probability)
                    if not any(failures.values()): probability_groups["after_all_nonconfidence_gates"].append(probability)
                    key = (h["type"], h["text"], relation, t["type"], t["text"])
                    reasons = []
                    for threshold in thresholds:
                        reason = rejection_reason(h, t, relation, probability, threshold, args.min_entity_length, ontology, allowed)
                        reasons.append(reason or "accepted")
                        state = states[threshold]
                        if reason:
                            state["rejection_counts"][reason] += 1; case_rejects[threshold][reason] += 1
                            continue
                        state["retained_pairs"] += 1; state["relation_counts"][relation] += 1
                        state["typed_edges"].add(key); case_edges[threshold].add(key); case_counts[threshold] += 1
                        if threshold == 0 and triples is not None:
                            lo, hi = min(h["start"], t["start"]), max(h["end"], t["end"])
                            triple = {"case_id": cid, "head": h, "tail": t, "relation": relation, "probability": probability,
                                      "token_window": [left, right], "text_evidence": row["text"][lo:hi],
                                      "text_evidence_kind": "joint_character_mention_span_from_saved_NER_offsets",
                                      "relationship_existence_modeled": exists_head, "causal_gold": False,
                                      "candidate_protocol": "threshold0_argmax_sensitivity_with_other_default_gates"}
                            triples.write(json.dumps(triple, ensure_ascii=False, allow_nan=False) + "\n")
                            pair = []
                            for e in [h, t]:
                                nid = stable_id("n_", [e["type"], e["text"]]); pair.append(nid)
                                node = nodes.setdefault(nid, {"id": nid, "name": e["text"], "entity_type": e["type"], "analysis_eligible": True, "case_ids": set(), "mention_ids": set()})
                                node["case_ids"].add(cid); node["mention_ids"].add(e["occurrence_id"])
                            eid = stable_id("e_", [pair[0], relation, pair[1]])
                            edge = edges.setdefault(eid, {"id": eid, "source": pair[0], "target": pair[1], "relation_type": relation,
                                                         "analysis_eligible": True, "case_ids": set(), "probability_sum": 0., "n_occurrence_observations": 0, "causal_gold": False})
                            edge["case_ids"].add(cid); edge["probability_sum"] += probability; edge["n_occurrence_observations"] += 1
                    audit_pairs.append([i, j, label, probability, reasons])
                for threshold, count in case_counts.items():
                    if count: states[threshold]["retained_cases"].add(cid); states[threshold]["retained_bodies"].add(row["text"])
                case_records.append({"case_id": cid, "mentions": len(entities), "ordered_pairs": ordered, "classified_pairs": len(predicted),
                                     "unclassified_pairs": sum(unclassified.values()), "unclassified_reasons": unclassified,
                                     "thresholds": {str(t): {"retained_pairs": case_counts[t], "typed_canonical_edges_within_case": len(case_edges[t]),
                                                             "rejection_counts": dict(case_rejects[t])} for t in thresholds}})
                audit.write(json.dumps({"case_id": cid, "pair_indices_and_reasons": audit_pairs}, ensure_ascii=False, allow_nan=False) + "\n")
                if totals["processed_cases"] % 100 == 0: print(f"saved-prediction audit: {totals['processed_cases']}/{len(cases)} cases; {totals['classified_pairs']} pairs", flush=True)
    finally:
        if triples is not None: triples.close()
    if seen_cases != set(cases): raise ValueError("Raw predictions do not cover the complete NER case corpus")
    if any(sha(path) != before_hashes[name]["sha256"] for name, path in input_paths.items()): raise ValueError("Input changed during postprocessing")
    measured = {}
    for threshold, state in states.items():
        if state["retained_pairs"] + sum(state["rejection_counts"].values()) != totals["classified_pairs"]: raise AssertionError("Threshold pair accounting incomplete")
        measured[str(threshold)] = {"threshold": threshold, "confidence_gate_enabled": threshold != 0,
            "n_retained_occurrence_pairs": state["retained_pairs"], "fraction_of_all_classified_pairs_retained": state["retained_pairs"] / totals["classified_pairs"] if totals["classified_pairs"] else None,
            "n_retained_cases": len(state["retained_cases"]), "n_all_cases_denominator": len(cases),
            "fraction_of_all_cases_retained": len(state["retained_cases"]) / len(cases) if cases else None,
            "n_retained_unique_case_bodies": len(state["retained_bodies"]), "n_unique_case_bodies_denominator": len({r['text'] for r in rows}),
            "n_typed_canonical_edges": len(state["typed_edges"]), "retained_relation_counts": dict(state["relation_counts"]),
            "rejection_counts_first_failing_gate": {name: state["rejection_counts"][name] for name in GATE_ORDER},
            "relation_precision": None, "relationship_existence_accuracy": None, "causal_validity": None}
    source_binding = {"status": "not_available", "declared_source_inputs": None,
                      "saved_argmax_label_mapping_verified_against_source_relation_counts": False,
                      "main_0_8_rejection_and_retention_counts_reproduced": None}
    if source_summary is not None:
        if dict(raw_relations) != source_summary.get("relation_counts"): raise ValueError("rel2id interpretation disagrees with saved RE relation counts")
        for measured_key, recorded_key in [("processed_cases", "processed_cases"), ("ordered_pairs", "ordered_pairs"), ("classified_pairs", "classified_ordered_pairs"), ("mentions", "mentions")]:
            if totals[measured_key] != source_summary.get("counts", {}).get(recorded_key, 0): raise ValueError("Source RE completeness counts disagree")
        source_binding.update(status="recorded_metadata_and_complete_raw_counts_consistent",
                              declared_source_inputs=source_summary.get("input_sha256"),
                              saved_argmax_label_mapping_verified_against_source_relation_counts=True,
                              entities_SHA_matches_recorded_source=expected_entities == before_hashes["entities"]["sha256"],
                              source_RE_summary_sha256=before_hashes["source_RE_run_summary"]["sha256"])
        if .8 in thresholds and args.min_entity_length == 2 and source_summary.get("protocol", {}).get("confidence_threshold") == .8:
            main = measured["0.8"]
            actual_rejects = Counter(main["rejection_counts_first_failing_gate"]); actual_rejects.update(skipped_reasons)
            if (main["n_retained_occurrence_pairs"] != source_summary["counts"].get("qualified_observations", 0)
                    or main["n_retained_cases"] != source_summary["counts"].get("cases_with_qualified_edges", 0)
                    or main["n_typed_canonical_edges"] != source_summary.get("qualified_graph_edges", 0)
                    or any(actual_rejects[key] != source_summary.get("rejection_counts", {}).get(key, 0)
                           for key in set(actual_rejects) | set(source_summary.get("rejection_counts", {})))):
                raise ValueError("Independent audit does not reproduce unchanged main 0.8 gate counts")
            source_binding["main_0_8_rejection_and_retention_counts_reproduced"] = True
    if args.export_argmax_graph:
        for node in nodes.values(): node["case_ids"] = sorted(node["case_ids"]); node["mention_ids"] = sorted(node["mention_ids"])
        for edge in edges.values():
            edge["case_ids"] = sorted(edge["case_ids"])
            edge["mean_probability"] = edge.pop("probability_sum") / edge["n_occurrence_observations"]
        dump(output / "threshold0_qualified_predicted_graph.json", {"nodes": list(nodes.values()), "edges": list(edges.values()),
             "schema": {"role": "model_qualified_candidates_not_causal_gold", "relationship_existence_head": exists_head,
                        "support_mode": "model_occurrences", "confidence_threshold": 0,
                        "comparison_scope": "independent_argmax_sensitivity_does_not_replace_main_0.8_run"}})
    with (output / "case_counts_and_rejections.jsonl").open("w", encoding="utf-8") as handle:
        for row in case_records: handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    with (output / "threshold_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["threshold", "n_retained_occurrence_pairs", "n_retained_cases", "n_all_cases_denominator", "n_typed_canonical_edges", "fraction_of_all_cases_retained", "retained_relation_counts", "rejection_counts_first_failing_gate"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for result in measured.values(): writer.writerow({k: json.dumps(result[k], ensure_ascii=False) if isinstance(result[k], dict) else result[k] for k in fields})
    summary = {"status": "computed_sensitivity_diagnostic_only", "thresholds": measured, "counts": dict(totals),
        "source_RE_binding": source_binding,
        "raw_relation_counts": dict(raw_relations), "unclassified_reasons": dict(skipped_reasons),
        "independent_nonconfidence_gate_failure_counts_can_overlap": dict(independent_failures),
        "one_character_mention_fraction": totals["one_character_mentions"] / totals["mentions"] if totals["mentions"] else None,
        "one_character_endpoint_fraction_of_classified_pairs": totals["one_character_endpoint_classified_pairs"] / totals["classified_pairs"] if totals["classified_pairs"] else None,
        "saved_max_probability_distributions": {name: quantiles(values) for name, values in probability_groups.items()},
        "max_probability_by_argmax_relation": {name: quantiles(values) for name, values in by_relation.items()},
        "relationship_existence_head_available": exists_head, "relationship_existence_validated": False,
        "original_paper_specifies_added_confidence_threshold_0_8": False, "main_0_8_results_modified_or_replaced": False,
        "inference_or_label_annotation_performed": False, "argmax_labels_changed_to_fit_ontology": False,
        "elapsed_seconds": time.perf_counter() - started,
        "limits": ["All probabilities are saved uncalibrated maximum softmax values.",
                   "Threshold 0 keeps saved argmax classes subject to the other original qualification gates; it is not an accuracy improvement.",
                   "The supplied nine-positive-class model has no null relation head." if not exists_head else "Presence of a null label is not validation of relation existence.",
                   "Retained pairs, cases and graph edges are model candidates; no causal or diagnostic gold is supplied.",
                   "All numbered cases remain in every threshold denominator, including empty/unclassified cases."]}
    dump(output / "analysis_summary.json", summary)
    dump(output / "run_config.json", {"inputs": before_hashes, "thresholds": thresholds, "min_entity_length": args.min_entity_length,
        "declared_NER_variants": sorted({row.get("ner_variant", "unspecified") for row in rows}),
        "declared_NER_checkpoint_sha256": sorted({row["ner_checkpoint_sha256"] for row in rows if row.get("ner_checkpoint_sha256")}),
        "first_failure_gate_order": GATE_ORDER, "qualification_rules_source": rule_source,
        "implementation": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__), "stable_id_helper_sha256": sha(resolve_script('diesel_kg.data.neo4j', HERE))},
        "ontology": {relation: [{"head_types": sorted(h), "tail_types": sorted(t)} for h, t in rules] for relation, rules in ontology.items()},
        "pair_audit_schema": {"row": ["case_id", "pair_indices_and_reasons"], "pair": ["head_index", "tail_index", "saved_argmax_relation_id", "saved_max_probability", "first_rejection_or_accepted_in_threshold_order"], "threshold_order": thresholds},
        "export_argmax_graph": args.export_argmax_graph, "role": "postprocessing_sensitivity_only_no_model_calls_no_gold_annotation"})
    print(json.dumps({"status": summary["status"], "counts": summary["counts"], "thresholds": measured}, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["raw-predictions", "entities", "rel2id", "output"]: parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0, .2, .4, .6, .8])
    parser.add_argument("--min-entity-length", type=int, default=2)
    parser.add_argument("--export-argmax-graph", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__": main()
