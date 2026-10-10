#!/usr/bin/env python3
"""Read-only phase/label quality audit of the frozen historical graph analysis.

Lexical cues create a human review queue, never new gold labels or filtering.
All mention and marker offsets are zero-based Unicode codepoint half-open spans.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import time

RECOVERY_CUES = ["恢复", "正常", "消除", "精准", "顺畅"]
REPAIR_MARKERS = ["维修", "更换", "修复", "检修", "调整", "清理", "清洗", "紧固", "校正", "疏通", "重新安装"]
NEGATION_BEFORE_HINTS = ["不", "未", "无", "不能", "无法", "尚未", "没有", "未能"]
NEGATION_AFTER_HINTS = ["不了", "不成"]
INFLUENCE_RELATIONS = {"cause": False, "causes": False, "导致": False, "引起": False,
                       "caused_by": True, "由…引起": True, "由...引起": True,
                       "由……引起": True, "affect": False, "affects": False, "影响": False}
PHYSICAL_TYPES = {"故障部位", "部件", "组件", "系统"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, records):
    with Path(path).open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path, records, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(record.get(key), ensure_ascii=False) if isinstance(record.get(key), (list, dict))
                             else record.get(key, "") for key in fields})


def literal_hits(text, words):
    hits = []
    for word in words:
        for match in re.finditer(re.escape(word), text):
            start, end = match.span()
            left, right = text[max(0, start - 4):start], text[end:end + 4]
            negation = (["before:"+hint for hint in NEGATION_BEFORE_HINTS if left.endswith(hint)] +
                        ["after:"+hint for hint in NEGATION_AFTER_HINTS if right.startswith(hint)])
            hits.append({"word": word, "start": start, "end": end, "local_negation_hint": negation})
    return sorted(hits, key=lambda h: (h["start"], h["end"], h["word"]))


def safe_node(node):
    return {key: node.get(key) for key in ["id", "name", "entity_type", "labels", "analysis_eligible"]}


def source_edge_evidence(edge, nodes):
    # Source CSV row IDs retain provenance without copying arbitrary properties.
    return {"id": edge["id"], "relation_type": edge["relation_type"],
            "source": safe_node(nodes[edge["source"]]), "target": safe_node(nodes[edge["target"]]),
            "csv_source_rows": [{key: row.get(key) for key in ["csv_record_number", "csv_line_end"]}
                                for row in edge.get("sources", [])],
            "causal_gold_label": edge.get("causal_gold_label", False)}


def phase_context(case, terminal):
    text = case["text"]
    mentions = list(re.finditer(re.escape(terminal), text)) if terminal else []
    markers = literal_hits(text, REPAIR_MARKERS)
    result = []
    for mention_index, mention in enumerate(mentions):
        start, end = mention.span()
        preceding = [marker for marker in markers if marker["end"] <= start]
        following = [marker for marker in markers if marker["start"] >= end]
        if preceding and following:
            order = "repair_markers_both_before_and_after_terminal_mention"
        elif preceding:
            order = "repair_marker_precedes_terminal_mention"
        elif following:
            order = "terminal_mention_precedes_repair_marker"
        elif markers:
            order = "repair_marker_overlaps_terminal_mention"
        else:
            order = "no_repair_marker_in_case"
        bounds = [start, end]
        if preceding:
            bounds.append(preceding[-1]["start"])
        if following:
            bounds.append(following[0]["end"])
        snippet_start = max(0, min(bounds) - 24)
        snippet_end = min(len(text), max(bounds) + 24)
        result.append({"case_id": case["case_id"], "terminal": terminal,
            "mention_index": mention_index, "mention_start": start, "mention_end": end,
            "case_recovery_cue_hits": literal_hits(text, RECOVERY_CUES),
            "terminal_cue_hits_absolute": [{**hit, "start": start + hit["start"], "end": start + hit["end"]}
                                           for hit in literal_hits(terminal, RECOVERY_CUES)],
            "repair_marker_hits": markers, "preceding_repair_markers": preceding,
            "following_repair_markers": following, "lexical_order_category": order,
            "snippet_start": snippet_start, "snippet_end": snippet_end,
            "snippet": text[snippet_start:snippet_end], "phase_review_required": True,
            "interpretation": "literal marker order is evidence for review; it does not prove phase or causal direction"})
    if not mentions:
        result.append({"case_id": case["case_id"], "terminal": terminal, "mention_index": None,
            "mention_start": None, "mention_end": None, "case_recovery_cue_hits": literal_hits(text, RECOVERY_CUES),
            "terminal_cue_hits_absolute": [], "repair_marker_hits": markers,
            "preceding_repair_markers": [], "following_repair_markers": [],
            "lexical_order_category": "terminal_literal_mention_missing", "snippet_start": 0,
            "snippet_end": min(160, len(text)), "snippet": text[:160], "phase_review_required": True,
            "interpretation": "support/mention inconsistency requires review"})
    return result


def run(args):
    started = time.perf_counter()
    graph_dir, output = args.graph_dir, args.output
    output.mkdir(parents=True, exist_ok=True)
    paths = {"source_export": args.source_export, "cases": args.cases,
        "enhanced_graph": graph_dir / "enhanced_graph.json",
        "base_strict_paths": graph_dir / "base_propagation_candidate_paths.jsonl",
        "enhanced_strict_paths": graph_dir / "enhanced_propagation_candidate_paths.jsonl",
        "strict_top10": graph_dir / "propagation_candidate_chains.csv",
        "mechanism_rules": graph_dir / "mechanism_creation_rules.json",
        "frozen_run_summary": graph_dir / "run_summary.json", "frozen_run_config": graph_dir / "run_config.json"}
    before_hashes = {key: sha(path) for key, path in paths.items()}
    source = read_json(args.source_export)
    graph = read_json(paths["enhanced_graph"])
    cases = jsonl(args.cases)
    case_lookup = {case["case_id"]: case for case in cases}
    nodes = {node["id"]: node for node in source["nodes"]}
    enhanced_nodes = {node["id"]: node for node in graph["nodes"]}
    edges = {edge["id"]: edge for edge in source["edges"]}
    frozen_nodes = {node["id"]: node for node in graph["nodes"] if node.get("layer") == "base"}
    support = {node_id: node.get("case_support", []) for node_id, node in frozen_nodes.items()}
    cue_nodes = {nid: literal_hits(node["name"], RECOVERY_CUES) for nid, node in nodes.items()
                 if literal_hits(node["name"], RECOVERY_CUES)}
    for nid in cue_nodes:
        if nid not in support:
            support[nid] = [case["case_id"] for case in cases if nodes[nid]["name"] in case["text"]]
    context_cache = {}

    def contexts(nid, case_ids):
        result = []
        for cid in case_ids:
            key = (nid, cid)
            if key not in context_cache:
                context_cache[key] = phase_context(case_lookup[cid], nodes[nid]["name"])
            result.extend(context_cache[key])
        return result

    source_node_reviews, node_case_evidence = [], []
    for nid, hits in cue_nodes.items():
        evidence = contexts(nid, support[nid])
        node_case_evidence.extend({"node_id":nid, "entity_type":nodes[nid]["entity_type"], **record} for record in evidence)
        source_node_reviews.append({**safe_node(nodes[nid]), "node_cue_hits":hits,
            "phase_review_required":True, "case_ids":support[nid], "n_literal_support_cases":len(support[nid]),
            "context_order_counts":dict(collections.Counter(record["lexical_order_category"] for record in evidence)),
            "warning":"review cue only; negated cues or positive-state labels may be legitimate in a treatment-effect graph"})

    edge_reviews = []
    for edge in source["edges"]:
        relation = edge["relation_type"]
        if relation not in INFLUENCE_RELATIONS:
            continue
        reverse = INFLUENCE_RELATIONS[relation]
        origin, terminal = (edge["target"], edge["source"]) if reverse else (edge["source"], edge["target"])
        if terminal not in cue_nodes:
            continue
        origin_support = support.get(origin)
        if origin_support is None:
            origin_support = [case["case_id"] for case in cases if nodes[origin]["name"] in case["text"]]
            support[origin] = origin_support
        case_ids = sorted(set(origin_support) & set(support[terminal]))
        evidence = contexts(terminal, case_ids)
        source_type = nodes[origin]["entity_type"]
        source_repair_hits = literal_hits(nodes[origin]["name"],REPAIR_MARKERS)
        if source_type == "维修建议" or source_repair_hits:
            role_category = "intervention_to_recovery_context_possible"
        elif source_type in {"故障原因","故障状态","故障现象","故障"}:
            role_category = "fault_to_recovery_phase_mix_review"
        elif source_type in PHYSICAL_TYPES:
            role_category = "component_to_recovery_context_review"
        else:
            role_category = "other_source_role_to_recovery_review"
        edge_reviews.append({**source_edge_evidence(edge,nodes),
            "canonical_source_id":origin, "canonical_terminal_id":terminal,
            "orientation_rule":"reverse_caused_by" if reverse else "as_exported",
            "terminal_entity_type":nodes[terminal]["entity_type"], "terminal_name":nodes[terminal]["name"],
            "terminal_cue_hits":cue_nodes[terminal], "phase_review_required":True,
            "canonical_source_entity_type":source_type, "source_repair_marker_hits":source_repair_hits,
            "source_role_review_category":role_category,
            "case_ids":case_ids,"n_literal_support_cases":len(case_ids),
            "context_order_counts":dict(collections.Counter(record["lexical_order_category"] for record in evidence)),
            "case_support_kind":"new_literal_endpoint_comention_not_observed_relation"})

    path_audits, candidate_case_evidence, split_stats = [], [], {}
    for split in ["base", "enhanced"]:
        records = jsonl(paths[split + "_strict_paths"])
        flagged = []
        for record in records:
            terminal = record["node_ids"][-1]
            if terminal not in cue_nodes:
                continue
            path_id = hashlib.sha256(json.dumps(record["node_ids"]).encode()).hexdigest()[:24]
            basis_ids = sorted({eid for step in record["edge_qualification"] for eid in step["basis_base_edge_ids"]})
            evidence = contexts(terminal,record["case_ids"])
            intervention_ids = [nid for nid in record["node_ids"] if enhanced_nodes[nid]["entity_type"] == "维修建议" or
                                literal_hits(enhanced_nodes[nid]["name"],REPAIR_MARKERS)]
            audit = {"split":split,"path_id":path_id, "node_ids":record["node_ids"],
                "chain":" -> ".join(enhanced_nodes[nid]["name"] for nid in record["node_ids"]),
                "n_nodes":record["n_nodes"], "terminal_node":safe_node(nodes[terminal]),
                "path_source_entity_type":enhanced_nodes[record["node_ids"][0]]["entity_type"],
                "path_node_type_sequence":[enhanced_nodes[nid]["entity_type"] for nid in record["node_ids"]],
                "path_intervention_node_ids":intervention_ids,
                "path_contains_intervention_context":bool(intervention_ids),
                "terminal_cue_hits":cue_nodes[terminal], "phase_review_required":True,
                "case_ids":record["case_ids"], "literal_case_frequency":record["literal_case_frequency"],
                "unique_body_support_frequency":record["unique_body_support_frequency"],
                "basis_base_edge_ids":basis_ids, "original_relation_evidence":[source_edge_evidence(edges[eid],nodes) for eid in basis_ids],
                "context_order_counts":dict(collections.Counter(item["lexical_order_category"] for item in evidence)),
                "candidate_status_unchanged":True, "support_kind":"new_literal_case_support_not_observed_causal_chain"}
            flagged.append(audit)
            candidate_case_evidence.extend({"split":split,"path_id":path_id,"terminal_node_id":terminal,
                                           "basis_base_edge_ids":basis_ids, **item} for item in evidence)
        path_audits.extend(flagged)
        split_stats[split] = {"n_frozen_strict_paths":len(records),"n_phase_review_required_paths":len(flagged),
            "n_phase_review_required_paths_at_least_three_nodes":sum(row["n_nodes"]>=3 for row in flagged),
            "flagged_literal_path_case_incidences":sum(row["literal_case_frequency"] for row in flagged),
            "n_affected_unique_terminal_nodes":len({row["terminal_node"]["id"] for row in flagged}),
            "n_affected_original_case_ids":len({cid for row in flagged for cid in row["case_ids"]}),
            "n_affected_unique_case_bodies":len({case_lookup[cid]["text"] for row in flagged for cid in row["case_ids"]}),
            "n_flagged_paths_with_intervention_context_node":sum(row["path_contains_intervention_context"] for row in flagged),
            "n_flagged_paths_without_intervention_context_node":sum(not row["path_contains_intervention_context"] for row in flagged),
            "path_case_terminal_mention_order_counts":dict(collections.Counter(
                row["lexical_order_category"] for row in candidate_case_evidence if row["split"]==split)),
            "no_paths_filtered":True}

    top10 = []
    with paths["strict_top10"].open(encoding="utf-8-sig",newline="") as f:
        for row in csv.DictReader(f):
            ids = json.loads(row["node_ids"])
            terminal = ids[-1]
            top10.append({**row,"phase_review_required":terminal in cue_nodes,
                          "terminal_entity_type":enhanced_nodes[terminal]["entity_type"],
                          "terminal_name":enhanced_nodes[terminal]["name"],
                          "terminal_cue_hits":cue_nodes.get(terminal,[]),"candidate_status_unchanged":True})

    queue = []
    incident = collections.defaultdict(list)
    for edge in source["edges"]:
        incident[edge["source"]].append(edge);incident[edge["target"]].append(edge)
    by_name = collections.defaultdict(list)
    for node in nodes.values():
        by_name[node["name"]].append(node)
    for name, matching in by_name.items():
        types = sorted({node["entity_type"] for node in matching})
        if len(types)>1:
            queue.append({"category":"identical_literal_name_multiple_entity_types","review_id":hashlib.sha256(name.encode()).hexdigest()[:24],
                "node_ids":[node["id"] for node in matching],"names":[name],"entity_type_labels":types,
                "source_members":[safe_node(node) for node in matching],
                "n_eligible_members":sum(node.get("analysis_eligible",True) for node in matching),
                "basis_base_edge_ids":sorted({edge["id"] for node in matching for edge in incident[node["id"]]}),
                "case_ids":[case["case_id"] for case in cases if name in case["text"]],
                "evidence_summary":"Exact identical text has multiple source type labels; this may be legitimate roles, not verified alias identity."})
    for mechanism in [node for node in graph["nodes"] if node.get("layer")=="mechanism"]:
        members = mechanism["member_entity_ids"]
        member_edges = {edge["id"]:edge for member in members for edge in incident[member]}
        physicals = {nid for edge in member_edges.values() for nid in [edge["source"],edge["target"]]
                     if nodes[nid]["entity_type"] in PHYSICAL_TYPES}
        generic = [nid for nid in members if nodes[nid]["name"] in mechanism.get("dictionary_terms",[])]
        queue.append({"category":"global_mechanism_pooling_requires_component_and_sense_review",
            "review_id":mechanism["id"],"node_ids":members,"names":[nodes[nid]["name"] for nid in members],
            "entity_type_labels":sorted({nodes[nid]["entity_type"] for nid in members}),
            "basis_base_edge_ids":sorted(member_edges),"case_ids":mechanism["case_support"],
            "physical_context_node_ids":sorted(physicals),"n_physical_contexts":len(physicals),
            "literal_dictionary_terms":mechanism.get("dictionary_terms",[]),"short_generic_member_node_ids":generic,
            "recovery_cue_member_node_ids":[nid for nid in members if nid in cue_nodes],
            "evidence_summary":"Dictionary members pooled into one mechanism class across contexts; component-specific equivalence and causal sense are unvalidated. Existing rules remain unchanged."})

    write_jsonl(output/"source_node_phase_review.jsonl",source_node_reviews)
    write_jsonl(output/"source_node_case_phase_evidence.jsonl",node_case_evidence)
    write_jsonl(output/"source_influence_recovery_edges.jsonl",edge_reviews)
    write_jsonl(output/"strict_candidate_phase_review.jsonl",path_audits)
    write_jsonl(output/"strict_candidate_case_phase_evidence.jsonl",candidate_case_evidence)
    write_jsonl(output/"alias_mechanism_manual_review_queue.jsonl",queue)
    write_csv(output/"source_node_phase_review.csv",source_node_reviews,["id","name","entity_type","labels","analysis_eligible","node_cue_hits","n_literal_support_cases","case_ids","context_order_counts","phase_review_required"])
    write_csv(output/"source_influence_recovery_edges.csv",edge_reviews,["id","relation_type","canonical_source_id","canonical_source_entity_type","source_repair_marker_hits","source_role_review_category","canonical_terminal_id","terminal_name","terminal_entity_type","terminal_cue_hits","case_ids","n_literal_support_cases","context_order_counts","phase_review_required"])
    write_csv(output/"strict_candidate_phase_review.csv",path_audits,["split","path_id","chain","n_nodes","path_source_entity_type","path_node_type_sequence","path_intervention_node_ids","path_contains_intervention_context","terminal_node","terminal_cue_hits","literal_case_frequency","unique_body_support_frequency","case_ids","basis_base_edge_ids","context_order_counts","phase_review_required","candidate_status_unchanged"])
    write_csv(output/"strict_candidate_case_phase_evidence.csv",candidate_case_evidence,["split","path_id","case_id","terminal_node_id","terminal","mention_start","mention_end","terminal_cue_hits_absolute","repair_marker_hits","lexical_order_category","snippet_start","snippet_end","snippet","basis_base_edge_ids","phase_review_required"])
    write_csv(output/"strict_top10_phase_review.csv",top10,["rank","chain","literal_case_frequency","unique_body_support_frequency","terminal_name","terminal_entity_type","terminal_cue_hits","case_ids","phase_review_required","candidate_status_unchanged"])
    write_csv(output/"alias_mechanism_manual_review_queue.csv",queue,["category","review_id","node_ids","names","entity_type_labels","source_members","n_eligible_members","basis_base_edge_ids","case_ids","n_physical_contexts","literal_dictionary_terms","short_generic_member_node_ids","recovery_cue_member_node_ids","evidence_summary"])
    eligible_reviews = [row for row in source_node_reviews if row["analysis_eligible"]]
    eligible_edges = [row for row in edge_reviews if edges[row["id"]].get("analysis_eligible",True)]
    summary = {"status":"completed_read_only_quality_review_without_changing_frozen_graph_or_candidates",
        "n_original_cases":len(cases),"n_unique_case_bodies":len({case["text"] for case in cases}),
        "source_graph":{"n_all_nodes":len(nodes),"n_all_typed_edges":len(edges),
            "n_recovery_cue_nodes_all":len(source_node_reviews),"n_recovery_cue_nodes_eligible":len(eligible_reviews),
            "cue_node_entity_types_all":dict(collections.Counter(row["entity_type"] for row in source_node_reviews)),
            "cue_node_entity_types_eligible":dict(collections.Counter(row["entity_type"] for row in eligible_reviews)),
            "n_cue_nodes_with_local_negation_hint":sum(any(hit["local_negation_hint"] for hit in row["node_cue_hits"]) for row in source_node_reviews),
            "n_influence_edges_to_recovery_cue_terminal_all":len(edge_reviews),
            "n_influence_edges_to_recovery_cue_terminal_eligible":len(eligible_edges),
            "influence_edge_relations_all":dict(collections.Counter(row["relation_type"] for row in edge_reviews)),
            "influence_source_entity_types_all":dict(collections.Counter(row["canonical_source_entity_type"] for row in edge_reviews)),
            "influence_source_role_review_categories_all":dict(collections.Counter(row["source_role_review_category"] for row in edge_reviews)),
            "influence_relation_source_type_cross_counts_all":dict(collections.Counter(
                row["relation_type"]+" | "+row["canonical_source_entity_type"] for row in edge_reviews)),
            "influence_terminal_types_all":dict(collections.Counter(row["terminal_entity_type"] for row in edge_reviews))},
        "strict_candidate_splits":split_stats,"strict_top10_flagged_ranks":[int(row["rank"]) for row in top10 if row["phase_review_required"]],
        "candidate_case_mention_order_counts_base_plus_enhanced_nonindependent":dict(collections.Counter(row["lexical_order_category"] for row in candidate_case_evidence)),
        "candidate_evidence_row_scope":"path x original case x terminal literal mention; shared paths and cases occur in both base and enhanced",
        "source_case_mention_order_counts":dict(collections.Counter(row["lexical_order_category"] for row in node_case_evidence)),
        "manual_queue_categories":dict(collections.Counter(row["category"] for row in queue)),
        "no_gold_labels_assigned":True,"no_candidate_paths_filtered":True,
        "limitations":["Cue hits flag phase ambiguity, not necessarily annotation errors; normal and recovery outcomes can belong to treatment-effect knowledge.",
            "Repair-marker order does not prove chronological phase or causality; negation is only a local lexical hint.",
            "Case bindings are newly computed literal support rather than recovered original annotation provenance.",
            "Counts include overlapping review evidence and repeated original case IDs; path-case rows are not independent samples.",
            "Existing candidate qualifies by endpoint type but recovery/positive semantics require phase-aware expert review."]}
    after_hashes = {key:sha(path) for key,path in paths.items()}
    if after_hashes!=before_hashes:
        raise RuntimeError("Frozen input changed during audit; rerun after stabilizing inputs")
    summary["frozen_input_sha256_unchanged"] = True
    summary["elapsed_seconds"] = time.perf_counter()-started
    dump(output/"quality_audit.json",summary)
    summary_rows = [{"scope":"source_graph","metric":key,"value":value} for key,value in summary["source_graph"].items()]
    summary_rows += [{"scope":"strict_"+split,"metric":key,"value":value}
                     for split,stats in split_stats.items() for key,value in stats.items()]
    write_csv(output/"quality_audit_summary.csv",summary_rows,["scope","metric","value"])
    dump(output/"quality_audit_protocol.json",{"recovery_cues":RECOVERY_CUES,"repair_markers":REPAIR_MARKERS,
        "negation_hints":{"before_cue":NEGATION_BEFORE_HINTS,"after_cue":NEGATION_AFTER_HINTS},"influence_relation_orientation":INFLUENCE_RELATIONS,
        "offset_convention":"zero-based Unicode codepoint half-open spans", "snippet_rule":"terminal mention plus nearest repair marker on each side, with 24-codepoint context",
        "phase_flag_rule":"terminal node name contains a fixed recovery cue; candidate status remains unchanged regardless of context order",
        "manual_review_only":True,"input_sha256":before_hashes,"command":sys.argv,"script_sha256":sha(__file__)})
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir",type=Path,required=True)
    parser.add_argument("--source-export",type=Path,required=True)
    parser.add_argument("--cases",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())


if __name__=="__main__":
    main()
