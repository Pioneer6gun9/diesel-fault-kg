#!/usr/bin/env python3
"""Source-traceable completion of the paper's graph analysis under stated rules.

All basic diesel KG nodes/relations are retained. The analysis projection uses
causal, functional, structural, spatial and manifestation relationships. New
mechanism classes are literal dictionary abstractions, not recovered author
code or causal gold. No paper values or diagnostic/interpretability scores
are assigned. Original 1522 case IDs remain separate even for duplicate text.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
from pathlib import Path
import time

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SYSTEMS = ["fuel_combustion", "hydraulic", "cooling_heat", "electrical_control",
           "mechanical_sealing", "intake_exhaust", "transmission_steering"]
MECHANISMS = {
    "clogging": ["堵塞", "阻塞", "堵", "堵住"],
    "wear": ["磨损", "磨耗", "磨蚀"],
    "leakage": ["泄漏", "渗漏", "漏油", "漏水", "漏气", "渗油", "渗水"],
    "short_circuit": ["短路"], "open_circuit": ["断路", "断线"],
    "poor_atomization": ["雾化不良", "雾化差", "雾化不足"],
    "incomplete_combustion": ["燃烧不充分", "不完全燃烧", "燃烧不完全", "燃烧不足"],
    "low_pressure": ["压力不足", "压力下降", "压力过低", "油压不足", "油压下降", "低油压", "低压"],
    "overheating": ["过热", "温度过高", "高温", "过高温度"],
    "sticking": ["卡滞", "卡死", "卡住", "卡涩"],
    "corrosion": ["腐蚀", "锈蚀", "生锈"], "carbon_deposit": ["积碳", "积炭"],
    "failure": ["失效", "损坏"], "loosening": ["松动", "松脱"],
    "fracture": ["断裂", "裂纹", "破裂"], "aging": ["老化", "硬化"],
}
FAULT_TYPES = {"故障原因", "故障状态", "故障现象"}
PHYSICAL_TYPES = {"故障部位", "部件", "组件", "系统"}
RELATIONS = {
    "cause": ("causal_candidate", False), "causes": ("causal_candidate", False),
    "导致": ("causal_candidate", False), "引起": ("causal_candidate", False),
    "caused_by": ("causal_candidate", True), "由…引起": ("causal_candidate", True),
    "由...引起": ("causal_candidate", True), "由……引起": ("causal_candidate", True),
    "affect": ("functional_influence", False), "affects": ("functional_influence", False),
    "影响": ("functional_influence", False),
    "include": ("structural_context", False), "contains": ("structural_context", False),
    "包含": ("structural_context", False), "part_of": ("structural_context", True),
    "occur_at": ("spatial_context", True), "occurs_at": ("spatial_context", True),
    "发生于": ("spatial_context", True), "located_in": ("spatial_context", True),
    "show_as": ("manifestation_context", False), "表现为": ("manifestation_context", False),
}
VARIANTS = {"full": (1/3, 1/3, 1/3), "no_semantic": (0, .5, .5),
            "no_cooccurrence": (.5, 0, .5), "no_physical": (.5, .5, 0)}


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def csv_write(path, records, fields=None):
    records = list(records)
    if not fields:
        fields = list(records[0]) if records else []
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in record.items()})


def save_figure(fig, path):
    fig.savefig(path)
    fig.savefig(Path(path).with_suffix(".svg"))


def match_mechanisms(name):
    return {key: sorted({word for word in words if word in name})
            for key, words in MECHANISMS.items() if any(word in name for word in words)}


def load_topics(directory, cases, topic_model_dir=None):
    directory = Path(directory)
    fixed = Path(topic_model_dir) if topic_model_dir else directory / "paper_fixed_K07"
    doc = fixed / "document_topic.csv"
    if not doc.exists() and not topic_model_dir:
        doc = directory / "document_topic.csv"
        fixed = directory
    system_file = fixed / "case_system_probabilities.csv"
    if not system_file.exists():
        system_file = directory / "case_system_probabilities.csv"
    with doc.open(encoding="utf-8-sig", newline="") as f:
        document_rows = list(csv.DictReader(f))
    with system_file.open(encoding="utf-8-sig", newline="") as f:
        system_rows = list(csv.DictReader(f))
    topic_columns = sorted((c for c in document_rows[0] if re.fullmatch(r"T[0-9]+", c)), key=lambda c: int(c[1:]))
    if not topic_columns or [int(c[1:]) for c in topic_columns] != list(range(1, len(topic_columns) + 1)):
        raise ValueError("Topic CSV must contain contiguous T01..TK topic columns")
    if len({r["case_id"] for r in document_rows}) != len(document_rows) or len({r["case_id"] for r in system_rows}) != len(system_rows):
        raise ValueError("Duplicate case IDs in topic/system CSV")
    topic_vectors = {r["case_id"]: np.array([float(r[c]) for c in topic_columns]) for r in document_rows}
    system_vectors = {r["case_id"]: np.array([float(r[s]) for s in SYSTEMS]) for r in system_rows}
    expected = {c["case_id"] for c in cases}
    if set(topic_vectors) != expected or set(system_vectors) != expected:
        raise ValueError("Topic and system case IDs must exactly match the original case IDs")
    if any(not np.isfinite(v).all() or v.min() < 0 or not np.isclose(v.sum(), 1, atol=1e-5) for v in topic_vectors.values()):
        raise ValueError("Every document topic distribution must be finite, nonnegative and sum to one")
    if any(not np.isfinite(v).all() or v.min() < 0 or v.sum() > 1.00001 for v in system_vectors.values()):
        raise ValueError("System probabilities must be nonnegative with sum <= 1 (unmapped mass allowed)")
    return topic_vectors, system_vectors, {"topic_dimension": len(topic_columns), "topic_columns": topic_columns, "topic_model_directory": str(fixed), "engineering_system_dimension": len(SYSTEMS), "document_topic": str(doc), "document_topic_sha256": sha(doc),
        "case_system_probabilities": str(system_file), "case_system_sha256": sha(system_file)}


def ground_nodes(nodes, cases, topic_vectors, system_vectors, support_mode="literal"):
    topic_dimension = len(next(iter(topic_vectors.values())))
    case_text = {c["case_id"]: c["text"] for c in cases}
    if support_mode not in {"literal", "model_occurrences"}:
        raise ValueError("Unknown case support mode")
    support = {}
    for node in nodes.values():
        if support_mode == "model_occurrences":
            if not isinstance(node.get("case_ids"), list) or not node["case_ids"]:
                raise ValueError(f"Model node requires explicit nonempty case_ids: {node['id']}")
            literal = set(node["case_ids"])
            if not literal <= case_text.keys():
                raise ValueError(f"Model node references unknown case IDs: {node['id']}")
            if not node["name"] or any(node["name"] not in case_text[cid] for cid in literal):
                raise ValueError(f"Model node name is not locatable in its declared cases: {node['id']}")
            support_kind = "input_model_qualified_occurrence_case_ids_not_causal_gold"
        else:
            literal = {case["case_id"] for case in cases if node["name"] and node["name"] in case["text"]}
            support_kind = "new_exact_name_literal_grounding_not_recovered_historical_case_id"
        support[node["id"]] = literal
        node["case_support"] = sorted(literal)
        node["case_support_kind"] = support_kind
        node["topic_vector_source"] = f"mean_grounded_case_K{topic_dimension}" if literal else "zero_no_literal_case_support"
        if support_mode == "model_occurrences":
            node["topic_vector_source"] = f"mean_input_model_occurrence_cases_K{topic_dimension}"
        node["topic_distribution"] = np.mean([topic_vectors[c] for c in sorted(literal)], axis=0).tolist() if literal else [0.] * topic_dimension
        node["system_distribution"] = np.mean([system_vectors[c] for c in sorted(literal)], axis=0).tolist() if literal else [0.] * 7
    return support


def build_layers(export, cases, topic_vectors, system_vectors, support_mode="literal"):
    topic_dimension = len(next(iter(topic_vectors.values())))
    base_nodes = {n["id"]: json.loads(json.dumps(n)) for n in export["nodes"] if n.get("analysis_eligible", True)}
    base_edges = [json.loads(json.dumps(e)) for e in export["edges"]
                  if e.get("analysis_eligible", True) and e["source"] in base_nodes and e["target"] in base_nodes]
    for n in base_nodes.values():
        n["layer"] = "base"
    support = ground_nodes(base_nodes, cases, topic_vectors, system_vectors, support_mode)
    known_cases = {c["case_id"] for c in cases}
    edge_support_kind = ("input_model_qualified_ordered_occurrence_pair_case_ids_not_causal_gold"
                         if support_mode == "model_occurrences" else "new_literal_endpoint_comention_not_observed_relation")
    analysis = []
    incident = collections.defaultdict(set)
    for e in base_edges:
        if support_mode == "model_occurrences":
            if not isinstance(e.get("case_ids"), list) or not e["case_ids"]:
                raise ValueError(f"Model edge requires explicit nonempty case_ids: {e['id']}")
            case_ids = set(e["case_ids"])
            if not case_ids <= known_cases or not case_ids <= (support[e["source"]] & support[e["target"]]):
                raise ValueError(f"Model edge support outside source/target occurrence cases: {e['id']}")
            # Membership is an abstraction of model-observed endpoint mentions;
            # its traceability may include non-propagation model relations.
            incident[e["source"]].add(e["id"]); incident[e["target"]].add(e["id"])
        else:
            case_ids = support[e["source"]] & support[e["target"]]
        e["case_support"] = sorted(case_ids)
        e["case_support_kind"] = edge_support_kind
        if e["relation_type"] not in RELATIONS:
            e["analysis_role"] = "metadata_retained_in_full_KG_not_propagation"
            continue
        kind, reverse = RELATIONS[e["relation_type"]]
        s, t = (e["target"], e["source"]) if reverse else (e["source"], e["target"])
        if kind == "spatial_context":
            # Prefer component -> fault, as a context edge, without inventing causality.
            if base_nodes[t]["entity_type"] in PHYSICAL_TYPES and base_nodes[s]["entity_type"] not in PHYSICAL_TYPES:
                s, t = t, s
            elif base_nodes[s]["entity_type"] not in PHYSICAL_TYPES and base_nodes[t]["entity_type"] not in PHYSICAL_TYPES:
                e["spatial_type_review_required"] = True
        a = {"id": "analysis_" + e["id"], "source": s, "target": t,
             "relation_type": e["relation_type"], "kind": kind, "layer": "base",
             "original_source": e["source"], "original_target": e["target"],
             "basis_base_edge_ids": [e["id"]], "case_support": sorted(case_ids),
             "case_support_kind": edge_support_kind,
             "causal_gold": False, "physical": 1.0,
             "direction_rule": "component_to_fault_context" if kind == "spatial_context" else ("reverse_caused_by_or_part_of" if reverse else "as_exported")}
        e["analysis_role"] = kind
        e["analysis_source"], e["analysis_target"] = s, t
        analysis.append(a)
        incident[s].add(e["id"]); incident[t].add(e["id"])
    enhanced_nodes = dict(base_nodes)
    entity_mechanisms = {}
    for node_id, n in base_nodes.items():
        if n["entity_type"] in FAULT_TYPES:
            found = match_mechanisms(n["name"])
            if found:
                entity_mechanisms[node_id] = found
    mechanism_members = collections.defaultdict(list)
    for nid, found in entity_mechanisms.items():
        for mechanism in found:
            mechanism_members[mechanism].append(nid)
    for mechanism, members in mechanism_members.items():
        mid = "mechanism_" + mechanism
        supported = set().union(*(support[nid] for nid in members))
        support[mid] = supported
        enhanced_nodes[mid] = {"id": mid, "name": "Mechanism:" + mechanism, "entity_type": "机制类",
            "layer": "mechanism", "dictionary_terms": MECHANISMS[mechanism], "member_entity_ids": sorted(members),
            "case_support": sorted(supported), "case_support_kind": ("union_of_input_model_qualified_member_occurrence_cases" if support_mode == "model_occurrences" else "union_of_literal_member_entity_support"),
            "topic_distribution": np.mean([topic_vectors[c] for c in sorted(supported)], axis=0).tolist() if supported else [0.] * topic_dimension,
            "system_distribution": np.mean([system_vectors[c] for c in sorted(supported)], axis=0).tolist() if supported else [0.] * 7,
            "topic_vector_source": "mean_grounded_member_cases" if supported else "zero_no_literal_case_support"}
    bridges = {}

    def bridge(s, t, kind, basis, supported, evidence):
        identity = (s, t, kind)
        if identity not in bridges:
            bid = "bridge_" + hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
            bridges[identity] = {"id": bid, "source": s, "target": t, "relation_type": kind,
                "kind": kind, "layer": "mechanism", "basis_base_edge_ids": set(), "case_support": set(),
                "creation_evidence": [], "causal_gold": False, "physical": 0.0,
                "case_support_kind": ("model_qualified_member_occurrence_or_inherited_input_model_edge_case_ids_not_causal_gold" if support_mode == "model_occurrences" else "literal_entity_support_and_inherited_base_edge_endpoint_comention_not_observed_chain")}
        e = bridges[identity]
        e["basis_base_edge_ids"].update(basis); e["case_support"].update(supported)
        e["creation_evidence"].append(evidence)
        e["physical"] = float(bool(e["basis_base_edge_ids"]))

    case_by_edge = {e["id"]: set(e["case_support"]) for e in base_edges} if support_mode == "model_occurrences" else None
    for nid, found in entity_mechanisms.items():
        for mechanism, terms in found.items():
            member_support = support[nid]
            if support_mode == "model_occurrences":
                member_support = support[nid] & set().union(*(case_by_edge[b] for b in incident[nid]))
            bridge(nid, "mechanism_" + mechanism, "mechanism_membership_context", incident[nid], member_support,
                   {"rule": "literal_dictionary_in_fault_entity", "entity_id": nid, "matched_terms": terms})
    for e in analysis:
        s, t = e["source"], e["target"]
        # Component -> mechanism only has an existing qualifying base connection.
        for physical, fault in ((s, t), (t, s)):
            if (base_nodes[physical]["entity_type"] in PHYSICAL_TYPES and fault in entity_mechanisms
                    and e["kind"] in {"causal_candidate", "spatial_context", "structural_context"}):
                for mechanism, terms in entity_mechanisms[fault].items():
                    bridge(physical, "mechanism_" + mechanism, "physical_mechanism_context",
                           e["basis_base_edge_ids"], set(e["case_support"]),
                           {"rule": "literal_mechanism_in_connected_fault_entity", "fault_entity_id": fault,
                            "matched_terms": terms, "inherited_base_edge": e["basis_base_edge_ids"][0]})
        # Inherits a stated causal/influence relation, never a new arbitrary similarity edge.
        if s in entity_mechanisms and e["kind"] in {"causal_candidate", "functional_influence"}:
            for mechanism, terms in entity_mechanisms[s].items():
                bridge("mechanism_" + mechanism, t, "mechanism_chain_inherited_candidate",
                       e["basis_base_edge_ids"], set(e["case_support"]),
                       {"rule": "abstract_source_of_existing_causal_or_influence_candidate",
                        "fault_entity_id": s, "matched_terms": terms,
                        "inherited_base_edge": e["basis_base_edge_ids"][0]})
    for e in bridges.values():
        e["basis_base_edge_ids"] = sorted(e["basis_base_edge_ids"])
        e["case_support"] = sorted(e["case_support"])
    augmented = analysis + list(bridges.values())
    for e in augmented:
        u, v = enhanced_nodes[e["source"]], enhanced_nodes[e["target"]]
        a, b = np.array(u["topic_distribution"]), np.array(v["topic_distribution"])
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        e["semantic"] = float(a @ b / denominator) if denominator else 0.0
        su, sv = support[e["source"]], support[e["target"]]
        e["cooccurrence"] = len(su & sv) / np.sqrt(len(su) * len(sv)) if su and sv else 0.0
        e["weight"] = (e["semantic"] + e["cooccurrence"] + e["physical"]) / 3
        e["weight_interpretation"] = "uncalibrated_association_score_not_causal_probability"
    return base_nodes, base_edges, enhanced_nodes, analysis, list(bridges.values()), augmented


def projection(nodes, edges, coefficients=(1/3, 1/3, 1/3)):
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    grouped = collections.defaultdict(list)
    for e in edges:
        if e["source"] != e["target"]:
            grouped[e["source"], e["target"]].append(e)
    for (s, t), observations in grouped.items():
        weights = [sum(c * e[key] for c, key in zip(coefficients, ["semantic", "cooccurrence", "physical"])) for e in observations]
        graph.add_edge(s, t, weight=float(np.mean(weights)), edge_ids=[e["id"] for e in observations],
                       case_support=set().union(*(set(e["case_support"]) for e in observations)),
                       kinds=sorted({e["kind"] for e in observations}),
                       basis_base_edge_ids=sorted({b for e in observations for b in e["basis_base_edge_ids"]}))
    return graph


def community_analysis(graph, seed=42):
    undirected = nx.Graph()
    undirected.add_nodes_from(graph)
    for s, t, a in graph.edges(data=True):
        if a["weight"] > 0:
            previous = undirected.get_edge_data(s, t, {}).get("weight", 0.)
            undirected.add_edge(s, t, weight=previous + a["weight"])
    active = undirected.subgraph([n for n, d in undirected.degree if d]).copy()
    communities = nx.community.louvain_communities(active, weight="weight", seed=seed, resolution=1.) if active.number_of_edges() else []
    communities = sorted(communities, key=lambda c: (-len(c), min(c)))
    assigned = {nid: i for i, group in enumerate(communities) for nid in group}
    groups = {"within": [], "between": [], "unassigned": []}
    for s, t, a in graph.edges(data=True):
        kind = "unassigned" if s not in assigned or t not in assigned else ("within" if assigned[s] == assigned[t] else "between")
        groups[kind].append(a["weight"])
    stats = {}
    for key, values in groups.items():
        stats[key] = {"n_directed_pairs": len(values), "mean": float(np.mean(values)) if values else None,
                      "std": float(np.std(values)) if values else None, "weights": values}
    return assigned, {"n_auto_communities": len(communities), "unassigned_nodes": len(graph) - len(assigned),
        "modularity": nx.community.modularity(active, communities, weight="weight") if communities else None,
        "n_active_nodes": len(active), "n_positive_undirected_edges": active.number_of_edges(),
        "within_between": stats, "definition": "automatic Louvain on positive-weight undirected projection; reciprocal weights summed; excludes isolated nodes"}


def reachability(graph):
    condensation = nx.condensation(graph)
    sizes = {c: len(a["members"]) for c, a in condensation.nodes(data=True)}
    descendants = {}
    for c in reversed(list(nx.topological_sort(condensation))):
        descendants[c] = set()
        for child in condensation.successors(c):
            descendants[c].add(child); descendants[c].update(descendants[child])
    intra = sum(n * (n - 1) for n in sizes.values())
    inter = sum(sizes[c] * sum(sizes[d] for d in ds) for c, ds in descendants.items())
    return {"strongly_connected_components": len(sizes), "largest_scc_nodes": max(sizes.values(), default=0),
        "reachable_distinct_ordered_endpoint_pairs": intra + inter, "within_scc_pairs": intra,
        "between_scc_pairs": inter, "counting": "exact SCC-condensation transitive reachability; counts endpoint pairs, not simple paths; no hop cap"}


def count_patterns(graph):
    linear = sum(len(set(graph.predecessors(n)) - {n}) * len(set(graph.successors(n)) - {n})
                 - len(set(graph.predecessors(n)) & set(graph.successors(n)) - {n}) for n in graph)
    branch = sum(len(list(graph.successors(n))) * (len(list(graph.successors(n))) - 1) // 2 for n in graph)
    converge = sum(len(list(graph.predecessors(n))) * (len(list(graph.predecessors(n))) - 1) // 2 for n in graph)
    return {"linear_distinct_three_node_walks": linear, "branching_neighbor_pairs": branch,
            "convergent_neighbor_pairs": converge, "interpretation": "nonexclusive static local motifs; not observed fault evolution or hazard/velocity"}


def enumerate_case_paths(graph, cases, max_case_paths=0, max_case_seconds=0, support_mode="literal"):
    """Enumerate every simple directed path prefix >=2 nodes for each case.

    No hop cap. Resource limits, if requested, are explicit and mark incomplete.
    Same local edge signature is cached, but each original case ID contributes.
    """
    edge_by_case = collections.defaultdict(list)
    text_by_case = {c["case_id"]: c["text"] for c in cases}
    for s, t, a in graph.edges(data=True):
        for cid in a["case_support"]:
            edge_by_case[cid].append((s, t))
    cache, frequencies, completion = {}, collections.defaultdict(set), []
    for case in cases:
        cid = case["case_id"]
        signature = tuple(sorted(set(edge_by_case[cid])))
        if signature not in cache:
            local = nx.DiGraph(); local.add_edges_from(signature)
            found, state = set(), {"complete": True, "stop_reason": None}
            start = time.perf_counter()
            stopped = False
            for source in sorted(local):
                stack = [(source, (source,), frozenset({source}))]
                while stack:
                    node, path, visited = stack.pop()
                    for nxt in sorted(local.successors(node), reverse=True):
                        if nxt in visited:
                            continue
                        extended = path + (nxt,)
                        found.add(extended)
                        if ((max_case_paths and len(found) >= max_case_paths) or
                            (max_case_seconds and time.perf_counter() - start >= max_case_seconds)):
                            state = {"complete": False, "stop_reason": "explicit_resource_limit"}
                            stopped = True; break
                        stack.append((nxt, extended, visited | {nxt}))
                    if stopped:
                        break
                if stopped:
                    break
            cache[signature] = (found, state)
        found, state = cache[signature]
        for path in found:
            frequencies[path].add(cid)
        completion.append({"case_id": cid, "n_grounded_edges": len(signature), "n_simple_paths": len(found), **state})
    records = []
    for path, support in frequencies.items():
        weights = [graph[s][t]["weight"] for s, t in zip(path, path[1:])]
        record = {"node_ids": list(path), "n_nodes": len(path), "literal_case_frequency": len(support),
                        "unique_body_support_frequency": len({text_by_case[c] for c in support}),
                        "case_ids": sorted(support), "average_edge_weight": float(np.mean(weights)),
                        "minimum_edge_weight": min(weights), "product_edge_weight": float(np.prod(weights)),
                        "support_interpretation": ("all links have declared input-model-qualified occurrence case support; not observed causal sequence" if support_mode == "model_occurrences" else "all links have new literal case co-mention grounding; not observed causal sequence")}
        if support_mode == "model_occurrences":
            record["model_qualified_case_frequency"] = len(support)
            record["legacy_frequency_field_note"] = "literal_case_frequency is retained for interface compatibility and counts input-model-qualified case support in this branch"
        records.append(record)
    records.sort(key=lambda r: (-r["literal_case_frequency"], -r["average_edge_weight"], r["node_ids"]))
    return records, completion, {"n_unique_case_grounded_simple_paths": len(records),
        "path_case_incidences": sum(r["literal_case_frequency"] for r in records),
        "n_cases_complete": sum(r["complete"] for r in completion), "n_cases_incomplete": sum(not r["complete"] for r in completion),
        "hop_cap": None, "resource_limits": {"max_case_paths": max_case_paths, "max_case_seconds": max_case_seconds},
        "n_unique_case_subgraphs": len(cache), "definition": ("all simple directed paths of >=2 nodes in declared input-model-qualified edge-case subgraphs; no full-corpus same-name case expansion; original duplicate-text IDs retained" if support_mode == "model_occurrences" else "all simple directed paths of >=2 nodes in each literal-case-grounded edge subgraph; original duplicate-text cases retain separate IDs")}


def path_scope_levels(paths, nodes, analysis_edges):
    context = collections.defaultdict(set)
    for edge in analysis_edges:
        s, t = edge["source"], edge["target"]
        if edge["kind"] in {"structural_context", "spatial_context"}:
            for physical, fault in [(s,t),(t,s)]:
                if nodes[physical]["entity_type"] in PHYSICAL_TYPES and nodes[fault]["entity_type"] in FAULT_TYPES:
                    context[fault].add(physical)
    counts, incidences = collections.Counter(), collections.Counter()
    for record in paths:
        ids = record["node_ids"]
        physicals = {n for n in ids if nodes[n]["entity_type"] in PHYSICAL_TYPES}
        for n in ids: physicals.update(context[n])
        systems = {int(np.argmax(nodes[n]["system_distribution"])) for n in ids
                   if sum(nodes[n]["system_distribution"]) > 0}
        engine_consequence = any(nodes[n]["entity_type"] == "影响后果" and
            any(term in nodes[n]["name"] for term in ["整机", "停机", "无法启动", "动力完全", "发动机损坏"]) for n in ids)
        if engine_consequence or len(systems) > 1:
            scope = "cross_mapped_system_or_engine_consequence_candidate"
        elif len(physicals) == 1:
            scope = "one_component_context_candidate"
        elif len(systems) == 1:
            scope = "within_one_mapped_system_candidate"
        else:
            scope = "unmapped_scope"
        record["scope_level_candidate"] = scope
        counts[scope] += 1; incidences[scope] += record["literal_case_frequency"]
    return {"unique_path_counts":dict(counts), "literal_path_case_incidences":dict(incidences),
        "no_temporal_observations":True,
        "definition":"heuristic context levels from explicit component/location links and weak system argmax mapping; no propagation speed or verified evolution levels"}


def system_coupling(graph, nodes):
    matrix = np.zeros((7, 7)); within, between, unassigned = [], [], []
    for s, t, a in graph.edges(data=True):
        sv, tv = np.array(nodes[s]["system_distribution"]), np.array(nodes[t]["system_distribution"])
        matrix += a["weight"] * np.outer(sv, tv)
        if sv.sum() == 0 or tv.sum() == 0:
            unassigned.append(a["weight"])
        elif int(np.argmax(sv)) == int(np.argmax(tv)):
            within.append(a["weight"])
        else:
            between.append(a["weight"])
    norm = matrix / matrix.max() if matrix.max() else matrix.copy()
    return matrix, norm, {"definition": "sum(weight*source_soft_system_probability*target_soft_system_probability), max-normalized display; no 0.05 floor",
        "system_order": SYSTEMS, "mapping_kind": "keyword-anchored topic/system distributions; not expert physical-system gold",
        "hard_argmax_weight_stats": {k: {"n_directed_pairs": len(v), "mean": float(np.mean(v)) if v else None}
                                    for k, v in [("within", within), ("between", between), ("unassigned", unassigned)]},
        "sum_assigned_weight_mass": float(matrix.sum()),
        "all_edge_weight_mass": sum(a["weight"] for _, _, a in graph.edges(data=True))}


def propagation_candidates(paths, graph, nodes, support_mode="literal"):
    """Filter context paths by disclosed propagation-candidate qualifications.

    A literal association with a mechanism class is not itself propagation.
    Context can enter through a physical first hop or a final manifestation,
    while every retained path needs an actual inherited causal/influence link.
    This is still candidate screening without verified causal sequence labels.
    """
    influential = {"causal_candidate", "functional_influence", "mechanism_chain_inherited_candidate"}
    first_context = {"structural_context", "spatial_context", "physical_mechanism_context"}
    accepted, audits, reasons = [], [], collections.Counter()
    for path in paths:
        ids = path["node_ids"]
        failure_reasons, edge_checks = [], []
        if nodes[ids[-1]]["entity_type"] not in {"故障现象", "影响后果"}:
            failure_reasons.append("endpoint_not_fault_phenomenon_or_consequence")
        path_entities = set(ids)
        if any(nodes[n].get("layer") == "mechanism" and
               (path_entities & set(nodes[n].get("member_entity_ids", []))) for n in ids):
            failure_reasons.append("same_path_literal_member_and_mechanism_class_redundancy")
        has_influence = False
        for index, (s, t) in enumerate(zip(ids, ids[1:])):
            data = graph[s][t]
            kinds = set(data["kinds"])
            has_influence |= bool(kinds & influential)
            allowed = kinds & influential
            if index == 0 and nodes[s]["entity_type"] in PHYSICAL_TYPES:
                allowed |= kinds & first_context
            if index == len(ids) - 2:
                allowed |= kinds & {"manifestation_context"}
            if not allowed:
                failure_reasons.append("edge_has_only_membership_or_context_outside_allowed_position")
            edge_checks.append({"source":s,"target":t,"available_kinds":sorted(kinds),
                                "allowed_kinds":sorted(allowed),"basis_base_edge_ids":data["basis_base_edge_ids"]})
        if not has_influence:
            failure_reasons.append("no_causal_functional_or_inherited_influence_candidate")
        failure_reasons = sorted(set(failure_reasons))
        audit = {**path, "propagation_candidate_qualified":not failure_reasons,
                 "filter_rejection_reasons":failure_reasons,"edge_qualification":edge_checks,
                 "filter_rule_version":"strict_candidate_qualification_v1",
                 "support_interpretation":("input_model_qualified_occurrence_case_support_not_observed_causal_chain" if support_mode == "model_occurrences" else "new_literal_case_support_not_observed_causal_chain")}
        audits.append(audit)
        if not failure_reasons:
            accepted.append(audit)
        else:
            reasons.update(failure_reasons)
    return accepted, audits, {
        "n_context_paths_screened":len(paths),"n_qualified_propagation_candidate_paths":len(accepted),
        "n_qualified_paths_at_least_three_nodes":sum(p["n_nodes"]>=3 for p in accepted),
        "qualified_literal_path_case_incidences":sum(p["literal_case_frequency"] for p in accepted),
        "rejected_path_counts_by_reason_nonexclusive":dict(reasons),
        "definition":"qualification screening of context simple paths; not observed fault paths or root-cause accuracy",
        "rules":["endpoint must be fault phenomenon or impact consequence",
                 "at least one causal, functional influence, or inherited mechanism-chain candidate edge",
                 "membership-only pairs excluded; any path containing a literal member and its same mechanism class excluded",
                 "structural/spatial/physical-mechanism context allowed only at physical source first hop",
                 "manifestation context allowed only at final hop",
                 ("original and inherited case supports are restricted to input-model-qualified occurrence case IDs, not verified causal sequences" if support_mode == "model_occurrences" else "original and inherited case supports are literal co-mention, not verified causal sequences")]}


def save_plots(output, base_nodes, enhanced_nodes, basic_edges, bridges, base_graph, graph, cases, matrix, norm, stats, pattern_counts, support_mode="literal"):
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 120})
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = np.arange(3)
    before = [len(base_nodes), len(basic_edges), base_graph.number_of_edges()]
    after = [len(enhanced_nodes), len(basic_edges) + len(bridges), graph.number_of_edges()]
    ax.bar(bars - .18, before, .36, label="Before: retained basic KG")
    ax.bar(bars + .18, after, .36, label="After: traceable mechanism layer")
    ax.set_xticks(bars, ["Nodes (all retained)", "Typed KG edges", "Analysis directed pairs"])
    for xs, vals in [(bars - .18, before), (bars + .18, after)]:
        for x, v in zip(xs, vals): ax.text(x, v, str(v), ha="center", va="bottom")
    ax.set_ylabel("Computed counts"); ax.legend(); ax.set_title("Global graph counts under disclosed completion rules")
    fig.tight_layout(); save_figure(fig, output / "global_scale_comparison.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 8)); image = ax.imshow(norm, vmin=0, vmax=1, cmap="YlOrRd")
    ax.set_xticks(range(7), SYSTEMS, rotation=45, ha="right"); ax.set_yticks(range(7), SYSTEMS)
    ax.set_xlabel("Target system (soft keyword/topic mapping)"); ax.set_ylabel("Source system")
    for i in range(7):
        for j in range(7): ax.text(j, i, f"{norm[i,j]:.3f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Max-normalized computed weight mass")
    fig.tight_layout(); save_figure(fig, output / "seven_system_coupling.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, color in [("within", "tab:orange"), ("between", "tab:blue")]:
        values = stats["within_between"][key]["weights"]
        if values: ax.hist(values, bins=np.linspace(0,1,21), alpha=.6, color=color, label=f"{key}; n={len(values)}; mean={np.mean(values):.3f}")
    ax.set_xlabel("Equal-third uncalibrated fusion score"); ax.set_ylabel("Number of directed pairs")
    ax.set_title("Automatic-community grouping; separate from seven-system mapping")
    if any(stats["within_between"][key]["weights"] for key in ["within", "between"]):
        ax.legend()
    else:
        ax.text(.5, .5, "No within/between analysis-edge observations\nCommunity statistics are not evaluable", transform=ax.transAxes,
                ha="center", va="center")
    fig.tight_layout(); save_figure(fig, output / "automatic_within_between.png"); plt.close(fig)
    candidates = [c for c in cases if any(c["case_id"] in a["case_support"] for _, _, a in graph.edges(data=True))]
    chosen = next((c for c in candidates if any(c["case_id"] in e["case_support"] for e in bridges)), candidates[0] if candidates else None)
    if chosen:
        cid = chosen["case_id"]
        graphs = []
        for g in [base_graph, graph]:
            sub = nx.DiGraph()
            for s, t, a in g.edges(data=True):
                if cid in a["case_support"]: sub.add_edge(s, t)
            graphs.append(sub)
        union = nx.compose(*graphs)
        # A shared circular layout keeps all labels separate and makes the
        # before/after node identities directly comparable for this small case.
        pos = nx.circular_layout(sorted(union))
        labels = {n: (n.replace("mechanism_", "M:") if n.startswith("mechanism_") else f"N{i+1}") for i,n in enumerate(sorted(union))}
        fig, axes = plt.subplots(1,2, figsize=(16,7))
        for ax,g,title in zip(axes,graphs,["Basic analysis context + candidates", "Enhanced with inherited mechanism bridges"]):
            nx.draw_networkx(g, pos, ax=ax, labels={n:labels[n] for n in g}, node_size=750,
                             node_color=["#ffdd55" if n.startswith("mechanism_") else "#8dc7ec" for n in g], font_size=7, arrowsize=14)
            ax.set_title(f"{title}\n{cid}: {len(g)} nodes / {g.number_of_edges()} pairs"); ax.axis("off")
        fig.suptitle("One model-qualified occurrence case illustration; links are unverified contextual candidates" if support_mode == "model_occurrences" else "One exact-literal-grounded case illustration; links are unverified contextual candidates")
        fig.tight_layout(); save_figure(fig, output / "case_base_and_enhanced.png"); plt.close(fig)
        csv_write(output / "example_node_lookup.csv", [{"plot_label": labels[n], "node_id":n,"name":enhanced_nodes[n]["name"],"type":enhanced_nodes[n]["entity_type"]} for n in union])
        dump(output / "example_case.json", {"case":chosen,"base_nodes":list(graphs[0]),"base_edges":list(graphs[0].edges),"enhanced_nodes":list(graphs[1]),"enhanced_edges":list(graphs[1].edges)})
    fig, ax = plt.subplots(figsize=(9,5))
    names = ["linear_distinct_three_node_walks", "branching_neighbor_pairs", "convergent_neighbor_pairs"]
    ax.bar(["Linear 3-node motifs", "Branching pairs", "Convergent pairs"], [pattern_counts[k] for k in names])
    ax.set_ylabel("Computed static motif counts"); ax.set_title("Nonexclusive topology patterns; not measured temporal evolution")
    fig.tight_layout(); save_figure(fig, output / "static_patterns.png"); plt.close(fig)


def run(export, cases, topic_vectors, system_vectors, output, topic_sources, args):
    started = time.perf_counter(); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    support_mode = getattr(args, "support_mode", "literal")
    bn, be, en, ae, bridges, augmented = build_layers(export, cases, topic_vectors, system_vectors, support_mode)
    base_graph, graph = projection(bn, ae), projection(en, augmented)
    assigned, stats = community_analysis(graph)
    matrix, norm, coupling_stats = system_coupling(graph, en)
    base_paths, base_completion, base_path_stats = enumerate_case_paths(base_graph, cases, args.max_case_paths, args.max_case_seconds, support_mode)
    enhanced_paths, enhanced_completion, enhanced_path_stats = enumerate_case_paths(graph, cases, args.max_case_paths, args.max_case_seconds, support_mode)
    scope_levels = {"base":path_scope_levels(base_paths,en,ae),"enhanced":path_scope_levels(enhanced_paths,en,augmented)}
    base_candidates, base_filter_audit, base_candidate_stats = propagation_candidates(base_paths,base_graph,en,support_mode)
    enhanced_candidates, enhanced_filter_audit, enhanced_candidate_stats = propagation_candidates(enhanced_paths,graph,en,support_mode)
    pattern_counts = count_patterns(graph)
    abl = []
    for name, coeff in VARIANTS.items():
        g = projection(en, augmented, coeff); ca, st = community_analysis(g)
        cm, cn, cs = system_coupling(g, en)
        abl.append({"variant":name,"semantic_coefficient":coeff[0],"cooccurrence_coefficient":coeff[1],"physical_coefficient":coeff[2],
                    "fixed_nodes":len(en),"fixed_directed_pairs":g.number_of_edges(),"auto_communities":st["n_auto_communities"],
                    "modularity":st["modularity"],"within_n":st["within_between"]["within"]["n_directed_pairs"],
                    "within_mean":st["within_between"]["within"]["mean"],"between_n":st["within_between"]["between"]["n_directed_pairs"],
                    "between_mean":st["within_between"]["between"]["mean"],"unassigned_pair_n":st["within_between"]["unassigned"]["n_directed_pairs"]})
    dump(output / "base_graph.json", {"nodes":list(bn.values()),"edges":be,"analysis_edges":ae,"all_basic_relations_retained":True,"support_mode":support_mode})
    dump(output / "enhanced_graph.json", {"nodes":list(en.values()),"basic_edges":be,"mechanism_edges":bridges,"analysis_edges":augmented,"support_mode":support_mode})
    dump(output / "mechanism_creation_rules.json", {"dictionary":MECHANISMS,"fault_types":sorted(FAULT_TYPES),"physical_types":sorted(PHYSICAL_TYPES),
        "rules":["fault entity name contains a literal dictionary term -> fault-to-mechanism membership context",
                 "existing causal/structural/spatial physical-fault base connection and literal mechanism in fault name -> physical-to-mechanism context",
                 "source fault contains mechanism and existing causal/influence source-to-target relation -> inherited mechanism-to-target candidate"],
        "no_random_edges":True,"no_paper_value_filling":True,"new_rules_not_recovered_author_program":True,
        "support_mode":support_mode,"model_case_expansion_allowed":False if support_mode=="model_occurrences" else None})
    dump(output / "automatic_communities.json", {"node_community":assigned,"stats":stats})
    dump(output / "system_coupling.json", {"raw_weight_mass":matrix.tolist(),"max_normalized":norm.tolist(),**coupling_stats})
    csv_write(output / "system_coupling_raw.csv", [{"source_system":SYSTEMS[i],**{SYSTEMS[j]:matrix[i,j] for j in range(7)}} for i in range(7)])
    csv_write(output / "ablation_metrics.csv", abl)
    for label, records, completion in [("base",base_paths,base_completion),("enhanced",enhanced_paths,enhanced_completion)]:
        with (output / (label + "_case_grounded_paths.jsonl")).open("w",encoding="utf-8") as f:
            for record in records:f.write(json.dumps(record,ensure_ascii=False)+"\n")
        csv_write(output / (label + "_case_path_completion.csv"),completion)
    for label, candidates, audit in [("base",base_candidates,base_filter_audit),("enhanced",enhanced_candidates,enhanced_filter_audit)]:
        for suffix, records in [("propagation_candidate_paths",candidates),("propagation_path_filter_audit",audit)]:
            with (output / (label+"_"+suffix+".jsonl")).open("w",encoding="utf-8") as f:
                for record in records:f.write(json.dumps(record,ensure_ascii=False)+"\n")
    chains = [p for p in enhanced_paths if p["n_nodes"] == 3]
    csv_write(output / "top10_three_node_chains.csv", [{"rank":i+1, "node_ids":p["node_ids"],"chain":" -> ".join(en[n]["name"] for n in p["node_ids"]),
        "literal_case_frequency":p["literal_case_frequency"],"unique_body_support_frequency":p["unique_body_support_frequency"],
        "average_edge_weight":p["average_edge_weight"],"case_ids":p["case_ids"],
        "support_kind":("input_model_qualified_occurrence_cases_on_existing_or_inherited_links_not_observed_causal_chain" if support_mode=="model_occurrences" else "literal_comention_on_existing_or_inherited_links_not_observed_causal_chain")} for i,p in enumerate(chains[:10])],
        ["rank","node_ids","chain","literal_case_frequency","unique_body_support_frequency","average_edge_weight","case_ids","support_kind"])
    qualified_chains = [p for p in enhanced_candidates if p["n_nodes"]>=3]
    csv_write(output / "propagation_candidate_chains.csv", [{"rank":i+1,"n_nodes":p["n_nodes"],"node_ids":p["node_ids"],
        "chain":" -> ".join(en[n]["name"] for n in p["node_ids"]),"literal_case_frequency":p["literal_case_frequency"],
        "unique_body_support_frequency":p["unique_body_support_frequency"],"average_edge_weight":p["average_edge_weight"],
        "case_ids":p["case_ids"],"edge_qualification":p["edge_qualification"],"filter_rule_version":p["filter_rule_version"],
        "support_kind":("input_model_qualified_occurrence_case_support_not_observed_causal_chain" if support_mode=="model_occurrences" else "new_literal_case_support_not_observed_causal_chain")} for i,p in enumerate(qualified_chains[:10])],
        ["rank","n_nodes","node_ids","chain","literal_case_frequency","unique_body_support_frequency","average_edge_weight","case_ids","edge_qualification","filter_rule_version","support_kind"])
    evidence = {"semantic_coverage":{"mechanism_classes":len(en)-len(bn),"entities_with_literal_case_support":sum(bool(n["case_support"]) for n in en.values())},
                "path_support":{"before":base_path_stats,"after":enhanced_path_stats},"coupling":coupling_stats,
                "diagnostic_accuracy":None,"SIS":None,"expert_ratings_available":False}
    dump(output / "interpretability_evidence_only.json",evidence)
    csv_write(output / "expert_score_template.csv", [{"subindex":s,"paper_weight":w,"expert_score_0_to_10":"","rater":"","rating_evidence":"","rating_date":""} for s,w in [("SSC",.3),("SPC",.25),("SCC",.25),("SDA",.2)]])
    save_plots(output,bn,en,be,bridges,base_graph,graph,cases,matrix,norm,stats,pattern_counts,support_mode)
    summary = {"status":"completed_under_disclosed_author_framework_completion_rules_not_exact_original_program",
        "n_cases_original":len(cases),"n_unique_case_texts":len({c["text"] for c in cases}),
        "base_full_KG":{"nodes":len(bn),"typed_edges":len(be)},"enhanced_full_KG":{"nodes":len(en),"typed_edges":len(be)+len(bridges)},
        "base_analysis_projection":{"nodes":len(base_graph),"directed_pairs":base_graph.number_of_edges(),"typed_analysis_edges":len(ae)},
        "enhanced_analysis_projection":{"nodes":len(graph),"directed_pairs":graph.number_of_edges(),"typed_analysis_edges":len(augmented)},
        "analysis_relation_kinds":dict(collections.Counter(e["kind"] for e in ae)),"metadata_relation_counts":dict(collections.Counter(e["relation_type"] for e in be if e["analysis_role"].startswith("metadata"))),
        "added_mechanism_nodes":len(en)-len(bn),"added_mechanism_edges":len(bridges),"automatic_communities":{k:v for k,v in stats.items() if k!="within_between"},
        "auto_within_between":{k:{j:v for j,v in stat.items() if j!="weights"} for k,stat in stats["within_between"].items()},
        "seven_system_coupling":coupling_stats,"base_reachability":reachability(base_graph),"enhanced_reachability":reachability(graph),
        "case_path_statistics":{"before":base_path_stats,"after":enhanced_path_stats},"patterns":pattern_counts,
        "propagation_candidate_paths":{"before":base_candidate_stats,"after":enhanced_candidate_stats},
        "path_scope_levels":scope_levels,
        "zero_topic_support_nodes":sum(not n["case_support"] for n in en.values()),"grounding":("input model-qualified occurrence case IDs; no full-corpus same-name expansion; not observed causal chain" if support_mode=="model_occurrences" else "new literal matching, not recovered historical case provenance or observed causal chain"),
        "support_mode":support_mode,
        "fixed_coefficients":dict(zip(["semantic","cooccurrence","physical"],VARIANTS["full"])),
        "path_count_complete":base_path_stats["n_cases_incomplete"]==enhanced_path_stats["n_cases_incomplete"]==0,
        "SIS":None,"root_cause_accuracy":None,"diagnostic_time_improvement":None,
        "propagation_evaluation_status": ("not_evaluable_no_qualified_propagation_candidates" if not enhanced_candidates else "candidate_structure_computed_without_independent_causal_gold"),
        "no_propagation_edges_added_to_rescue_empty_projection": True,
        "seed":42,"topic_sources":topic_sources,"elapsed_seconds":time.perf_counter()-started,
        "limitations":["Original relation quality has not been established by independent expert gold.","Mechanism bridge rules and equal fusion coefficients are disclosed completion assumptions, not recovered author code.",
                       "Physical feature is explicit relation support, not measured physical propagation strength.",("Input model-qualified occurrence case support is not observed causal sequencing, calibrated relation existence or temporal fault propagation." if support_mode=="model_occurrences" else "Literal case co-mention is not observed causal sequencing or temporal fault propagation."),
                       "Automatic communities and seven-system keyword/topic mappings are different analyses; seven communities are not forced.",
                       "SCC reachable endpoint-pair counts are not simple-path counts; all-case simple paths are separately enumerated."]}
    dump(output / "analysis_summary.json",summary)
    dump(output / "run_summary.json",summary)
    return summary



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topics-dir",type=Path)
    parser.add_argument("--topic-model-dir",type=Path,help="Explicit fitted model directory containing T01..TK document_topic.csv and seven-system probabilities; default is paper_fixed_K07")
    parser.add_argument("--support-mode",choices=["literal","model_occurrences"],default="literal",help="literal preserves the historical-export analysis; model_occurrences strictly uses input node/edge case_ids without whole-corpus same-name expansion")
    parser.add_argument("--export-json",type=Path)
    parser.add_argument("--cases",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--max-case-paths",type=int,default=0,help="Explicit resource cap; 0 means uncapped. Any reached cap marks incomplete.")
    parser.add_argument("--max-case-seconds",type=float,default=0,help="Explicit resource safeguard, not a hop cap; 0 means uncapped.")
    args=parser.parse_args()
    if not all([args.topics_dir or args.topic_model_dir,args.export_json,args.cases]):
        parser.error("--topics-dir or --topic-model-dir, --export-json and --cases are required for full run")
    export=json.loads(args.export_json.read_text());cases=rows(args.cases)
    if len({c["case_id"] for c in cases})!=len(cases):raise ValueError("Case IDs must be unique")
    tv,sv,topic_sources=load_topics(args.topics_dir or args.topic_model_dir,cases,args.topic_model_dir)
    summary=run(export,cases,tv,sv,args.output,topic_sources,args)
    dump(args.output/"run_config.json",{"command":__import__('sys').argv,"seed":42,"input_sha256":{"export":sha(args.export_json),"cases":sha(args.cases),"script":sha(__file__)},
        "arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},"fixed_K":len(next(iter(tv.values()))),"engineering_system_count":len(SYSTEMS),"fixed_fusion_coefficients":VARIANTS["full"],
        "relation_mapping":RELATIONS,"all_basic_nodes_and_relations_retained":True,"new_mechanism_rules_file":"mechanism_creation_rules.json","libraries":{"numpy":np.__version__,"networkx":nx.__version__,"matplotlib":matplotlib.__version__}})
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
