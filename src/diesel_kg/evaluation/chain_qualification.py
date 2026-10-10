#!/usr/bin/env python3
"""Case-local relation qualification for grouped graph Table1-style candidates.

Global typed-kind unions are valid candidate-graph projection metadata, but a
case cannot borrow a causal/influence type seen only in a different case.
This independent stage preserves earlier co-support exports and writes a new
case-role-qualified candidate set, still explicitly not observed causal gold.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import networkx as nx
import numpy as np

from diesel_kg.evaluation.engineering import path_qualified
from diesel_kg.graph.ablation import sha, read_rows, dump, write_rows, csv_write, normalized_body


def qualify_case_path(ids, cid, nodes, edge_rows):
    """Use ONLY typed rows whose actual support includes cid on every link."""
    graph = nx.DiGraph()
    links = []
    for u, v in zip(ids, ids[1:]):
        supported = [e for e in edge_rows[u, v] if cid in e['case_support']]
        if not supported:
            return None
        kinds = sorted({e['kind'] for e in supported})
        weight = float(np.mean([e['weight'] for e in supported]))
        graph.add_edge(u, v, kinds=kinds, weight=weight)
        links.append({'source': u, 'target': v, 'case_local_analysis_edge_ids': [e['id'] for e in supported],
            'case_local_relation_kinds': kinds, 'case_local_basis_candidate_edge_ids':
            sorted({b for e in supported for b in e['basis_base_edge_ids']}),
            'case_local_mean_association_weight': weight})
    if not path_qualified(ids, nodes, graph):
        return None
    return links


def audit_variant(folder, train_cases, output):
    graph_file = folder / 'graph.json'
    chain_file = folder / 'three_node_candidate_chains_with_case_quotes.jsonl'
    data = json.loads(graph_file.read_text())
    nodes = {n['id']: n for n in data['nodes']}
    edge_rows = defaultdict(list)
    for e in data['analysis_edges']:
        if not set(e['case_support']) <= train_cases.keys():
            raise ValueError('Heldout case support in graph supplied to chain audit')
        edge_rows[e['source'], e['target']].append(e)
    chains = read_rows(chain_file)
    retained, reviewed = [], []
    removed_supports = 0
    for chain in chains:
        ids = chain['node_ids']
        citations = {c['case_id']: c for c in chain['citations']}
        accepted_citations = []
        for cid in chain['supporting_train_case_ids']:
            if cid not in train_cases or cid not in citations:
                raise ValueError('Candidate chain citation missing or outside train cases')
            evidence = qualify_case_path(ids, cid, nodes, edge_rows)
            reviewed.append({'node_ids': ids, 'case_id': cid, 'qualified_by_case_local_relation_kinds': evidence is not None,
                'reason': 'supported_case_local_relation_roles' if evidence is not None else 'global_kind_union_not_qualified_in_this_case',
                'case_local_edge_roles': evidence, 'human_causal_gold': False})
            if evidence is None:
                removed_supports += 1
                continue
            citation = dict(citations[cid])
            if citation['full_case_quote'] != train_cases[cid]['text']:
                raise ValueError('Full case citation changed')
            for node in citation['node_evidence']:
                for q in node['quotes']:
                    if train_cases[cid]['text'][slice(*q['span'])] != q['quote']:
                        raise ValueError('Node quote offset mismatch')
            citation['case_local_edge_roles'] = evidence
            citation['case_local_path_qualified'] = True
            accepted_citations.append(citation)
        if not accepted_citations:
            continue
        retained_chain = dict(chain)
        retained_chain.update(supporting_train_case_ids=[c['case_id'] for c in accepted_citations],
            n_train_numbered_case_supports=len(accepted_citations),
            n_train_unique_body_supports=len({normalized_body(train_cases[c['case_id']]['text']) for c in accepted_citations}),
            citations=accepted_citations, qualification_scope='actual per-case supported typed relation kinds',
            case_local_role_audit_passed=True, candidate_chain_not_causal_gold=True)
        retained.append(retained_chain)
    retained.sort(key=lambda r: (-r['n_train_unique_body_supports'], -r['mean_association_weight'], r['node_ids']))
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / 'case_local_candidate_chains_with_quotes.jsonl', retained)
    write_rows(output / 'every_chain_case_role_review.jsonl', reviewed)
    csv_write(output / 'table1_style_top10_candidates.csv', ({key: row[key] for key in ['node_ids', 'node_names',
        'n_train_numbered_case_supports', 'n_train_unique_body_supports', 'mean_association_weight',
        'product_association_score', 'supporting_train_case_ids', 'candidate_chain_not_causal_gold',
        'hop_policy', 'qualification_scope']} for row in retained[:10]))
    summary = {'status': 'completed_case_local_relation_role_candidate_audit',
        'n_source_global_candidate_chains': len(chains), 'n_retained_case_local_candidate_chains': len(retained),
        'n_source_candidate_case_incidences': len(reviewed),
        'n_rejected_case_incidences_without_local_qualifying_roles': removed_supports,
        'n_retained_case_local_candidate_incidences': sum(r['n_train_numbered_case_supports'] for r in retained),
        'source_graph_sha256': sha(graph_file), 'source_chain_sha256': sha(chain_file),
        'all_citations_directly_checked': True, 'borrowed_relation_roles_across_cases': False,
        'original_co_support_exports_preserved': True, 'causal_accuracy': None,
        'path_completeness_rate': None, 'original_paper_table1_reproduced': False,
        'limits': ['Only exactly three-node candidate chains are audited.',
            'Case-local model relation support is still not reviewed causal evidence.',
            'Frequency is unique body/case candidate support, not measured temporal propagation occurrence.']}
    dump(output / 'analysis_summary.json', summary)
    return summary


def run(graph_ablation, output):
    start = time.perf_counter()
    graph_ablation, output = Path(graph_ablation).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Audit output must be new/empty')
    main = json.loads((graph_ablation / 'analysis_summary.json').read_text())
    if main.get('status') != 'completed_grouped_heldout_silver_retrieval_ablation':
        raise ValueError('Graph ablation producer is not completed')
    train_cases = {c['case_id']: c for c in read_rows(graph_ablation / 'train_cases.jsonl')}
    splits = read_rows(graph_ablation / 'case_group_split.jsonl')
    expected = {cid for row in splits if row['split'] == 'train' for cid in row['case_ids']}
    if set(train_cases) != expected:
        raise ValueError('Train case file and split manifest disagree')
    results = {name: audit_variant(graph_ablation / name, train_cases, output / name)
               for name in ['full_fusion__no_mechanisms', 'full_fusion__mechanisms']}
    dump(output / 'analysis_summary.json', {'status': 'completed', 'variants': results,
        'argv': sys.argv, 'python': sys.executable, 'cwd': str(Path.cwd()),
        'script_path': __file__, 'script_sha256': sha(__file__),
        'upstream_analysis_summary_sha256': sha(graph_ablation / 'analysis_summary.json'),
        'train_cases_sha256': sha(graph_ablation / 'train_cases.jsonl'),
        'split_manifest_sha256': sha(graph_ablation / 'case_group_split.jsonl'),
        'elapsed_seconds': time.perf_counter() - start})
    print(json.dumps(results, ensure_ascii=False), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph-ablation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.graph_ablation, args.output)
