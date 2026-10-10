#!/usr/bin/env python3
"""Grouped held-out graph retrieval ablations on frozen model candidates.

This is a new experiment, not a recovery of the original paper's numbers.
Case-body groups are isolated before graph, NER lexicon and LDA construction.
The supplied NER/RE checkpoints remain fixed; their supervised training-source
independence is unproven. Text/NER-derived references are explicitly SILVER.
No no_relation labels, physical gold, causal gold, SIS or event times are made up.
"""
from __future__ import annotations
from diesel_kg.paths import PROJECT_ROOT, resolve_script

import argparse
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
import time
import unicodedata

import gensim
from gensim.corpora import Dictionary
import networkx as nx
import numpy as np

from diesel_kg.data.neo4j import stable_id
from diesel_kg.graph.topics import Settings, clean_case_text, vocabulary_terms, tokenize_cases, fit_lda, document_topic_matrix, topic_keywords, system_mapping
from diesel_kg.graph.analysis import build_layers, community_analysis
from diesel_kg.evaluation.engineering import infer_query, path_qualified, ROOT_TYPES, SYMPTOM_TYPES, RECOVERY, paired_bootstrap

HERE = PROJECT_ROOT
SOURCE_MODULES = [__file__, resolve_script('diesel_kg.data.neo4j', HERE), resolve_script('diesel_kg.graph.topics', HERE),
                  resolve_script('diesel_kg.graph.analysis', HERE), resolve_script('diesel_kg.evaluation.engineering', HERE)]
WEIGHT_MODES = ('base_unweighted', 'semantic_only', 'cooccurrence_only',
                'physical_proxy_only', 'full_fusion')
# This is a conservative, disclosed textual boundary; not human event annotation.
QUERY_BOUNDARY = re.compile(r'检查发现|经检查|检查|检测|拆检|排查|发现|经查|更换|清洗|调整|维修|修复|处理|修理|检修|原因为|原因是|原因在于|由于|因为|故障原因|诊断|确认')
DIAGNOSIS = re.compile(r'检查发现|经检查|检查|检测|拆检|排查|发现|经查|原因为|原因是|原因在于|由于|因为|故障原因|诊断|确认')
INTERVENTION = re.compile(r'更换|清洗|调整|维修|修复|处理|修理|检修|拆卸|重新安装|加注|补充|清除')
SILVER_TYPES = {'故障原因', '故障状态', '故障部位'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def write_rows(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', encoding='utf-8') as f:
        for row in values:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def csv_write(path, values, fields=None):
    values = list(values)
    fields = fields or (list(values[0]) if values else [])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in values:
            w.writerow({k: json.dumps(row.get(k), ensure_ascii=False)
                        if isinstance(row.get(k), (list, dict)) else row.get(k) for k in fields})


def normalized_body(text):
    _, body = clean_case_text(text)
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', body))


def grouped_split(cases, seed=42, forced_splits=None):
    """Order-independent split; exact or NFKC/whitespace duplicates stay together."""
    groups = defaultdict(list)
    for case in cases:
        normalized = normalized_body(case['text'])
        if not normalized:
            raise ValueError('Empty normalized case body')
        gid = 'body_' + hashlib.sha256(normalized.encode('utf-8')).hexdigest()
        groups[gid].append(case['case_id'])
    keys = sorted(groups)
    if len(keys) < 7:
        raise ValueError('At least 7 unique case-body groups are required')
    case_group = {cid: g for g, ids in groups.items() for cid in ids}
    forced = {}
    for split, ids in (forced_splits or {}).items():
        if split not in {'train', 'dev', 'test'}:
            raise ValueError('Unknown forced split')
        for cid in ids:
            if cid not in case_group:
                raise ValueError('Unknown forced case ID: ' + cid)
            gid = case_group[cid]
            if gid in forced and forced[gid] != split:
                raise ValueError('Conflicting forced splits inside normalized body group')
            forced[gid] = split
    remaining = [g for g in keys if g not in forced]
    shuffled = list(np.random.default_rng(seed).permutation(remaining))
    n_train = int(math.floor(.70 * len(remaining)))
    n_dev = int(math.floor(.15 * len(remaining)))
    assignments = {g: ('train' if i < n_train else 'dev' if i < n_train + n_dev else 'test')
                   for i, g in enumerate(shuffled)}
    assignments.update(forced)
    rows = [{'group_id': g, 'split': assignments[g], 'case_ids': sorted(groups[g]),
             'representative_case_id': min(groups[g]), 'n_numbered_cases': len(groups[g]),
             'assignment_source': 'fixed_domain_adaptation_partition_constraint' if g in forced else 'seed42_random_remaining_groups'}
            for g in keys]
    by_case = {cid: row for row in rows for cid in row['case_ids']}
    if len(by_case) != len(cases):
        raise ValueError('Duplicate case IDs')
    return rows, by_case


def validate_inputs(cases, entities, triples, candidate_graph):
    by_case = {c['case_id']: c for c in cases}
    by_ner = {e['case_id']: e for e in entities}
    if len(by_case) != len(cases) or len(by_ner) != len(entities) or set(by_case) != set(by_ner):
        raise ValueError('Case and NER IDs must be complete and unique')
    occurrences = {}
    for cid, row in by_ner.items():
        if row['text'] != by_case[cid]['text']:
            raise ValueError('Case text differs from frozen occurrence text: ' + cid)
        for e in row['entities']:
            if not (isinstance(e['start'], int) and isinstance(e['end'], int)
                    and 0 <= e['start'] < e['end'] <= len(row['text'])
                    and row['text'][e['start']:e['end']] == e['text']):
                raise ValueError('Invalid occurrence span')
            oid = e['occurrence_id']
            if oid in occurrences:
                raise ValueError('Duplicate occurrence ID')
            occurrences[oid] = (cid, e)
    nodes = {n['id']: n for n in candidate_graph['nodes']}
    edges = {e['id']: e for e in candidate_graph['edges']}
    counts = Counter()
    supports = defaultdict(set)
    probabilities = defaultdict(float)
    node_supports = defaultdict(set)
    for row in triples:
        cid = row['case_id']
        if cid not in by_case or row.get('causal_gold') is not False:
            raise ValueError('Unknown case or improperly declared candidate gold')
        ids = []
        for e in [row['head'], row['tail']]:
            if occurrences.get(e['occurrence_id']) != (cid, e):
                raise ValueError('Triple endpoint does not match frozen NER occurrence')
            nid = stable_id('n_', [e['type'], e['text']])
            if nid not in nodes:
                raise ValueError('Triple endpoint missing from candidate graph')
            ids.append(nid)
            node_supports[nid].add(cid)
        eid = stable_id('e_', [ids[0], row['relation'], ids[1]])
        if eid not in edges:
            raise ValueError('Triple relation missing from candidate graph')
        probability = row['probability']
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Nonfinite or invalid candidate probability')
        counts[eid] += 1
        probabilities[eid] += probability
        supports[eid].add(cid)
    if set(counts) != set(edges) or set(node_supports) != set(nodes):
        raise ValueError('Saved triples do not recover complete candidate graph')
    for eid, count in counts.items():
        edge = edges[eid]
        if (count != edge['n_occurrence_observations'] or supports[eid] != set(edge['case_ids'])
                or not math.isclose(probabilities[eid] / count, edge['mean_probability'], abs_tol=1e-12)):
            raise ValueError('Candidate graph support/probability does not match saved triples')
    if any(node_supports[nid] != set(nodes[nid]['case_ids']) for nid in nodes):
        raise ValueError('Candidate node support mismatch')
    return {'complete_case_and_NER_alignment': True, 'all_occurrence_offsets_verified': True,
            'candidate_graph_recovered_from_actual_triples': True,
            'n_cases': len(cases), 'n_occurrences': len(occurrences),
            'n_candidate_observations': len(triples), 'n_candidate_typed_edges': len(edges)}


def aggregate_train_graph(triples, train_ids):
    """Build from TRAIN rows afresh, never trim globally aggregated means/mentions."""
    nodes, edges = {}, {}
    n_observations = 0
    for row in triples:
        cid = row['case_id']
        if cid not in train_ids:
            continue
        ids = []
        for e in [row['head'], row['tail']]:
            nid = stable_id('n_', [e['type'], e['text']])
            ids.append(nid)
            node = nodes.setdefault(nid, {'id': nid, 'name': e['text'], 'entity_type': e['type'],
                'analysis_eligible': True, 'case_ids': set(), 'mention_ids': set()})
            node['case_ids'].add(cid)
            node['mention_ids'].add(e['occurrence_id'])
        eid = stable_id('e_', [ids[0], row['relation'], ids[1]])
        edge = edges.setdefault(eid, {'id': eid, 'source': ids[0], 'target': ids[1],
            'relation_type': row['relation'], 'analysis_eligible': True, 'case_ids': set(),
            'n_occurrence_observations': 0, 'probability_sum': 0., 'causal_gold': False})
        edge['case_ids'].add(cid)
        edge['n_occurrence_observations'] += 1
        edge['probability_sum'] += row['probability']
        n_observations += 1
    for n in nodes.values():
        n['case_ids'] = sorted(n['case_ids'])
        n['mention_ids'] = sorted(n['mention_ids'])
    for e in edges.values():
        e['case_ids'] = sorted(e['case_ids'])
        e['mean_probability'] = e.pop('probability_sum') / e['n_occurrence_observations']
    return {'nodes': [nodes[n] for n in sorted(nodes)], 'edges': [edges[e] for e in sorted(edges)],
            'schema': {'support_mode': 'model_occurrences', 'split': 'train',
                       'confidence_threshold': 0, 'causal_gold': False,
                       'n_train_candidate_observations': n_observations}}


def fit_train_topics(train_cases, train_entities, by_case, output, settings):
    train_ids = {c['case_id'] for c in train_cases}
    if ({r['case_id'] for r in train_entities} != train_ids
            or any(by_case[cid]['split'] != 'train' for cid in train_ids)):
        raise ValueError('LDA/NER lexicon attempted to use non-training or incomplete cases')
    reps = sorted({by_case[cid]['representative_case_id'] for cid in train_ids})
    lookup = {c['case_id']: c for c in train_cases}
    unique_cases = [lookup[cid] for cid in reps]
    terms, reasons, accepted, rejected = vocabulary_terms(train_entities)
    documents = tokenize_cases(unique_cases, terms)
    dictionary = Dictionary(documents)
    dictionary.filter_extremes(no_below=settings.min_df, no_above=1.0, keep_n=None)
    dictionary.compactify()
    if len(dictionary) < 3:
        raise ValueError('Insufficient training vocabulary')
    retained = set(dictionary.token2id)
    filtered = [[w for w in words if w in retained] for words in documents]
    corpus = [dictionary.doc2bow(words) for words in filtered]
    model, metric, topic_cv = fit_lda(settings.fixed_paper_k, dictionary, corpus, filtered, settings)
    posteriors = document_topic_matrix(model, corpus)
    keywords = topic_keywords(model, min(settings.keyword_topn, len(dictionary)))
    affinities, mapping, anchors = system_mapping(keywords, settings)
    group_topics = {by_case[cid]['group_id']: v for cid, v in zip(reps, posteriors)}
    topic_vectors = {cid: group_topics[by_case[cid]['group_id']] for cid in sorted(train_ids)}
    system_vectors = {cid: topic_vectors[cid] @ affinities for cid in sorted(train_ids)}
    output.mkdir(parents=True, exist_ok=True)
    dictionary.save(str(output / 'dictionary.gensim'))
    model.save(str(output / 'lda.model'), separately=[])
    write_rows(output / 'train_unique_group_tokens.jsonl', ({'case_id': cid,
        'group_id': by_case[cid]['group_id'], 'tokens_before_min_df': words,
        'tokens_after_min_df': retained_words} for cid, words, retained_words in zip(reps, documents, filtered)))
    csv_write(output / 'professional_lexicon.csv', ({'term': term,
        'sources': sorted(reasons[term]), 'n_train_NER_mentions': accepted[term]} for term in sorted(terms)))
    csv_write(output / 'train_document_topic.csv', ({'case_id': cid,
        'group_id': by_case[cid]['group_id'], **{f'T{i+1:02d}': float(p) for i, p in enumerate(v)}}
        for cid, v in zip(reps, posteriors)))
    dump(output / 'topic_keywords.json', {'topics': [[{'word': w, 'probability': p} for w, p in words]
         for words in keywords], 'c_v_by_topic': topic_cv, 'system_mapping': mapping,
         'mapping_basis': 'fixed engineering anchors; not physical-system gold'})
    csv_write(output / 'system_anchor_matches.csv', anchors)
    summary = {'status': 'fit_on_train_unique_case_bodies_only', 'training_case_ids': sorted(train_ids),
        'training_unique_group_ids': sorted(group_topics), 'representative_case_ids': reps,
        'n_numbered_train_cases': len(train_ids), 'n_unique_training_groups': len(reps),
        'n_professional_lexicon_terms': len(terms), 'vocabulary_size': len(dictionary),
        'n_rejected_NER_mentions': sum(rejected.values()), 'lda': metric,
        'topic_number_selection': 'K7 fixed before evaluation from paper; not selected on dev/test',
        'training_dict_and_topic_fit_used_heldout_text': False,
        'fixed_dictionary_is_external_prior': True, 'unique_groups_equal_weight': True}
    dump(output / 'analysis_summary.json', summary)
    return topic_vectors, system_vectors, group_topics, summary


def grouped_features(nodes, edges, by_case, group_topics, system_vectors):
    """Count co-occurrence and mean topics once per unique body, not duplicate IDs."""
    for node in nodes.values():
        groups = sorted({by_case[cid]['group_id'] for cid in node['case_support']})
        node['training_group_support'] = groups
        node['topic_distribution'] = np.mean([group_topics[g] for g in groups], axis=0).tolist() if groups else [0.] * len(next(iter(group_topics.values())))
        representative_ids = [min(cid for cid in node['case_support'] if by_case[cid]['group_id'] == g) for g in groups]
        node['system_distribution'] = np.mean([system_vectors[cid] for cid in representative_ids], axis=0).tolist() if groups else [0.] * 7
        node['topic_vector_source'] = 'mean_train_unique_group_posteriors'
    for edge in edges:
        u, v = nodes[edge['source']], nodes[edge['target']]
        a, b = np.asarray(u['topic_distribution']), np.asarray(v['topic_distribution'])
        den = np.linalg.norm(a) * np.linalg.norm(b)
        edge['semantic'] = float(a @ b / den) if den else 0.
        su, sv = set(u['training_group_support']), set(v['training_group_support'])
        edge['cooccurrence'] = len(su & sv) / math.sqrt(len(su) * len(sv)) if su and sv else 0.
        edge['training_group_support'] = sorted({by_case[cid]['group_id'] for cid in edge['case_support']})
        edge['weight_feature_unit'] = 'unique training case-body group'
        edge['physical_feature'] = 'explicit candidate/input-relation support proxy, not measured physical strength'


def assert_train_support(nodes, edges, train_ids, by_case):
    if any(by_case[cid]['split'] != 'train' for cid in train_ids):
        raise ValueError('Train ID set contains heldout group')
    for row in list(nodes.values()) + list(edges):
        for key in ['case_ids', 'case_support']:
            if key in row and not set(row[key]) <= train_ids:
                raise ValueError('Heldout support in training graph: ' + row['id'])
        if 'training_group_support' in row:
            allowed = {by_case[cid]['group_id'] for cid in train_ids}
            if not set(row['training_group_support']) <= allowed:
                raise ValueError('Heldout group support in training graph')


def mode_weight(edge, mode):
    if mode == 'base_unweighted':
        return 1.
    if mode == 'semantic_only':
        return edge['semantic']
    if mode == 'cooccurrence_only':
        return edge['cooccurrence']
    if mode == 'physical_proxy_only':
        return edge['physical']
    if mode == 'full_fusion':
        return sum(edge[key] for key in ['semantic', 'cooccurrence', 'physical']) / 3
    raise ValueError('Unknown ablation')


def make_variant(nodes, edges, mode):
    weighted = []
    grouped = defaultdict(list)
    for original in edges:
        edge = copy.deepcopy(original)
        edge['weight'] = float(mode_weight(edge, mode))
        if not math.isfinite(edge['weight']) or not 0 <= edge['weight'] <= 1.000001:
            raise ValueError('Invalid ablation weight')
        edge['weight'] = min(edge['weight'], 1.)
        edge['weight_interpretation'] = 'association/ranking score, not causal probability'
        weighted.append(edge)
        if edge['source'] != edge['target']:
            grouped[edge['source'], edge['target']].append(edge)
    graph = nx.DiGraph()
    graph.graph['support_mode'] = 'train_group_model_occurrences'
    graph.add_nodes_from(nodes)
    for (u, v), rows in sorted(grouped.items()):
        weight = float(np.mean([r['weight'] for r in rows]))
        graph.add_edge(u, v, weight=weight, cost=-math.log(max(weight, 1e-15)),
            kinds=sorted({r['kind'] for r in rows}), edge_ids=[r['id'] for r in rows],
            case_support=sorted({cid for r in rows for cid in r['case_support']}),
            basis_base_edge_ids=sorted({b for r in rows for b in r['basis_base_edge_ids']}))
    return graph, weighted


def query_prefix(case):
    match = QUERY_BOUNDARY.search(case['text'])
    if match is None:
        return {'query_text': '', 'query_boundary_available': False,
                'query_boundary_rule': 'no recognized diagnosis/intervention boundary: abstain',
                'boundary_span': None, 'boundary_quote': None, 'boundary_is_human_verified': False}
    return {'query_text': case['text'][:match.start()], 'query_boundary_available': True,
            'query_boundary_rule': QUERY_BOUNDARY.pattern, 'boundary_span': [match.start(), match.end()],
            'boundary_quote': match.group(), 'boundary_is_human_verified': False}


def silver_reference(case, ner_row):
    """Save quoted diagnosis-window entities separately; never feed them to query."""
    diagnosis = DIAGNOSIS.search(case['text'])
    intervention = INTERVENTION.search(case['text'])
    end = intervention.start() if intervention else len(case['text'])
    if diagnosis is None or diagnosis.end() >= end:
        return {'status': 'no_rule_eligible_silver_reference', 'references': [], 'window': None,
                'human_gold': False, 'source': 'frozen NER plus disclosed text-window rule'}
    start = diagnosis.end()
    refs = []
    seen = set()
    for e in ner_row['entities']:
        if (e['type'] not in SILVER_TYPES or len(e['text']) < 2
                or not start <= e['start'] < e['end'] <= end or RECOVERY.search(e['text'])):
            continue
        if case['text'][e['start']:e['end']] != e['text']:
            raise ValueError('Silver evidence offset mismatch')
        nid = stable_id('n_', [e['type'], e['text']])
        if nid in seen:
            continue
        seen.add(nid)
        refs.append({'node_id': nid, 'name': e['text'], 'entity_type': e['type'],
            'span': [e['start'], e['end']], 'quote': case['text'][e['start']:e['end']],
            'NER_occurrence_id': e['occurrence_id'], 'NER_confidence': e.get('confidence'),
            'rule': 'ROOT typed NER occurrence fully inside diagnosis-before-first-intervention window; min length 2; exclude recovery wording',
            'is_human_gold': False, 'is_verified_causal_root': False})
    return {'status': 'silver_reference_available' if refs else 'no_rule_eligible_silver_reference',
        'references': refs, 'window': {'span': [start, end], 'quote': case['text'][start:end],
        'diagnosis_marker_span': [diagnosis.start(), diagnosis.end()], 'diagnosis_marker_quote': diagnosis.group(),
        'diagnosis_rule': DIAGNOSIS.pattern, 'intervention_rule': INTERVENTION.pattern},
        'human_gold': False, 'source': 'frozen NER plus disclosed text-window rule'}


def silver_scores(ranked_ids, target_ids):
    gold = set(target_ids)
    if not gold:
        return {**{f'silver_root_recall_at_{k}': None for k in (1, 3, 5)},
                **{f'silver_root_hit_at_{k}': None for k in (1, 3, 5)},
                'silver_root_mrr': None, 'silver_candidate_recall': None}
    result = {f'silver_root_recall_at_{k}': len(set(ranked_ids[:k]) & gold) / len(gold) for k in (1, 3, 5)}
    result.update({f'silver_root_hit_at_{k}': float(bool(set(ranked_ids[:k]) & gold)) for k in (1, 3, 5)})
    ranks = [i + 1 for i, nid in enumerate(ranked_ids) if nid in gold]
    result['silver_root_mrr'] = 1. / min(ranks) if ranks else 0.
    result['silver_candidate_recall'] = len(set(ranked_ids) & gold) / len(gold)
    return result


def evaluate_variant(name, nodes, graph, cases, by_case, ner_by_case, output, timing_repeats):
    root_universe = {nid for nid, node in nodes.items() if node['entity_type'] in ROOT_TYPES}
    rows = []
    for index, case in enumerate(cases):
        prefix = query_prefix(case)
        text = prefix['query_text']
        matches = []
        for nid, node in sorted(nodes.items()):
            if node['entity_type'] not in SYMPTOM_TYPES or not node['name'] or RECOVERY.search(node['name']):
                continue
            spans = [[m.start(), m.end()] for m in re.finditer(re.escape(node['name']), text)]
            if spans:
                matches.append({'node_id': nid, 'name': node['name'], 'query_spans': spans,
                                'quotes': [text[a:b] for a, b in spans]})
        symptoms = [m['node_id'] for m in matches]
        timings = []
        result = None
        for _ in range(timing_repeats):
            started = time.perf_counter()
            result = infer_query(symptoms, nodes, graph, root_universe)
            timings.append((time.perf_counter() - started) * 1000.)
        refs = silver_reference(case, ner_by_case[case['case_id']])
        targets = [r['node_id'] for r in refs['references']]
        ranked = [r['node_id'] for r in result['candidate_roots']]
        row = {'case_id': case['case_id'], 'group_id': by_case[case['case_id']]['group_id'],
            'split': by_case[case['case_id']]['split'], 'variant': name, **prefix,
            'query_symptoms': matches, 'query_has_graph_symptom': float(bool(symptoms)),
            'query_has_candidate_root': float(bool(ranked)), 'n_root_candidates': len(ranked),
            'machine_query_ms': float(np.median(timings)), 'timing_samples_ms': timings,
            'timing_includes': 'reverse constrained graph retrieval, ranking and candidate explanation generation; excludes text matching and silver extraction',
            'timing_is_engineer_localization_time': False, 'silver_reference': refs,
            'silver_reference_available': bool(targets),
            'silver_targets_in_train_graph_fraction': len(set(targets) & nodes.keys()) / len(targets) if targets else None,
            **silver_scores(ranked, targets), 'ranked_candidate_ids': ranked,
            'ranked_candidate_scores': [r['best_explanation']['product_association_score'] for r in result['candidate_roots']],
            'top5_candidate_explanations': result['candidate_roots'][:5],
            'core_nodes': result['core_nodes'], 'human_gold': False, 'causal_accuracy': None,
            'n_rejected_redundant_paths': result['n_rejected_best_or_first_context_paths']}
        rows.append(row)
        if (index + 1) % 100 == 0:
            print(f'{name}: heldout queries {index+1}/{len(cases)}', flush=True)
    write_rows(output / 'heldout_queries.jsonl', rows)
    summaries = {}
    for split in ['dev', 'test']:
        selected = [r for r in rows if r['split'] == split]
        summaries[split] = summarize_rows(selected)
    dump(output / 'retrieval_summary.json', summaries)
    return rows, summaries


def summarize_rows(rows):
    fields = ['query_has_graph_symptom', 'query_has_candidate_root', 'n_root_candidates', 'machine_query_ms']
    summary = paired_bootstrap(rows, fields, n_boot=2000, seed=42)
    silver = [r for r in rows if r['silver_reference_available']]
    silver_fields = ['silver_root_recall_at_1', 'silver_root_recall_at_3', 'silver_root_recall_at_5',
        'silver_root_hit_at_1', 'silver_root_hit_at_3', 'silver_root_hit_at_5', 'silver_root_mrr',
        'silver_candidate_recall', 'silver_targets_in_train_graph_fraction']
    summary['silver_reference_subset'] = paired_bootstrap(silver, silver_fields, n_boot=2000, seed=42)
    summary['silver_reference_subset']['evaluation_is_human_gold'] = False
    summary['silver_reference_subset']['reference_rule'] = 'diagnosis-window typed NER occurrences; direct quotes saved'
    summary['query_boundary_available_cases'] = sum(r['query_boundary_available'] for r in rows)
    summary['numbered_case_denominator'] = len(rows)
    summary['resampling_unit'] = 'NFKC+whitespace-normalized complete case-body groups; each group equal weight'
    summary['metrics_are_causal_or_industrial_accuracy'] = False
    return summary


def compare_variants(results):
    comparisons = []
    baseline = results['base_unweighted__no_mechanisms']
    metric_fields = ['query_has_graph_symptom', 'query_has_candidate_root', 'n_root_candidates', 'machine_query_ms',
        'silver_root_recall_at_1', 'silver_root_recall_at_3', 'silver_root_recall_at_5',
        'silver_root_hit_at_1', 'silver_root_hit_at_3', 'silver_root_hit_at_5', 'silver_root_mrr', 'silver_candidate_recall']
    lookup = {name: {r['case_id']: r for r in rows} for name, rows in results.items()}
    pairs = [('base_unweighted__no_mechanisms', name, 'versus_unweighted_basic_graph')
             for name in results if name != 'base_unweighted__no_mechanisms']
    pairs += [(mode + '__no_mechanisms', mode + '__mechanisms', 'paired_mechanism_effect') for mode in WEIGHT_MODES]
    for before, after, comparison in pairs:
        if set(lookup[before]) != set(lookup[after]):
            raise ValueError('Unpaired heldout cases between ablations')
        for split in ['dev', 'test']:
            for field in metric_fields:
                deltas = []
                for cid, a in lookup[before].items():
                    b = lookup[after][cid]
                    if a['split'] != split or a[field] is None or b[field] is None:
                        continue
                    deltas.append({'case_id': cid, 'group_id': a['group_id'], 'delta': float(b[field]) - float(a[field])})
                stat = paired_bootstrap(deltas, ['delta'], seed=42, n_boot=2000)
                value = stat.get('metrics', {}).get('delta', {'mean': None, 'ci95': [None, None]})
                comparisons.append({'before': before, 'after': after, 'comparison': comparison,
                    'split': split, 'metric': field, 'paired_numbered_cases': len(deltas),
                    'paired_case_groups': stat['n_case_groups'], 'mean_paired_difference': value['mean'],
                    'bootstrap_ci95': value['ci95'], 'bootstrap_samples': 2000,
                    'resampling_unit': 'unique normalized body; all paired configurations retained',
                    'multiple_comparison_adjustment': 'none; exploratory fixed-protocol comparisons'})
    return comparisons


def structure_summary(graph, nodes, edges):
    assignments, community = community_analysis(graph, seed=42)
    for entry in community['within_between'].values():
        entry.pop('weights', None)
    weights = [d['weight'] for _, _, d in graph.edges(data=True)]
    topology = sorted([u, v] for u, v in graph.edges())
    weighted = sorted([u, v, d['weight']] for u, v, d in graph.edges(data=True))
    return {'nodes': len(nodes), 'typed_analysis_edges': len(edges), 'directed_pairs': graph.number_of_edges(),
        'positive_weight_directed_pairs': sum(w > 0 for w in weights),
        'weak_components': nx.number_weakly_connected_components(graph),
        'strong_components': nx.number_strongly_connected_components(graph),
        'mean_association_weight': float(np.mean(weights)) if weights else None,
        'association_weight_quantiles': [float(v) for v in np.quantile(weights, [0, .25, .5, .75, 1])] if weights else [],
        'automatic_communities': community, 'community_assignments': assignments,
        'topology_sha256': hashlib.sha256(json.dumps(topology).encode()).hexdigest(),
        'weighted_projection_sha256': hashlib.sha256(json.dumps(weighted).encode()).hexdigest(),
        'physical_proxy_values': sorted({e['physical'] for e in edges}),
        'physical_gold_present': False}


def export_short_candidate_chains(nodes, graph, weighted_edges, train_cases, ner_by_case, output):
    """All qualified three-node chains; support must intersect in an actual train case.

    This bounded Table1-style audit is distinct from exhaustive all-hop mining.
    Frequencies describe candidate-support case groups, not causal occurrence.
    """
    cases = {c['case_id']: c for c in train_cases}
    grouped_edges = defaultdict(list)
    for e in weighted_edges:
        grouped_edges[e['source'], e['target']].append(e)
    chains = []
    for u, v in sorted(graph.edges()):
        if nodes[u]['entity_type'] not in ROOT_TYPES:
            continue
        for w in sorted(graph.successors(v)):
            ids = [u, v, w]
            if not path_qualified(ids, nodes, graph):
                continue
            common = sorted(set(graph[u][v]['case_support']) & set(graph[v][w]['case_support']))
            if not common:
                continue
            citations = []
            for cid in common:
                text = cases[cid]['text']
                mentions = ner_by_case[cid]['entities']
                node_evidence = []
                for nid in ids:
                    node = nodes[nid]
                    if node.get('layer') == 'mechanism':
                        quotes = [{'span': [m.start(), m.end()], 'quote': m.group(), 'dictionary_term': term}
                                  for term in node.get('dictionary_terms', [])
                                  for m in re.finditer(re.escape(term), text)]
                        kind = 'dictionary mechanism abstraction; not observed causal mechanism gold'
                    else:
                        quotes = [{'span': [e['start'], e['end']], 'quote': text[e['start']:e['end']],
                                   'NER_occurrence_id': e['occurrence_id']}
                                  for e in mentions if e['type'] == node['entity_type'] and e['text'] == node['name']]
                        kind = 'actual frozen NER mention'
                    node_evidence.append({'node_id': nid, 'evidence_kind': kind, 'quotes': quotes})
                edge_evidence = [{'source': a, 'target': b,
                    'train_supporting_analysis_edge_ids': [e['id'] for e in grouped_edges[a, b] if cid in e['case_support']],
                    'basis_candidate_edge_ids': sorted({eid for e in grouped_edges[a, b] if cid in e['case_support'] for eid in e['basis_base_edge_ids']})}
                    for a, b in zip(ids, ids[1:])]
                citations.append({'case_id': cid, 'full_case_quote': text, 'node_evidence': node_evidence,
                                  'edge_evidence': edge_evidence, 'causal_gold': False})
            chains.append({'node_ids': ids, 'node_names': [nodes[n]['name'] for n in ids],
                'n_train_numbered_case_supports': len(common),
                'n_train_unique_body_supports': len({normalized_body(cases[cid]['text']) for cid in common}),
                'mean_association_weight': float(np.mean([graph[u][v]['weight'], graph[v][w]['weight']])),
                'product_association_score': graph[u][v]['weight'] * graph[v][w]['weight'],
                'supporting_train_case_ids': common, 'citations': citations,
                'candidate_chain_not_causal_gold': True, 'hop_policy': 'exactly two directed hops / three nodes'})
    chains.sort(key=lambda c: (-c['n_train_unique_body_supports'], -c['mean_association_weight'], c['node_ids']))
    write_rows(output / 'three_node_candidate_chains_with_case_quotes.jsonl', chains)
    csv_write(output / 'table1_style_top10_candidates.csv', ({k: row[k] for k in ['node_ids', 'node_names',
        'n_train_numbered_case_supports', 'n_train_unique_body_supports', 'mean_association_weight',
        'product_association_score', 'supporting_train_case_ids', 'candidate_chain_not_causal_gold', 'hop_policy']}
        for row in chains[:10]))
    return {'n_three_node_candidate_chains': len(chains), 'exhaustive_all_hop_path_count': False,
            'candidate_case_support_not_observed_causal_frequency': True,
            'all_chain_citations_use_train_cases': True}


def run(args):
    started = time.perf_counter()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be new/empty; do not overwrite an executed result')
    output.mkdir(parents=True, exist_ok=True)
    inputs = {name: Path(getattr(args, name)).resolve() for name in ['cases', 'entities', 'triples', 'candidate_graph']}
    if getattr(args, 'forced_splits', None):
        inputs['forced_splits'] = args.forced_splits.resolve()
    before = {name: {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)} for name, path in inputs.items()}
    code = {str(Path(p).resolve()): sha(p) for p in SOURCE_MODULES}
    dump(output / 'run_config.json', {'argv': sys.argv, 'cwd': str(Path.cwd()), 'python': sys.executable,
        'python_version': platform.python_version(), 'numpy_version': np.__version__, 'gensim_version': gensim.__version__,
        'networkx_version': nx.__version__, 'device': 'CPU', 'inputs': before, 'code_sha256': code,
        'settings': {'seed': args.seed, 'grouping': 'strip case enumeration, NFKC, delete all whitespace, SHA256',
            'unconstrained_train_dev_test_group_fractions': [.70, .15, .15], 'fraction_rounding': 'floor train/dev; test remainder',
            'fixed_domain_partition_constraints': str(inputs.get('forced_splits')) if 'forced_splits' in inputs else None,
            'topic_k': args.topics, 'lda_passes': args.passes, 'lda_iterations': args.iterations,
            'min_df': args.min_df, 'timing_repeats': args.timing_repeats,
            'fixed_ablations': list(WEIGHT_MODES), 'mechanism_levels': [False, True],
            'configuration_selection_on_dev_or_test': False},
        'supervised_source_independence_proven': False, 'human_gold_provided': False,
        'relation_candidates_model_existence': False, 'no_relation_annotation_available': False})
    cases, entities, triples = (read_rows(inputs[name]) for name in ['cases', 'entities', 'triples'])
    candidate_graph = json.loads(inputs['candidate_graph'].read_text())
    validation = validate_inputs(cases, entities, triples, candidate_graph)
    forced = None
    if 'forced_splits' in inputs:
        manifest = json.loads(inputs['forced_splits'].read_text())
        forced = {split: manifest['domain_' + split] for split in ['train', 'dev', 'test']}
    groups, by_case = grouped_split(cases, args.seed, forced)
    write_rows(output / 'case_group_split.jsonl', groups)
    split_ids = {split: {cid for cid, group in by_case.items() if group['split'] == split} for split in ['train', 'dev', 'test']}
    split_summary = {split: {'numbered_cases': len(ids), 'unique_groups': sum(g['split'] == split for g in groups),
        'actual_unique_group_fraction': sum(g['split'] == split for g in groups) / len(groups),
        'domain_constrained_unique_groups': sum(g['split'] == split and g['assignment_source'] == 'fixed_domain_adaptation_partition_constraint' for g in groups)}
        for split, ids in split_ids.items()}
    dump(output / 'split_summary.json', {'splits': split_summary, 'group_overlap': 0,
        'normalized_duplicate_grouping': True, 'NER_RE_supervised_sources_disjoint_from_graph_test': 'unproven',
        'graph_and_LDA_fit_groups_disjoint_from_dev_test': True,
        'domain_constraints_are_posthoc_exploratory_context_not_independent_industrial_gold': bool(forced)})
    print('Grouped case split: ' + json.dumps(split_summary), flush=True)
    ner_by_case = {r['case_id']: r for r in entities}
    train_cases = sorted((c for c in cases if c['case_id'] in split_ids['train']), key=lambda c: c['case_id'])
    train_entities = [ner_by_case[c['case_id']] for c in train_cases]
    heldout_cases = sorted((c for c in cases if c['case_id'] not in split_ids['train']), key=lambda c: c['case_id'])
    write_rows(output / 'train_cases.jsonl', train_cases)
    write_rows(output / 'heldout_cases.jsonl', heldout_cases)
    train_graph = aggregate_train_graph(triples, split_ids['train'])
    dump(output / 'train_candidate_graph.json', train_graph)
    train_triples = [r for r in triples if r['case_id'] in split_ids['train']]
    write_rows(output / 'train_candidate_triples.jsonl', train_triples)
    settings = Settings(seed=args.seed, fixed_paper_k=args.topics, passes=args.passes,
                        iterations=args.iterations, min_df=args.min_df)
    tv, sv, group_topics, topics = fit_train_topics(train_cases, train_entities, by_case, output / 'train_topics', settings)
    base_nodes, basic_edges, enhanced_nodes, analysis_edges, bridges, augmented = build_layers(
        train_graph, train_cases, tv, sv, 'model_occurrences')
    grouped_features(enhanced_nodes, augmented, by_case, group_topics, sv)
    assert_train_support(enhanced_nodes, augmented, split_ids['train'], by_case)
    write_rows(output / 'heldout_query_and_silver_reference.jsonl', ({'case_id': c['case_id'],
        'group_id': by_case[c['case_id']]['group_id'], 'split': by_case[c['case_id']]['split'],
        **query_prefix(c), 'silver_reference': silver_reference(c, ner_by_case[c['case_id']])} for c in heldout_cases))
    results, structural, retrieval, chain_summaries = {}, {}, {}, {}
    for mode in WEIGHT_MODES:
        for mechanisms in [False, True]:
            name = mode + ('__mechanisms' if mechanisms else '__no_mechanisms')
            folder = output / name
            folder.mkdir()
            nodes = enhanced_nodes if mechanisms else base_nodes
            edges = augmented if mechanisms else analysis_edges
            graph, weighted_edges = make_variant(nodes, edges, mode)
            assert_train_support(nodes, weighted_edges, split_ids['train'], by_case)
            dump(folder / 'graph.json', {'nodes': [nodes[n] for n in sorted(nodes)],
                'analysis_edges': weighted_edges, 'support_mode': 'train_group_model_occurrences',
                'split': 'train', 'candidate_causal_gold': False, 'weight_mode': mode,
                'mechanisms_enabled': mechanisms, 'heldout_support_added': False})
            structural[name] = structure_summary(graph, nodes, weighted_edges)
            dump(folder / 'structure_summary.json', structural[name])
            print(f'{name}: {len(nodes)} nodes, {graph.number_of_edges()} analysis pairs; evaluate heldout', flush=True)
            results[name], retrieval[name] = evaluate_variant(name, nodes, graph, heldout_cases, by_case, ner_by_case,
                                                              folder, args.timing_repeats)
            if mode == 'full_fusion':
                chain_summaries[name] = export_short_candidate_chains(nodes, graph, weighted_edges, train_cases, ner_by_case, folder)
    comparisons = compare_variants(results)
    csv_write(output / 'paired_bootstrap_differences.csv', comparisons)
    dump(output / 'paired_bootstrap_differences.json', comparisons)
    compact = []
    for name, splits in retrieval.items():
        for split, summary in splits.items():
            row = {'variant': name, 'split': split, 'numbered_cases': summary['numbered_case_denominator'],
                   'unique_groups': summary['n_case_groups'], 'silver_groups': summary['silver_reference_subset']['n_case_groups']}
            row.update({key: value['mean'] for key, value in summary['metrics'].items()})
            row.update({key: value['mean'] for key, value in summary['silver_reference_subset']['metrics'].items()})
            compact.append(row)
    csv_write(output / 'retrieval_comparison.csv', compact)
    csv_write(output / 'structure_comparison.csv', ({'variant': name, **{key: row[key] for key in ['nodes',
        'typed_analysis_edges', 'directed_pairs', 'positive_weight_directed_pairs', 'weak_components',
        'strong_components', 'mean_association_weight', 'physical_proxy_values', 'topology_sha256',
        'weighted_projection_sha256']}, 'n_automatic_communities': row['automatic_communities']['n_auto_communities'],
        'modularity': row['automatic_communities']['modularity']} for name, row in structural.items()))
    equal_proxy = {level: structural['base_unweighted__' + level]['weighted_projection_sha256'] ==
                         structural['physical_proxy_only__' + level]['weighted_projection_sha256']
                   for level in ['no_mechanisms', 'mechanisms']}
    if any(sha(path) != before[name]['sha256'] for name, path in inputs.items()):
        raise ValueError('Frozen input changed during experiment')
    summary = {'status': 'completed_grouped_heldout_silver_retrieval_ablation', 'split_summary': split_summary,
        'input_validation': validation, 'all_train_support_checks_passed': True,
        'LDA_and_NER_lexicon_fit_train_only': True, 'n_executed_graph_variants': len(results),
        'n_heldout_query_executions': sum(len(r) * args.timing_repeats for r in results.values()),
        'train_topics': topics, 'structure': structural, 'retrieval': retrieval,
        'candidate_chain_exports': chain_summaries, 'base_equals_physical_proxy_weighted_graph': equal_proxy,
        'elapsed_seconds': time.perf_counter() - started, 'input_files_unchanged': True,
        'paper_numeric_reproduction': False, 'root_cause_accuracy': None, 'path_completeness_rate': None,
        'SIS': None, 'engineer_localization_time': None, 'temporal_speed': None,
        'limitations': ['Frozen NER/RE training-source independence from case test groups has not been proved.',
            'Graph split isolation does not make same-model text-derived references human gold.',
            'Threshold 0 keeps existing argmax classes; relation existence and causal correctness are unvalidated.',
            'Diagnosis-window references are NER/text silver labels with exact source quotes, not confirmed root causes.',
            'Physical feature is binary candidate/input-relation support; its constant value makes this ablation degenerate.',
            'Weight-only ablations preserve adjacency; changes in ranking or communities do not prove accuracy/completeness.',
            'Query boundary rules abstain when no marker is recognized; boundaries have not been human reviewed.',
            'Maximum-product explanations are association rankings and not exhaustive all-hop paths.',
            'Machine timing is local CPU retrieval time, not engineer root-cause localization time.',
            'No expert SIS, independent physical labels, reviewed no_relation or dated fault events were fabricated.']}
    dump(output / 'analysis_summary.json', summary)
    dump(output / 'result_checks.json', {'status': 'passed', **validation,
        'unique_body_group_split_disjoint': True, 'train_graph_has_no_heldout_case_support': True,
        'topic_fit_and_lexicon_train_only': True, 'all_10_graph_configurations_executed': len(results) == 10,
        'paired_query_case_sets_identical': True, 'input_SHA_unchanged': True,
        'physical_proxy_degeneracy_recorded': equal_proxy, 'no_causal_accuracy_or_SIS_invented': True})
    print('Finished actual grouped graph ablations: ' + str(output / 'analysis_summary.json'), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['cases', 'entities', 'triples', 'candidate-graph', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--topics', type=int, default=7)
    p.add_argument('--passes', type=int, default=20)
    p.add_argument('--iterations', type=int, default=200)
    p.add_argument('--min-df', type=int, default=2)
    p.add_argument('--timing-repeats', type=int, default=3)
    p.add_argument('--forced-splits', type=Path, help='Fixed domain_train/domain_dev/domain_test case IDs; entire normalized groups obey constraints')
    args = p.parse_args()
    if min(args.topics, args.passes, args.iterations, args.min_df, args.timing_repeats) <= 0:
        p.error('Model and timing integer settings must be positive')
    run(args)


if __name__ == '__main__':
    main()
