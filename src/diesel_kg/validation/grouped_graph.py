#!/usr/bin/env python3
"""Independently verify saved split, feature, quote and metric boundaries."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np
from gensim.corpora import Dictionary

from diesel_kg.graph.ablation import normalized_body, sha as file_sha, read_rows as file_rows, dump
from diesel_kg.paths import resolve_recorded_path, audit_recorded_sources


def sha(path):
    return file_sha(resolve_recorded_path(path))


def read_rows(path):
    return file_rows(resolve_recorded_path(path))


def run(root):
    started = time.perf_counter()
    root = Path(root).resolve()
    config = json.loads((root / 'run_config.json').read_text())
    summary = json.loads((root / 'analysis_summary.json').read_text())
    checks = {}

    def require(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise AssertionError(name)

    require('main_actual_execution_completed', summary['status'] == 'completed_grouped_heldout_silver_retrieval_ablation')
    require('frozen_inputs_unchanged', all(sha(v['path']) == v['sha256'] for v in config['inputs'].values()))
    source_audit = audit_recorded_sources(config['code_sha256'])
    if source_audit['all_source_bytes_verified']:
        require('executed_source_code_unchanged', True)
    else:
        require('removed_historical_source_fingerprints_disclosed', bool(source_audit['sources']))
    cases = {c['case_id']: c for c in read_rows(config['inputs']['cases']['path'])}
    ner = {r['case_id']: r for r in read_rows(config['inputs']['entities']['path'])}
    groups = read_rows(root / 'case_group_split.jsonl')
    by_case = {cid: g for g in groups for cid in g['case_ids']}
    require('complete_unique_group_case_assignment', len(by_case) == len(cases) and set(by_case) == set(cases))
    require('normalization_recovered_for_every_case', all(by_case[cid]['group_id'] == 'body_' +
        hashlib.sha256(normalized_body(c['text']).encode()).hexdigest() for cid, c in cases.items()))
    train_ids = {cid for cid, g in by_case.items() if g['split'] == 'train'}
    heldout_ids = set(cases) - train_ids
    if 'forced_splits' in config['inputs']:
        domain = json.loads(resolve_recorded_path(config['inputs']['forced_splits']['path']).read_text())
        require('all_fixed_domain_partitions_respected', all(by_case[cid]['split'] == split
            for split in ['train', 'dev', 'test'] for cid in domain['domain_' + split]))
    train_file = read_rows(root / 'train_cases.jsonl')
    require('training_case_file_equals_split', {c['case_id'] for c in train_file} == train_ids)
    graph = json.loads((root / 'train_candidate_graph.json').read_text())
    require('candidate_graph_train_support_only', all(set(row['case_ids']) <= train_ids for row in graph['nodes'] + graph['edges']))
    triples = read_rows(root / 'train_candidate_triples.jsonl')
    require('all_train_triples_use_train_cases', all(r['case_id'] in train_ids for r in triples))
    source_triples = read_rows(config['inputs']['triples']['path'])
    require('train_triples_exact_filter_of_frozen_source', triples == [r for r in source_triples if r['case_id'] in train_ids])
    topics = json.loads((root / 'train_topics/analysis_summary.json').read_text())
    require('topic_training_case_ids_exactly_train', set(topics['training_case_ids']) == train_ids)
    token_rows = read_rows(root / 'train_topics/train_unique_group_tokens.jsonl')
    require('topic_fit_uses_one_train_representative_per_group', len(token_rows) == len({by_case[cid]['group_id'] for cid in train_ids})
        and all(r['case_id'] in train_ids and r['case_id'] == by_case[r['case_id']]['representative_case_id'] for r in token_rows))
    reconstructed = Dictionary([r['tokens_before_min_df'] for r in token_rows])
    reconstructed.filter_extremes(no_below=config['settings']['min_df'], no_above=1., keep_n=None)
    reconstructed.compactify()
    saved_dictionary = Dictionary.load(str(root / 'train_topics/dictionary.gensim'))
    require('topic_dictionary_recovered_from_training_tokens', reconstructed.token2id == saved_dictionary.token2id)
    raw_train_lexicon = {e['text'].strip() for cid in train_ids for e in ner[cid]['entities']}
    with (root / 'train_topics/professional_lexicon.csv').open(encoding='utf-8-sig') as f:
        lexicon = list(csv.DictReader(f))
    require('all_learned_lexicon_terms_exist_in_train_NER', all('provided_NER_entity' not in json.loads(r['sources'])
        or r['term'] in raw_train_lexicon for r in lexicon))
    with (root / 'train_topics/train_document_topic.csv').open(encoding='utf-8-sig') as f:
        topic_rows = list(csv.DictReader(f))
    group_topics = {r['group_id']: np.array([float(r[f'T{i+1:02d}']) for i in range(config['settings']['topic_k'])]) for r in topic_rows}
    require('train_topic_probability_vectors_valid', all(np.isfinite(v).all() and (v >= 0).all() and math.isclose(float(v.sum()), 1, abs_tol=1e-6) for v in group_topics.values()))
    require('ten_actual_variant_artifacts_present', len(summary['retrieval']) == 10 and len(summary['structure']) == 10)
    metric_fields = ['query_has_graph_symptom', 'query_has_candidate_root', 'n_root_candidates', 'machine_query_ms']
    silver_fields = ['silver_root_recall_at_1', 'silver_root_recall_at_3', 'silver_root_recall_at_5',
        'silver_root_hit_at_1', 'silver_root_hit_at_3', 'silver_root_hit_at_5', 'silver_root_mrr', 'silver_candidate_recall',
        'silver_targets_in_train_graph_fraction']
    variant_queries = {}
    feature_ok, query_ok, silver_ok, means_ok = True, True, True, True
    for name in summary['retrieval']:
        saved = json.loads((root / name / 'graph.json').read_text())
        nodes = {n['id']: n for n in saved['nodes']}
        for n in nodes.values():
            feature_ok &= set(n['case_support']) <= train_ids
            supports = sorted({by_case[cid]['group_id'] for cid in n['case_support']})
            feature_ok &= supports == n['training_group_support']
            expected = np.mean([group_topics[g] for g in supports], axis=0) if supports else np.zeros(config['settings']['topic_k'])
            feature_ok &= bool(np.allclose(expected, n['topic_distribution'], atol=1e-12))
        for e in saved['analysis_edges']:
            feature_ok &= set(e['case_support']) <= train_ids
            u, v = nodes[e['source']], nodes[e['target']]
            a, b = np.array(u['topic_distribution']), np.array(v['topic_distribution'])
            den = np.linalg.norm(a) * np.linalg.norm(b)
            cosine = float(a @ b / den) if den else 0.
            su, sv = set(u['training_group_support']), set(v['training_group_support'])
            ochiai = len(su & sv) / math.sqrt(len(su) * len(sv)) if su and sv else 0.
            feature_ok &= math.isclose(cosine, e['semantic'], abs_tol=1e-12) and math.isclose(ochiai, e['cooccurrence'], abs_tol=1e-12)
        rows = read_rows(root / name / 'heldout_queries.jsonl')
        variant_queries[name] = {r['case_id']: r for r in rows}
        query_ok &= set(variant_queries[name]) == heldout_ids
        for r in rows:
            text = cases[r['case_id']]['text']
            query_ok &= r['query_text'] == text[:r['boundary_span'][0]] if r['boundary_span'] else r['query_text'] == ''
            query_ok &= all(nid in nodes for nid in r['ranked_candidate_ids'])
            for match in r['query_symptoms']:
                query_ok &= all(r['query_text'][a:b] == quote for (a, b), quote in zip(match['query_spans'], match['quotes']))
            for reference in r['silver_reference']['references']:
                a, b = reference['span']
                silver_ok &= text[a:b] == reference['quote'] and not reference['is_human_gold'] and not reference['is_verified_causal_root']
                silver_ok &= any(e['occurrence_id'] == reference['NER_occurrence_id'] and e['text'] == reference['quote']
                                  and e['start'] == a and e['end'] == b for e in ner[r['case_id']]['entities'])
            query_ok &= r['causal_accuracy'] is None and not r['timing_is_engineer_localization_time']
        for split in ['dev', 'test']:
            for field in metric_fields + silver_fields:
                applicable = [r for r in rows if r['split'] == split and r[field] is not None]
                grouped = defaultdict(list)
                for r in applicable:
                    grouped[r['group_id']].append(float(r[field]))
                mean = float(np.mean([np.mean(v) for v in grouped.values()])) if grouped else None
                saved_metrics = summary['retrieval'][name][split]
                saved_metrics = saved_metrics['silver_reference_subset']['metrics'] if field in silver_fields else saved_metrics['metrics']
                actual = saved_metrics.get(field, {}).get('mean')
                means_ok &= mean is None and actual is None or mean is not None and actual is not None and math.isclose(mean, actual, abs_tol=1e-12)
    require('every_node_and_edge_feature_recomputed_from_train_groups', feature_ok)
    require('all_heldout_query_cases_and_source_quotes_checked', query_ok)
    require('every_silver_reference_has_real_NER_span_and_false_gold_flags', silver_ok)
    require('all_saved_group_mean_metrics_independently_recomputed', means_ok)
    comparisons = json.loads((root / 'paired_bootstrap_differences.json').read_text())
    differences_ok = True
    for row in comparisons:
        paired = defaultdict(list)
        for cid, a in variant_queries[row['before']].items():
            b = variant_queries[row['after']][cid]
            f = row['metric']
            if a['split'] == row['split'] and a[f] is not None and b[f] is not None:
                paired[a['group_id']].append(float(b[f]) - float(a[f]))
        mean = float(np.mean([np.mean(v) for v in paired.values()])) if paired else None
        differences_ok &= row['paired_case_groups'] == len(paired)
        differences_ok &= mean is None and row['mean_paired_difference'] is None or mean is not None and row['mean_paired_difference'] is not None and math.isclose(mean, row['mean_paired_difference'], abs_tol=1e-12)
    require('all_pairwise_difference_means_and_group_counts_recomputed', differences_ok)
    require('missing_external_metrics_remain_null', all(summary[k] is None for k in ['root_cause_accuracy',
        'path_completeness_rate', 'SIS', 'engineer_localization_time', 'temporal_speed']))
    model_bindings = {}
    ner_summary_path = resolve_recorded_path(config['inputs']['entities']['path']).with_suffix('.summary.json')
    adapted_ner = False
    if ner_summary_path.exists():
        producer = json.loads(ner_summary_path.read_text())
        checkpoint = resolve_recorded_path(producer['checkpoint_path'])
        adapted_ner = bool(producer.get('selected_head_verified_equal'))
        require('frozen_NER_producer_output_SHA_matches', producer['output_sha256'] == config['inputs']['entities']['sha256'])
        require('frozen_NER_checkpoint_SHA_rechecked', sha(checkpoint) == producer['checkpoint_sha256'])
        model_bindings['NER'] = {'producer_summary_path': str(ner_summary_path), 'producer_summary_sha256': sha(ner_summary_path),
            'checkpoint_path': str(checkpoint), 'checkpoint_sha256': producer['checkpoint_sha256'], 'variant': producer['variant']}
    re_sources = [ner_summary_path.parent / 're_locator_fixed/run_summary.json',
                  resolve_recorded_path(config['inputs']['triples']['path']).parent.parent / 're_occurrences/run_summary.json']
    re_summary_path = next((p for p in re_sources if p.exists()), re_sources[0])
    if re_summary_path.exists():
        producer = json.loads(re_summary_path.read_text())
        checkpoints = [(path, digest) for path, digest in producer['input_sha256'].items() if Path(path).suffix == '.pt']
        require('one_frozen_RE_checkpoint', len(checkpoints) == 1)
        require('frozen_RE_checkpoint_SHA_rechecked', all(sha(path) == digest for path, digest in checkpoints))
        model_bindings['RE'] = {'producer_summary_path': str(re_summary_path), 'producer_summary_sha256': sha(re_summary_path),
            'checkpoint_path': str(resolve_recorded_path(checkpoints[0][0])), 'checkpoint_sha256': checkpoints[0][1]}
    dump(root / 'fixed_model_producer_bindings.json', {'models': model_bindings,
        'supervised_training_source_independence_proven': False, 'checkpoint_and_prediction_provenance_checked': True,
        'domain_adaptation_models_used_in_this_graph_run': adapted_ner})
    result = {'status': 'passed', 'checks': checks, 'n_passed': sum(checks.values()),
        'argv': sys.argv, 'python': sys.executable, 'script_sha256': sha(__file__),
        'analysis_summary_sha256': sha(root / 'analysis_summary.json'),
        'recorded_source_audit': source_audit,
        'elapsed_seconds': time.perf_counter() - started,
        'verification_is_causal_or_industrial_validation': False}
    dump(root / 'independent_result_checks.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph-ablation', type=Path, required=True)
    run(parser.parse_args().graph_ablation)
