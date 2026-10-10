"""Collect measured computations from one completed experiment, with provenance."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def collect(root: Path):
    root = Path(root)
    sources = {}
    rows = []

    def read(name):
        path = root / name
        content = path.read_bytes()
        sources[name] = hashlib.sha256(content).hexdigest()
        return json.loads(content)

    def add(experiment, metric, value, scope, source, field):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Invalid measured value: {source}: {field}')
        rows.append({'experiment': experiment, 'metric': metric, 'value': value,
                     'scope': scope, 'source': source, 'field': field})

    name = 'ner_domain/adaptation/summary.json'
    ner = read(name)
    if ner['status'] != 'completed':
        raise ValueError('NER training record is incomplete')
    for model in ('baseline', 'selected'):
        add('NER: same fixed domain-test records', model + ' strict span+type F1',
            ner['domain_test_' + model]['f1'], 'exploratory internal resplit; original labels',
            name, 'domain_test_' + model + '.f1')
    name = 'ner_domain/adaptation/corpus_entities.summary.json'
    corpus = read(name)
    if corpus['status'] != 'completed':
        raise ValueError('NER corpus inference is incomplete')
    add('NER: full corpus inference', 'mentions', corpus['counts']['mentions'],
        'all occurrence offsets preserved', name, 'counts.mentions')

    name = 're_clean/grouped_training/summary.json'
    re = read(name)
    if re['status'] != 'completed':
        raise ValueError('RE training record is incomplete')
    for field in ('accuracy', 'macro_f1_all_nine_classes'):
        add('RE: template-heldout examples', field, re['test']['group_heldout'][field],
            '9 positive relation classes; inferred-template holdout; no relation-existence gold',
            name, 'test.group_heldout.' + field)
    add('RE: previously evaluated additional benchmark', 'accuracy',
        re['test']['original225_previously_evaluated']['accuracy'],
        'previously evaluated original 225 examples', name,
        'test.original225_previously_evaluated.accuracy')
    name = 'adapted_model_graph/re_occurrences/run_summary.json'
    relations = read(name)
    if not relations['status'].startswith('completed'):
        raise ValueError('Relation corpus inference is incomplete')
    add('Selected models: full corpus', 'ordered occurrence pairs',
        relations['counts']['ordered_pairs'], 'uncapped actual inference', name, 'counts.ordered_pairs')

    name = 'adapted_model_graph/relation_threshold/threshold_comparison.csv'
    path = root / name
    sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding='utf-8-sig', newline='') as stream:
        thresholds = list(csv.DictReader(stream))
    for threshold in (0., .8):
        matches = [row for row in thresholds if float(row['threshold']) == threshold]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one threshold {threshold} result')
        add(f'Selected models: threshold {threshold:g}', 'retained candidate observations',
            int(matches[0]['n_retained_occurrence_pairs']),
            'diagnostic threshold sensitivity; no causal gold', name,
            f'threshold={threshold:g}:n_retained_occurrence_pairs')

    for model, directory in [('Reference models', 'graph_ablation'),
                             ('Selected models', 'adapted_model_graph/graph_ablation')]:
        name = directory + '/analysis_summary.json'
        report = read(name)
        if not report['status'].startswith('completed'):
            raise ValueError(f'Graph experiment is incomplete: {directory}')
        for variant in ('base_unweighted__no_mechanisms', 'full_fusion__mechanisms'):
            metrics = report['retrieval'][variant]['test']['silver_reference_subset']['metrics']
            for metric in ('silver_root_recall_at_3', 'silver_root_mrr'):
                add(model + ': ' + variant, metric, metrics[metric]['mean'],
                    'branch-specific model-derived silver references; not comparable between model branches',
                    name, f'retrieval.{variant}.test.silver_reference_subset.metrics.{metric}.mean')

    name = 'cross_model_silver_evaluation/summary.json'
    comparison = read(name)
    if comparison['status'] != 'completed':
        raise ValueError('Fixed-reference comparison is incomplete')
    for variant in ('base_unweighted__no_mechanisms', 'full_fusion__mechanisms'):
        for model in ('baseline', 'new'):
            for metric in ('recall_at_3', 'mrr'):
                add('Fixed-reference comparison: ' + variant, model + ' ' + metric,
                    comparison['comparisons'][variant][model][metric]['mean'],
                    'paired models on the same frozen silver references; no human causal gold',
                    name, f'comparisons.{variant}.{model}.{metric}.mean')
    return rows, sources


def write_summary(root: Path, output: Path, replace: bool = False):
    output = Path(output)
    destinations = [output / 'measured_metrics.csv', output / 'metrics_provenance.json']
    if not replace and any(path.exists() for path in destinations):
        raise ValueError('Measured summary already exists; choose a new output directory')
    rows, sources = collect(root)
    output.mkdir(parents=True, exist_ok=True)
    columns = ['experiment', 'metric', 'value', 'scope']
    with destinations[0].open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    metadata = {'status': 'completed', 'data_status': 'MEASURED_COMPUTATION',
                'experiment': str(Path(root).resolve()), 'row_count': len(rows),
                'source_sha256': sources,
                'row_sources': [{'row': i + 2, 'source': row['source'], 'field': row['field']}
                                for i, row in enumerate(rows)],
                'summary_sha256': hashlib.sha256(destinations[0].read_bytes()).hexdigest(),
                'human_causal_or_industrial_validation': False}
    destinations[1].write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--replace', action='store_true', help='Refresh derived summary files from unchanged source results')
    args = parser.parse_args()
    result = write_summary(args.results_dir, args.output_dir or args.results_dir, args.replace)
    print(json.dumps({'status': result['status'], 'rows': result['row_count']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
