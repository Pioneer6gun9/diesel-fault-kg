"""Audit supplied data, preserve provenance and remove duplicate case bodies.

This module uses the standard library and never alters source data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def jsonl_write(path, rows):
    with Path(path).open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def read_cases(path):
    cases, chunks, first = [], [], None
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    for n, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        if line.startswith('案例') and chunks:
            cases.append({'text': ''.join(chunks), 'first_line': first, 'last_line': n - 1})
            chunks, first = [], None
        if first is None:
            first = n
        chunks.append(line)
    if chunks:
        cases.append({'text': ''.join(chunks), 'first_line': first, 'last_line': len(lines)})
    for n, row in enumerate(cases, 1):
        row['case_id'] = f'case_{n:05d}'
        row['body'] = re.sub(r'^案例\s*\d+\s*[：:]?\s*', '', row['text'])
    return cases


def read_bio(path):
    rows, chars, labels = [], [], []
    for line in Path(path).read_text(encoding='utf-8').splitlines() + ['']:
        if not line.strip():
            if chars:
                rows.append({'text': ''.join(chars), 'labels': labels})
                chars, labels = [], []
        else:
            char, label = line.split('\t', 1)
            chars.append(char); labels.append(label)
    return rows


def audit(kg_build, output):
    output.mkdir(parents=True, exist_ok=True)
    corpus = read_cases(kg_build / '案例语料.txt')
    first_by_body, unique, mapping = {}, [], []
    for row in corpus:
        body = row['body']
        if body not in first_by_body:
            first_by_body[body] = row['case_id']
            unique.append({'case_id': row['case_id'], 'text': body,
                           'source_first_line': row['first_line'], 'source_last_line': row['last_line']})
        mapping.append({'case_id': row['case_id'], 'canonical_case_id': first_by_body[body],
                        'is_duplicate': row['case_id'] != first_by_body[body],
                        'body_sha256': hashlib.sha256(body.encode()).hexdigest()})
    jsonl_write(output / 'cases_all.jsonl', [{'case_id': x['case_id'], 'text': x['body']} for x in corpus])
    jsonl_write(output / 'cases_unique.jsonl', unique)
    with (output / 'case_duplicate_mapping.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(mapping[0])); writer.writeheader(); writer.writerows(mapping)
    json_write(output / 'case_duplicate_mapping.json', mapping)
    bio = {split: read_bio(kg_build / f'NER/ner_data_all_generate/{split}.txt') for split in ['train', 'dev', 'test']}
    ner_texts = {split: {x['text'] for x in data} for split, data in bio.items()}
    rel = {split: json.loads((kg_build / f'RE/re_data_generate/{split}.json').read_text()) for split in ['train', 'dev', 'test']}
    signature = lambda x: tuple(x[k] for k in ['text', 'subject', 'object', 'relation'])
    rel_sigs = {split: {signature(x) for x in data} for split, data in rel.items()}
    rel_texts = {split: {x['text'] for x in data} for split, data in rel.items()}
    splits = [('train', 'dev'), ('train', 'test'), ('dev', 'test')]
    counts = Counter(x['body'] for x in corpus)
    summary = {
        'corpus': {'provided_numbered_cases': len(corpus), 'unique_exact_case_bodies': len(unique),
                   'duplicate_rows_removed': len(corpus) - len(unique),
                   'number_of_duplicate_groups': sum(n > 1 for n in counts.values()),
                   'max_group_multiplicity': max(counts.values()),
                   'deduplication': 'Remove leading case number; exact remaining Unicode body equality; retain first source ID.'},
        'ner': {'splits': {s: {'sentences': len(d), 'unique_texts': len(ner_texts[s]),
                               'entity_type_counts': dict(Counter(label[2:] for row in d for label in row['labels'] if label.startswith('B-')))} for s, d in bio.items()},
                'exact_text_overlap': {f'{a}_{b}': len(ner_texts[a] & ner_texts[b]) for a, b in splits}},
        're': {'splits': {s: {'records': len(d), 'unique_labelled_records': len(rel_sigs[s]),
                              'unique_texts': len(rel_texts[s]), 'relations': dict(Counter(x['relation'] for x in d))} for s, d in rel.items()},
               'exact_labelled_record_overlap': {f'{a}_{b}': len(rel_sigs[a] & rel_sigs[b]) for a, b in splits},
               'exact_text_overlap': {f'{a}_{b}': len(rel_texts[a] & rel_texts[b]) for a, b in splits},
               'includes_explicit_null_relations': any(x['relation'].lower() in {'none', 'na', 'no_relation', '无关系'} for d in rel.values() for x in d)},
        'interpretation': [
            'Exact text deduplication does not establish independent engines, original collection sources or non-overlapping templates.',
            'Existing checkpoints have no auditable training provenance sufficient to establish every test case was unseen.',
            'Positive relation labels evaluate relation classification with supplied entities; they do not validate null-relation rejection or causal validity.',
            'The static text corpus has no timestamped fault trajectories or independently annotated root causes.'
        ]
    }
    json_write(output / 'data_audit.json', summary)
    sources = [kg_build / '案例语料.txt', kg_build / 'neo4j_query_table_data.csv',
               kg_build / 'NER/best_roberta_ner_model/best_roberta_ner_model.pt',
               kg_build / 'RE/best_re_model/best_re_model.pt', kg_build / 'RE/best_re_model/rel2id.json']
    sources += [kg_build / f'NER/ner_data_all_generate/{s}.txt' for s in bio]
    sources += [kg_build / f'RE/re_data_generate/{s}.json' for s in rel]
    json_write(output / 'source_manifest.json', [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sources])
    return summary


def filter_predictions(data_dir, model_dir):
    mapping = json.loads((data_dir / 'case_duplicate_mapping.json').read_text())
    keep = {x['case_id'] for x in mapping if not x['is_duplicate']}
    counts = {}
    for name in ['corpus_entities.jsonl', 'corpus_triples.jsonl']:
        if not (model_dir / name).exists():
            counts[name] = {'status': 'source_not_produced', 'interpretation': 'No full-corpus relation output was produced by this run.'}
            continue
        target = data_dir / name
        n_all = n_keep = 0
        with (model_dir / name).open(encoding='utf-8') as source, target.open('w', encoding='utf-8') as out:
            for line in source:
                if not line.strip(): continue
                row = json.loads(line); n_all += 1
                if row['case_id'] in keep:
                    out.write(line); n_keep += 1
        counts[name] = {'raw_rows': n_all, 'unique_case_rows': n_keep}
    json_write(data_dir / 'filtered_predictions_summary.json', counts)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kg-build', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--filter-model-dir', type=Path)
    args = parser.parse_args()
    if args.filter_model_dir:
        result = filter_predictions(args.output_dir, args.filter_model_dir)
    else:
        if args.kg_build is None: parser.error('--kg-build required for audit')
        result = audit(args.kg_build.resolve(), args.output_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
