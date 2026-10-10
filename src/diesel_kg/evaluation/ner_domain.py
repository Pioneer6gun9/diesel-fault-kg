"""Read-only NER domain/annotation-shift diagnostics from frozen BIO and predictions.

Fits descriptive vocabularies and text skeletons on TRAIN ONLY. Test data are
used solely for reporting; no gold, decoding, checkpoint or threshold is changed.
No torch/model execution is needed. Inferences about annotation conventions are
review flags, never authenticated annotation errors or replacement gold.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import sys
import time
import unicodedata


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized_body(text):
    """NFKC and punctuation/whitespace removal only; never alters gold offsets."""
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold()
                   if not c.isspace() and not unicodedata.category(c).startswith('P'))


def corpus_matches(rows, cases):
    exact, normalized = defaultdict(list), defaultdict(list)
    for case in cases:
        exact[case['text']].append(case['case_id'])
        normalized[normalized_body(case['text'])].append(case['case_id'])
    records = []
    body_labels = defaultdict(set)
    for row in rows:
        body = normalized_body(row['text'])
        body_labels[body].add(json.dumps(row['gold_entities'], sort_keys=True))
        records.append({'sample_id': row['sample_id'], 'text': row['text'],
                        'normalized_body_sha256': hashlib.sha256(body.encode()).hexdigest(),
                        'exact_case_ids': exact.get(row['text'], []),
                        'normalized_case_ids': normalized.get(body, [])})
    return {'samples': len(rows), 'unique_normalized_bodies': len(body_labels),
            'exact_matched_samples': sum(bool(r['exact_case_ids']) for r in records),
            'normalized_matched_samples': sum(bool(r['normalized_case_ids']) for r in records),
            'repeated_normalized_body_extra_samples': len(rows)-len(body_labels),
            'normalized_groups_with_different_raw_coordinate_gold': sum(len(v)>1 for v in body_labels.values()),
            'interpretation': 'Coordinate-label differences across normalized text require review; normalized text is never relabelled.'}, records


def read_bio(path):
    rows, units, labels = [], [], []
    for line in Path(path).read_text(encoding='utf-8').splitlines() + ['']:
        if not line.strip():
            if units:
                offsets, cursor = [], 0
                for unit in units:
                    offsets.append([cursor, cursor + len(unit)])
                    cursor += len(unit)
                rows.append({'text': ''.join(units), 'labels': labels, 'offsets': offsets})
                units, labels = [], []
        else:
            fields = line.split('\t', 1)
            if len(fields) != 2 or not fields[0]:
                raise ValueError(f'Invalid BIO record in {path}')
            units.append(fields[0])
            labels.append(fields[1].strip())
    return rows


def spans(labels, offsets):
    result, current = [], None
    for label, (start, end) in zip(labels, offsets):
        if end <= start:
            continue
        kind = label[2:] if label.startswith(('B-', 'I-')) else None
        continuation = (label.startswith('I-') and current is not None and
                        current['type'] == kind and start == current['end'])
        if not continuation:
            if current:
                result.append(current)
            current = {'start': start, 'end': end, 'type': kind} if kind else None
        elif current:
            current['end'] = end
    if current:
        result.append(current)
    return result


def key(entity, with_type=True):
    return (entity['start'], entity['end'], entity['type']) if with_type else (entity['start'], entity['end'])


def with_text(text, entities):
    return [{**e, 'text': text[e['start']:e['end']]} for e in entities]


def skeleton(row):
    parts, cursor = [], 0
    for e in row['gold_entities']:
        if e['start'] < cursor:
            raise ValueError('Overlapping gold spans cannot form a BIO skeleton')
        parts.extend((row['text'][cursor:e['start']], '<' + e['type'] + '>'))
        cursor = e['end']
    return ''.join(parts) + row['text'][cursor:]


def decorate_bio(rows):
    return [{**r, 'sample_id': i, 'gold_entities': spans(r['labels'], r['offsets'])}
            for i, r in enumerate(rows)]


def attach_predictions(rows, path):
    predictions = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows) != len(predictions):
        raise ValueError(f'Prediction count mismatch: {path}')
    for i, (row, pred) in enumerate(zip(rows, predictions)):
        if pred['sample_id'] != i or pred['text'] != row['text']:
            raise ValueError(f'Prediction identity/text mismatch at {path}:{i}')
        if {key(e) for e in pred['gold_entities']} != {key(e) for e in row['gold_entities']}:
            raise ValueError(f'Gold mismatch at {path}:{i}')
        if {key(e) for e in spans(pred['predicted_token_labels'], pred['token_offsets'])} != {
                key(e) for e in pred['predicted_entities']}:
            raise ValueError(f'Strict decoder mismatch at {path}:{i}')
        for e in pred['predicted_entities']:
            if not 0 <= e['start'] < e['end'] <= len(row['text']):
                raise ValueError(f'Prediction offset outside text at {path}:{i}')
        row.update({k: pred[k] for k in ('predicted_entities', 'predicted_token_labels', 'token_offsets')})
    return len(rows)


def fit_train(rows):
    lexicon, templates, character_counts = defaultdict(Counter), Counter(), Counter()
    for row in rows:
        templates[skeleton(row)] += 1
        character_counts.update(row['text'])
        for entity in row['gold_entities']:
            lexicon[entity['type']][row['text'][entity['start']:entity['end']]] += 1
    reverse = defaultdict(set)
    for kind, terms in lexicon.items():
        for text in terms:
            reverse[text].add(kind)
    return {'lexicon': lexicon, 'reverse': reverse, 'templates': templates,
            'characters': character_counts}


def numbers(values):
    ordered = sorted(values)
    if not ordered:
        return {'count': 0, 'mean': None, 'min': None, 'p50': None, 'p95': None, 'max': None}
    return {'count': len(ordered), 'mean': sum(ordered) / len(ordered), 'min': ordered[0],
            'p50': ordered[int((len(ordered)-1) * .5)],
            'p95': ordered[int((len(ordered)-1) * .95)], 'max': ordered[-1]}


def metric(rows, with_type=True):
    gold, predicted = set(), set()
    for row in rows:
        gold.update((row['sample_id'],) + key(e, with_type) for e in row['gold_entities'])
        predicted.update((row['sample_id'],) + key(e, with_type) for e in row.get('predicted_entities', []))
    matched = len(gold & predicted)
    precision = matched / len(predicted) if predicted else 0.
    recall = matched / len(gold) if gold else 0.
    return {'gold': len(gold), 'predicted': len(predicted), 'matched': matched,
            'precision': precision, 'recall': recall,
            'f1': 2*precision*recall/(precision+recall) if precision+recall else 0.}


def overlap(a, b):
    return max(0, min(a['end'], b['end']) - max(a['start'], b['start']))


def gold_error(gold, predictions):
    exact = [p for p in predictions if key(p) == key(gold)]
    if exact:
        return 'exact_match', exact
    boundary = [p for p in predictions if key(p, False) == key(gold, False)]
    if boundary:
        return 'exact_boundary_wrong_type', boundary
    same_type = [p for p in predictions if p['type'] == gold['type'] and overlap(gold, p)]
    if same_type:
        return 'same_type_overlap_boundary_error', same_type
    wrong_type = [p for p in predictions if overlap(gold, p)]
    if wrong_type:
        return 'overlap_only_wrong_type', wrong_type
    return 'no_predicted_overlap', []


def char_labels(text, entities):
    result = ['O'] * len(text)
    for e in entities:
        for i in range(e['start'], e['end']):
            if result[i] != 'O':
                raise ValueError('Entities overlap in character confusion audit')
            result[i] = e['type']
    return result


def profile(rows, train):
    gold_counts = Counter(e['type'] for r in rows for e in r['gold_entities'])
    seen, other, unseen, train_shape = Counter(), Counter(), Counter(), 0
    gold_lengths, predicted_lengths, char_confusion = [], [], Counter()
    errors, invalid_i, prediction_count = Counter(), 0, 0
    novel_characters = Counter()
    for r in rows:
        train_shape += skeleton(r) in train['templates']
        novel_characters.update(c for c in r['text'] if c not in train['characters'])
        for e in r['gold_entities']:
            term, kind = r['text'][e['start']:e['end']], e['type']
            gold_lengths.append(e['end'] - e['start'])
            if term in train['lexicon'].get(kind, {}):
                seen[kind] += 1
            elif term in train['reverse']:
                other[kind] += 1
            else:
                unseen[kind] += 1
            if 'predicted_entities' in r:
                errors[gold_error(e, r['predicted_entities'])[0]] += 1
        if 'predicted_entities' in r:
            prediction_count += 1
            predicted_lengths.extend(e['end']-e['start'] for e in r['predicted_entities'])
            for a, b in zip(char_labels(r['text'], r['gold_entities']),
                            char_labels(r['text'], r['predicted_entities'])):
                char_confusion[a, b] += 1
            previous = 'O'
            for label in r['predicted_token_labels']:
                invalid_i += label.startswith('I-') and previous not in ('B-'+label[2:], 'I-'+label[2:])
                previous = label
    result = {
        'samples': len(rows), 'unique_texts': len({r['text'] for r in rows}),
        'characters': numbers([len(r['text']) for r in rows]),
        'gold_entity_lengths': numbers(gold_lengths), 'predicted_entity_lengths': numbers(predicted_lengths),
        'gold_entities': sum(gold_counts.values()), 'gold_type_counts': dict(gold_counts),
        'gold_entities_per_sample': numbers([len(r['gold_entities']) for r in rows]),
        'training_gold_entity_unique_terms': {k: len(v) for k, v in train['lexicon'].items()},
        'gold_term_seen_in_training_with_same_type': sum(seen.values()),
        'gold_term_seen_in_training_only_other_type': sum(other.values()),
        'gold_term_unseen_in_training': sum(unseen.values()),
        'per_type_term_coverage': {k: {'gold': gold_counts[k], 'seen_same_type': seen[k],
                                    'seen_only_other_type': other[k], 'unseen': unseen[k]} for k in gold_counts},
        'gold_skeleton_seen_in_training_samples': train_shape,
        'gold_skeleton_seen_in_training_fraction': train_shape/len(rows) if rows else 0.,
        'unseen_characters_count': sum(novel_characters.values()),
        'unseen_characters_unique': len(novel_characters),
        'novel_characters': dict(novel_characters.most_common()),
        'invalid_predicted_I_transitions': invalid_i,
        'gold_error_categories': dict(errors),
        'predicted_single_character_entities': sum(x == 1 for x in predicted_lengths),
        'predicted_single_character_entity_fraction': sum(x == 1 for x in predicted_lengths)/len(predicted_lengths) if predicted_lengths else None,
    }
    if prediction_count:
        if prediction_count != len(rows):
            raise ValueError('Partial predictions within a stratum')
        result['strict_span_and_type'] = metric(rows)
        result['exact_span_ignoring_type_diagnostic_only'] = metric(rows, False)
    return result, char_confusion


def annotation_flags(row, train):
    """Review flags using fixed rules and TRAIN statistics, not corrected gold."""
    flags = []
    for e in row['gold_entities']:
        text, kind = row['text'][e['start']:e['end']], e['type']
        base = {**e, 'text': text}
        other = train['reverse'].get(text, set()) - {kind}
        if text in train['reverse'] and kind not in train['reverse'][text]:
            flags.append({'flag': 'exact_term_type_differs_from_training', **base, 'training_types': sorted(other)})
        if kind == '发动机型号' and re.search(r'发动机|柴油机|叉车|压路机|收割机', text):
            flags.append({'flag': 'model_span_includes_machine_or_engine_noun', **base,
                          'interpretation': 'Span convention review only; inclusion may be intentional.'})
        if kind == '影响后果' and re.search(r'正常|恢复|减轻|解决|消失|改善|合格', text):
            flags.append({'flag': 'consequence_label_contains_recovery_language', **base,
                          'interpretation': 'Recovery should be distinguished from fault consequence for propagation; no label changed.'})
        embedded = []
        for term, kinds in train['reverse'].items():
            if len(term) > 1 and len(term) < len(text) and term in text:
                embedded.append({'training_term': term, 'training_types': sorted(kinds),
                                 'same_type': kind in kinds})
        if embedded:
            flags.append({'flag': 'gold_span_contains_shorter_training_entity', **base,
                          'embedded_training_entities': embedded,
                          'interpretation': 'Possible granularity difference, not proof of an annotation error.'})
    return flags


def sample_record(row, stratum, train):
    text = row['text']
    errors = []
    for entity in row['gold_entities']:
        term = text[entity['start']:entity['end']]
        category, matches = gold_error(entity, row.get('predicted_entities', []))
        errors.append({**entity, 'text': term, 'error_category': category,
                       'seen_training_same_type': term in train['lexicon'].get(entity['type'], {}),
                       'training_types_for_exact_term': sorted(train['reverse'].get(term, set())),
                       'overlapping_predictions': with_text(text, matches)})
    return {'stratum': stratum, 'sample_id': row['sample_id'], 'text': text,
            'raw_text_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'gold_entities': with_text(text, row['gold_entities']),
            'predicted_entities': with_text(text, row.get('predicted_entities', [])),
            'gold_skeleton': skeleton(row), 'gold_skeleton_seen_in_training': skeleton(row) in train['templates'],
            'gold_error_decomposition': errors if 'predicted_entities' in row else None,
            'annotation_convention_review_flags': annotation_flags(row, train),
            'scope': 'Existing supervised labels preserved verbatim; flags and overlaps are descriptive, not new gold.'}


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def dump_jsonl(path, rows):
    with Path(path).open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def dump_csv(path, rows, keys):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--predictions-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=Path, help='Original numbered corpus JSONL for overlap audit')
    parser.add_argument('--test-stratum-boundary', type=int, default=800)
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = {s: args.data/(s+'.txt') for s in ('train', 'dev', 'test')}
    pred_paths = {s: args.predictions_dir/(s+'_predictions.jsonl') for s in ('dev', 'test')}
    rows = {s: decorate_bio(read_bio(p)) for s, p in inputs.items()}
    checked_predictions = {s: attach_predictions(rows[s], p) for s, p in pred_paths.items()}
    if not 0 < args.test_stratum_boundary < len(rows['test']):
        raise ValueError('Test stratum boundary must lie inside test records')
    fitted = fit_train(rows['train'])
    strata = {'train': rows['train'], 'dev': rows['dev'], 'test_all': rows['test'],
              'test_initial_'+str(args.test_stratum_boundary): rows['test'][:args.test_stratum_boundary],
              'test_appended_'+str(len(rows['test'])-args.test_stratum_boundary): rows['test'][args.test_stratum_boundary:]}
    profiles, confusion_rows, type_rows, coverage_rows, split_rows, error_rows = {}, [], [], [], [], []
    for name, records in strata.items():
        profiles[name], confusion = profile(records, fitted)
        p = profiles[name]
        for (g, pred), count in sorted(confusion.items()):
            confusion_rows.append({'stratum': name, 'gold_type': g, 'predicted_type': pred, 'characters': count})
        for kind in sorted(set(fitted['lexicon']) | set(p['gold_type_counts'])):
            n = p['gold_type_counts'].get(kind, 0)
            type_rows.append({'stratum': name, 'type': kind, 'gold_entities': n,
                              'gold_fraction': n/p['gold_entities'] if p['gold_entities'] else 0.})
            coverage = p['per_type_term_coverage'].get(kind, {'gold': 0, 'seen_same_type': 0, 'seen_only_other_type': 0, 'unseen': 0})
            coverage_rows.append({'stratum': name, 'type': kind, **coverage})
        for category, count in p['gold_error_categories'].items():
            error_rows.append({'stratum': name, 'category': category, 'gold_entities': count})
        split_rows.append({'stratum': name, 'samples': p['samples'], 'gold_entities': p['gold_entities'],
                           'mean_text_length': p['characters']['mean'], 'mean_gold_length': p['gold_entity_lengths']['mean'],
                           'training_template_fraction': p['gold_skeleton_seen_in_training_fraction'],
                           'training_same_type_term_fraction': p['gold_term_seen_in_training_with_same_type']/p['gold_entities'],
                           'predicted_one_char_fraction': p['predicted_single_character_entity_fraction'],
                           'strict_f1': p.get('strict_span_and_type', {}).get('f1'),
                           'boundary_only_f1': p.get('exact_span_ignoring_type_diagnostic_only', {}).get('f1')})
    overlap_checks = {}
    for s, records in rows.items():
        others = {other: {r['text'] for r in other_rows} for other, other_rows in rows.items() if other != s}
        overlap_checks[s] = {other: sum(r['text'] in text_set for r in records) for other, text_set in others.items()}
    train_templates = [{'skeleton': k, 'training_samples': n} for k, n in fitted['templates'].most_common()]
    dump_json(args.output/'training_lexicon.json', {k: dict(v.most_common()) for k, v in fitted['lexicon'].items()})
    dump_json(args.output/'training_templates.json', train_templates)
    dump_csv(args.output/'split_profiles.csv', split_rows, list(split_rows[0]))
    dump_csv(args.output/'entity_type_distribution.csv', type_rows, list(type_rows[0]))
    dump_csv(args.output/'training_entity_coverage.csv', coverage_rows, list(coverage_rows[0]))
    dump_csv(args.output/'gold_error_decomposition.csv', error_rows, ['stratum', 'category', 'gold_entities'])
    dump_csv(args.output/'character_type_confusion.csv', confusion_rows, ['stratum', 'gold_type', 'predicted_type', 'characters'])
    all_diagnostics = (sample_record(row, name, fitted) for name in ('train', 'dev', 'test_all') for row in strata[name])
    dump_jsonl(args.output/'all_sample_diagnostics.jsonl', all_diagnostics)
    appended_name = 'test_appended_'+str(len(rows['test'])-args.test_stratum_boundary)
    appended_diagnostics = [sample_record(row, appended_name, fitted) for row in strata[appended_name]]
    dump_jsonl(args.output/'appended_domain_review_queue.jsonl', appended_diagnostics)
    corpus_overlap = None
    if args.cases:
        cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
        corpus_overlap, matched = corpus_matches(strata[appended_name], cases)
        corpus_overlap['cases'] = len(cases)
        dump_jsonl(args.output/'appended_original_corpus_matches.jsonl', matched)
    flag_counts = Counter(flag['flag'] for r in appended_diagnostics for flag in r['annotation_convention_review_flags'])
    input_manifest = {str(p): sha256(p) for p in [*inputs.values(), *pred_paths.values()]}
    if args.cases:
        input_manifest[str(args.cases)] = sha256(args.cases)
    summary = {'status': 'completed', 'execution': {'argv': sys.argv, 'script_path': str(Path(__file__).resolve()),
               'script_sha256': sha256(__file__), 'python': sys.executable, 'elapsed_seconds': time.monotonic()-started},
               'input_sha256': input_manifest,
               'invariants': {'gold_unchanged': True, 'saved_prediction_raw_text_gold_and_strict_decoder_checks': checked_predictions,
                              'descriptive_lexicon_and_templates_fit_on_train_only': True,
                              'model_forward_or_training_executed': False, 'test_used_for_tuning': False,
                              'independent_annotation_provenance_available': False},
               'profiles': profiles, 'exact_text_overlap': overlap_checks,
               'training_unique_gold_terms': len(fitted['reverse']), 'training_unique_gold_skeletons': len(fitted['templates']),
               'training_top10_skeleton_samples': sum(t['training_samples'] for t in train_templates[:10]),
               'appended_review_flag_counts': dict(flag_counts),
               'appended_original_corpus_overlap': corpus_overlap,
               'interpretation': [
                   'Exact-text non-overlap does not establish generalization when label-replaced template skeletons and tiny entity vocabularies recur.',
                   'Character overlap or span-only scores are diagnostics, not replacements for the primary strict span-and-type score.',
                   'Unknown gold provenance and review flags do not establish that the supplied labels are wrong.',
                   'A decoder selected on test or replacing gold with predictions would not repair the evaluation.',
                   'Any future adaptation must select changes on train/dev and keep the 90 already inspected test records fixed as a disclosed diagnostic stratum; new independent generalization claims require independent labels.'
               ]}
    dump_json(args.output/'summary.json', summary)
    md = ['# NER 领域与标注差异诊断', '',
          '本报告仅读取已固定的 BIO 数据与模型预测。词表和句式统计只用训练集；没有修改标签、模型、阈值或解码。', '',
          '| 分组 | 样本 | 训练句式覆盖 | 训练同类词条覆盖 | 严格跨度及类型 F1 | 仅边界 F1 | 预测单字比例 |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for p in split_rows:
        fmt = lambda x: '—' if x is None else f'{x:.4f}'
        md.append('| '+p['stratum']+' | '+str(p['samples'])+' | '+fmt(p['training_template_fraction'])+' | '+
                  fmt(p['training_same_type_term_fraction'])+' | '+fmt(p['strict_f1'])+' | '+fmt(p['boundary_only_f1'])+' | '+fmt(p['predicted_one_char_fraction'])+' |')
    md.extend(['', f'训练集共有 {len(fitted["reverse"])} 个不同实体词条、{len(fitted["templates"])} 种标签替换句式。前 10 种句式覆盖 {summary["training_top10_skeleton_samples"]}/{len(rows["train"])} 条训练记录。', '',
               '追加测试的每一条原文、原始标签、预测跨度、错误类别和标注惯例复核提示保存在 appended_domain_review_queue.jsonl。提示不构成新 gold。', '',
               '后续可以检查训练数据覆盖、在 train/dev 上选择解码或训练调整，并分别报告模板测试与领域测试。仅增加训练轮次、改变测试标签或按这 90 条测试记录编规则，不能证明真实领域泛化。'])
    (args.output/'诊断说明.md').write_text('\n'.join(md)+'\n')
    content = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>NER 领域差异诊断</title><style>body{font:16px system-ui;max-width:1100px;margin:36px auto;line-height:1.6}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px}pre{white-space:pre-wrap;word-break:break-word}summary{cursor:pointer}details{margin:16px 0}</style><h1>NER 领域差异诊断</h1><p>词表与句式只使用训练集。原始 gold 和模型预测保持不变；以下差异是审计证据，不是自动改标注。</p><table><tr><th>分组</th><th>样本</th><th>训练句式覆盖</th><th>训练同类词条覆盖</th><th>严格 F1</th><th>仅边界 F1</th><th>预测单字比例</th></tr>'
    for p in split_rows:
        content += '<tr>'+''.join('<td>'+html.escape(str(p[k]) if k in ('stratum', 'samples') else fmt(p[k]))+'</td>' for k in ('stratum','samples','training_template_fraction','training_same_type_term_fraction','strict_f1','boundary_only_f1','predicted_one_char_fraction'))+'</tr>'
    content += '</table><h2>追加领域测试逐条证据</h2>'
    for r in appended_diagnostics:
        content += '<details><summary>'+str(r['sample_id'])+' — '+html.escape(r['text'])+'</summary><pre>'+html.escape(json.dumps(r, ensure_ascii=False, indent=2))+'</pre></details>'
    content += '</html>'
    (args.output/'NER领域差异诊断.html').write_text(content)
    # Live stdout/stderr logs can change after this manifest is written.
    artifacts = {p.name: {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in args.output.iterdir()
                 if p.is_file() and p.name != 'artifact_manifest.json' and p.suffix != '.log'}
    dump_json(args.output/'artifact_manifest.json', {'status': 'completed', 'artifacts': artifacts})
    print(json.dumps({'status': 'completed', 'training_unique_terms': summary['training_unique_gold_terms'],
                      'training_templates': summary['training_unique_gold_skeletons'],
                      'profiles': {k: {field: v.get(field) for field in ('samples', 'gold_term_seen_in_training_with_same_type', 'gold_term_unseen_in_training', 'gold_skeleton_seen_in_training_fraction', 'strict_span_and_type', 'exact_span_ignoring_type_diagnostic_only', 'gold_error_categories')} for k, v in profiles.items()},
                      'appended_review_flag_counts': dict(flag_counts)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
