"""Audit official encoder identity, BIO alignment and supplied checkpoint metrics."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import torch
from torch.utils.data import DataLoader
from transformers import BertModel, RobertaModel
from diesel_kg.models.loading import bio_spans, file_sha256, load_tokenizer, read_bio, predict_ner
from diesel_kg.training.source_ner import evaluate, source_definitions, trim_collate
from diesel_kg.inference.ner import load_ner
from diesel_kg.paths import resolve_recorded_path


def add_stratified_test_metrics(run_directory, boundary=800):
    """Post-selection only; initial800 template records and appended90 records.

    This reports an observed file stratum, not independently authenticated gold.
    Nothing in this function changes model selection, training or predictions.
    """
    run_directory = Path(run_directory)
    prediction_path = run_directory / 'test_predictions.jsonl'
    rows = [json.loads(line) for line in prediction_path.read_text().splitlines() if line.strip()]
    strata = {}
    for name, first, last in [('initial800_template_style', 0, boundary), ('appended90_domain_style', boundary, len(rows))]:
        gold, predicted = set(), set()
        for row in rows[first:last]:
            sid = row['sample_id']
            for item in row['gold_entities']: gold.add((sid, item['start'], item['end'], item['type']))
            for item in row['predicted_entities']: predicted.add((sid, item['start'], item['end'], item['type']))
        matched = len(gold & predicted)
        p, r = matched / len(predicted) if predicted else 0., matched / len(gold) if gold else 0.
        strata[name] = {'samples': max(0, last - first), 'gold': len(gold), 'predicted': len(predicted),
                       'matched': matched, 'precision': p, 'recall': r,
                       'f1': 2 * p * r / (p + r) if p + r else 0.}
    strata['interpretation'] = 'Observed ordered-file strata only; appended cases do not have independent annotation-provenance evidence.'
    strata['frozen_test_predictions_sha256'] = file_sha256(prediction_path)
    (run_directory / 'test_strata.json').write_text(json.dumps(strata, ensure_ascii=False, indent=2) + '\n')
    summary_path = run_directory / 'summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text()); summary['test_strata'] = strata
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return strata


def checkpoint_reload_checks(run_directory, pretrained_directory):
    run_directory = Path(run_directory)
    torch.set_num_threads(2)
    summary = json.loads((run_directory / 'summary.json').read_text())
    checkpoint = run_directory / 'best_roberta_ner_model.pt'
    if not checkpoint.is_file():
        checkpoint = resolve_recorded_path(summary['checkpoint_path'])
    if file_sha256(checkpoint) != summary['checkpoint_sha256']:
        raise ValueError('Completed NER checkpoint hash differs from its frozen summary')
    model, tokenizer, labels, variant = load_ner(checkpoint, Path(pretrained_directory), torch.device('cpu'))
    rows = [json.loads(line) for line in (run_directory / 'test_predictions.jsonl').read_text().splitlines() if line.strip()]
    selected = rows[:5] + rows[800:805]
    predictions = predict_ner(model, tokenizer, [row['text'] for row in selected], labels, torch.device('cpu'))
    checks = [{'sample_id': row['sample_id'], 'labels_identical': prediction['labels'] == row['predicted_token_labels'],
               'offsets_identical': [list(pair) for pair in prediction['offsets']] == row['token_offsets']}
              for row, prediction in zip(selected, predictions)]
    result = {'variant': variant, 'checkpoint_sha256': summary['checkpoint_sha256'], 'strict_state_match': True,
              'cpu_reload_vs_training_evaluation': checks,
              'all_selected_token_labels_and_offsets_identical': all(c['labels_identical'] and c['offsets_identical'] for c in checks),
              'scope': 'Ten frozen test predictions across initial800 and appended90; equality check only, no tuning.'}
    (run_directory / 'checkpoint_reload_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    if not result['all_selected_token_labels_and_offsets_identical']:
        raise AssertionError('NER checkpoint reloader changed held-out predictions')
    return result


def training_plot(run_directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run_directory = Path(run_directory)
    history = json.loads((run_directory / 'epoch_history.json').read_text())
    summary = json.loads((run_directory / 'summary.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    epochs = [row['epoch'] for row in history]
    axes[0].plot(epochs, [row['mean_train_loss'] for row in history], marker='o', color='#3b6e96')
    axes[0].set(xlabel='Epoch', ylabel='Training cross-entropy')
    axes[1].plot(epochs, [row['dev']['token_entity_micro_f1'] for row in history], marker='o', label='Source dev token F1')
    axes[1].plot(epochs, [row['dev']['exact_span_type']['f1'] for row in history], marker='s', label='Raw-text dev exact span + type F1')
    axes[1].axvline(summary['best_epoch'], color='#777777', linestyle=':', label='Best by source token F1')
    axes[1].set(xlabel='Epoch', ylabel='F1', ylim=(0, 1.02)); axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle(summary['variant']); fig.tight_layout()
    for extension in ('png', 'svg'): fig.savefig(run_directory / ('training_curve.' + extension), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--pretrained', type=Path)
    parser.add_argument('--supplied-checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--summarize-trained', nargs='+', type=Path)
    args = parser.parse_args()
    if args.summarize_trained:
        for directory in args.summarize_trained:
            print(json.dumps(add_stratified_test_metrics(directory), ensure_ascii=False), flush=True)
            if args.pretrained:
                print(json.dumps(checkpoint_reload_checks(directory, args.pretrained), ensure_ascii=False), flush=True)
            training_plot(directory)
        return
    if any(getattr(args, name) is None for name in ('source', 'data', 'pretrained', 'supplied_checkpoint', 'output')):
        parser.error('Full source audit requires --source, --data, --pretrained, --supplied-checkpoint and --output')
    args.output.mkdir(parents=True, exist_ok=True); torch.set_num_threads(4)
    tokenizer = load_tokenizer(args.pretrained)
    records = {split: read_bio(args.data / (split + '.txt')) for split in ('train', 'dev', 'test')}
    counts = {}
    for split, rows in records.items():
        invalid_i, multiunit, unmatched_roundtrip, multi_character_tokens = 0, 0, 0, 0
        for row in rows:
            previous = 'O'
            for label in row['labels']:
                invalid_i += int(label.startswith('I-') and previous not in ('B-' + label[2:], 'I-' + label[2:]))
                previous = label
            multiunit += sum(end - start != 1 for start, end in row['offsets'])
            encoded = tokenizer(row['text'], return_offsets_mapping=True)
            spans = [(a, b) for a, b in encoded['offset_mapping'] if b > a]
            multi_character_tokens += sum(b - a > 1 for a, b in spans)
            unmatched_roundtrip += sum(not row['text'][a:b] for a, b in spans)
        counts[split] = {'samples': len(rows), 'unique_texts': len({row['text'] for row in rows}),
                         'min_characters': min(len(row['text']) for row in rows),
                         'max_characters': max(len(row['text']) for row in rows),
                         'mean_characters': sum(len(row['text']) for row in rows) / len(rows),
                         'multi_character_BIO_units': multiunit, 'invalid_I_gold_transitions': invalid_i,
                         'whitespace_BIO_units_dropped_by_original_dataset': sum(row['text'][a:b].isspace() for row in rows for a,b in row['offsets']),
                         'multi_character_wordpiece_tokens': multi_character_tokens,
                         'empty_offset_slice_checks_failed': unmatched_roundtrip,
                         'exact_text_overlap_with_other_splits': {other: sum(row['text'] in {x['text'] for x in other_rows} for row in rows)
                                                                 for other, other_rows in records.items() if other != split},
                         'sha256': file_sha256(args.data / (split + '.txt'))}
    official_config = json.loads((args.pretrained / 'config.json').read_text())
    identity = {}
    for name, cls in [('source_roberta', RobertaModel), ('bert_identity_repair', BertModel)]:
        torch.manual_seed(42)
        model, info = cls.from_pretrained(str(args.pretrained), local_files_only=True, output_loading_info=True)
        encoder_keys = set(model.state_dict())
        missing = set(info['missing_keys'])
        identity[name] = {'class': model.__class__.__name__, 'configuration_class': model.config.__class__.__name__,
                          'model_type': model.config.model_type, 'pad_token_id': model.config.pad_token_id,
                          'encoder_state_tensor_count': len(encoder_keys), 'missing_tensor_count': len(missing),
                          'missing_keys': sorted(missing), 'unexpected_keys': info['unexpected_keys'],
                          'mismatched_keys': info.get('mismatched_keys', []),
                          'loader_error_messages': info.get('error_msgs', []),
                          'loaded_encoder_tensors': len(encoder_keys - missing)}
        del model
    namespace, hashes = source_definitions(args.source)
    supplied = torch.load(args.supplied_checkpoint, map_location='cpu', weights_only=True, mmap=True)
    label2id = supplied['label2id']; old_epoch = supplied.get('epoch'); old_f1 = supplied.get('best_f1')
    official_state = torch.load(args.pretrained / 'pytorch_model.bin', map_location='cpu', weights_only=True)
    weights_comparison = {}
    for suffix in ['embeddings.word_embeddings.weight', 'embeddings.position_embeddings.weight',
                   'encoder.layer.0.attention.self.query.weight']:
        a = official_state['bert.' + suffix].float().reshape(-1)
        b = supplied['model_state_dict']['roberta.' + suffix].float().reshape(-1)
        weights_comparison[suffix] = {'shape': list(official_state['bert.' + suffix].shape),
            'exact_equal': bool(torch.equal(a, b)),
            'cosine': float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
            'mean_absolute_difference': float((a - b).abs().mean()),
            'max_absolute_difference': float((a - b).abs().max())}
    (args.output / 'provided_encoder_vs_official.json').write_text(json.dumps(weights_comparison, indent=2) + '\n')
    del official_state, a, b
    checkpoint_summary = {'sha256': file_sha256(args.supplied_checkpoint), 'label2id': label2id,
                          'saved_epoch': old_epoch, 'saved_best_f1': old_f1,
                          'state_dictionary_keys': len(supplied['model_state_dict']),
                          'strict_state_match': True}
    del supplied
    model, _, id2label, variant = load_ner(args.supplied_checkpoint, args.pretrained, torch.device(args.device))
    dataset = namespace['NERDataset'](str(args.data / 'test.txt'), tokenizer, 512, label2id)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=trim_collate)
    checkpoint_summary['test'] = evaluate(model, loader, records['test'], id2label, torch.device(args.device),
                                          args.output / 'supplied_checkpoint_test_predictions.jsonl')
    audit = {'source_sha256': file_sha256(args.source), 'compiled_class_sha256': hashes,
             'official_checkpoint_config': official_config, 'initialization_loading_audit': identity,
             'data_audit': counts, 'tokenizer': {'vocabulary_sha256': file_sha256(args.pretrained / 'vocab.txt'),
             'vocab_size': tokenizer.vocab_size, 'pad_token_id': tokenizer.pad_token_id,
             'tokenizer_family': 'BertTokenizerFast, matching source and official HFL BERT architecture'},
             'provided_checkpoint': checkpoint_summary,
             'provided_encoder_vs_official': weights_comparison,
             'source_metric_bug': 'evaluate assigns precision_micro and recall_micro using f1_score; new outputs use proper precision and recall.',
             'original_log_mismatch': 'Provided historical log describes 13 labels, test800 and20 epochs; current source/data describes17 labels, test890 and15 epochs.',
             'gold_scope': 'Supplied BIO records are supervised extraction labels; external domain-case validity remains unmeasured.'}
    (args.output / 'ner_source_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'identity_counts': {name: item['loaded_encoder_tensors'] for name, item in identity.items()},
                      'supplied_test_span_f1': checkpoint_summary['test']['exact_span_type']['f1']}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
