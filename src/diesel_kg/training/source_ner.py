"""Faithful NER replay and an explicit encoder-identity repair, offline.

Original EntityExtraction.py is never imported or changed. Its dataset/model
classes are compiled from its AST. The faithful arm retains RobertaModel; the
repair arm changes only that encoder constructor to BertModel because the
official HFL Chinese RoBERTa-WWM checkpoint is a BERT-architecture checkpoint.
Selection and early stopping retain the original dev token entity micro F1.
Exact character span plus entity type is also measured, but never used to
choose a checkpoint on the held-out test set.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, RobertaModel, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support

from diesel_kg.models.loading import bio_spans, file_sha256, load_tokenizer, read_bio


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def source_definitions(source, encoder_class=RobertaModel):
    text = source.read_text(encoding='utf-8-sig')
    tree = ast.parse(text)
    names = {'NERDataset', 'build_label_mapping', 'RoBERTaOnlyNER'}
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in selected} != names:
        raise ValueError('Unexpected original NER source definitions')
    namespace = {'torch': torch, 'nn': nn, 'Dataset': Dataset, 'RobertaModel': encoder_class,
                 'logger': logging.getLogger('ner_replay')}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace, {node.name: hashlib.sha256(ast.get_source_segment(text, node).encode()).hexdigest() for node in selected}


def trim_collate(items):
    batch = {key: torch.stack([item[key] for item in items]) for key in items[0]}
    # Remove only columns that are padding in every item. Attention masks and
    # all real-token positions remain exactly as in the source max_length=512.
    width = int(batch['attention_mask'].sum(1).max())
    return {key: value[:, :width] for key, value in batch.items()}


def logits(model, ids, mask):
    hidden = model.roberta(input_ids=ids, attention_mask=mask).last_hidden_state
    return model.classifier(model.dropout(hidden))


def evaluate(model, loader, rows, id2label, device, output=None):
    model.eval()
    gold, predicted, token_gold, token_predicted = set(), set(), [], []
    cursor, truncated, invalid_i = 0, 0, 0
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            probabilities = logits(model, ids, mask).softmax(-1).cpu()
            scores, labels = probabilities.max(-1)
            raw_rows = rows[cursor:cursor + len(ids)]
            # Source NERDataset strips each BIO line, dropping whitespace
            # characters. Keep its original token metric, but run a separate
            # raw-text forward for original-character exact-span evaluation.
            raw_encoded = loader.dataset.tokenizer([row['text'] for row in raw_rows], padding=True,
                               truncation=True, max_length=512, return_offsets_mapping=True, return_tensors='pt')
            raw_offsets = raw_encoded.pop('offset_mapping').tolist()
            raw_probabilities = logits(model, raw_encoded['input_ids'].to(device), raw_encoded['attention_mask'].to(device)).softmax(-1).cpu()
            raw_scores, raw_labels = raw_probabilities.max(-1)
            for local in range(len(ids)):
                row = rows[cursor]
                valid_count = int(batch['attention_mask'][local].sum())
                # Same valid-token evaluation positions as the source.
                true_ids = batch['label_ids'][local, 1:valid_count - 1].tolist()
                pred_ids = labels[local, 1:valid_count - 1].tolist()
                token_gold.extend(id2label[i] for i in true_ids)
                token_predicted.extend(id2label[i] for i in pred_ids)
                offsets = raw_offsets[local]
                valid = [(i, tuple(span)) for i, span in enumerate(offsets) if span[1] > span[0]]
                p_labels = [id2label[int(raw_labels[local, i])] for i, _ in valid]
                p_offsets = [span for _, span in valid]
                p_scores = [float(raw_scores[local, i]) for i, _ in valid]
                predicted_spans = bio_spans(p_labels, p_offsets, p_scores)
                gold_spans = bio_spans(row['labels'], row['offsets'])
                for span in gold_spans:
                    gold.add((cursor, span['start'], span['end'], span['type']))
                for span in predicted_spans:
                    predicted.add((cursor, span['start'], span['end'], span['type']))
                previous = 'O'
                for label in p_labels:
                    invalid_i += int(label.startswith('I-') and previous not in ('B-' + label[2:], 'I-' + label[2:]))
                    previous = label
                visible = max((end for _, end in p_offsets), default=0)
                truncated += int(visible < len(row['text']))
                if output:
                    predictions.append({'sample_id': cursor, 'text': row['text'],
                                        'gold_entities': [{**span, 'text': row['text'][span['start']:span['end']]} for span in gold_spans],
                                        'predicted_entities': [{**span, 'text': row['text'][span['start']:span['end']]} for span in predicted_spans],
                                        'predicted_token_labels': p_labels, 'token_offsets': p_offsets})
                cursor += 1
    entity_labels = [id2label[i] for i in sorted(id2label) if id2label[i] != 'O']
    p, r, f1, _ = precision_recall_fscore_support(token_gold, token_predicted, labels=entity_labels, average='micro', zero_division=0)
    matched = len(gold & predicted)
    span_p, span_r = matched / len(predicted) if predicted else 0., matched / len(gold) if gold else 0.
    by_type = {}
    for kind in sorted({item[3] for item in gold | predicted}):
        g, q = {item for item in gold if item[3] == kind}, {item for item in predicted if item[3] == kind}
        n = len(g & q); pp, rr = n / len(q) if q else 0., n / len(g) if g else 0.
        by_type[kind] = {'gold': len(g), 'predicted': len(q), 'matched': n, 'precision': pp, 'recall': rr,
                         'f1': 2 * pp * rr / (pp + rr) if pp + rr else 0.}
    result = {'samples': cursor, 'token_entity_micro_precision': float(p), 'token_entity_micro_recall': float(r),
              'token_entity_micro_f1': float(f1), 'exact_span_type': {'precision': span_p, 'recall': span_r,
              'f1': 2 * span_p * span_r / (span_p + span_r) if span_p + span_r else 0.,
              'gold': len(gold), 'predicted': len(predicted), 'matched': matched, 'per_type': by_type},
              'truncated_samples': truncated, 'invalid_I_predictions': invalid_i,
              'token_metric_input': 'Original NERDataset cleaned text, preserving source line-strip behavior',
              'exact_span_input': 'Separate forward on unmodified original BIO text and character offsets',
              'span_decoder': 'I without matching preceding B/I starts a new entity, contiguous offsets required; identical in every arm',
              'evidence_boundary': 'Provided supervised BIO split; not independently annotated industrial case validation.'}
    if output:
        with output.open('w', encoding='utf-8') as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    return result


def equivalence_check(model, dataset, device):
    model.eval()
    full = {key: torch.stack([dataset[i][key] for i in range(2)]) for key in dataset[0]}
    short = trim_collate([dataset[i] for i in range(2)])
    width = short['input_ids'].shape[1]
    with torch.inference_mode():
        a = logits(model, full['input_ids'].to(device), full['attention_mask'].to(device))[:, :width]
        b = logits(model, short['input_ids'].to(device), short['attention_mask'].to(device))
        active = short['attention_mask'].to(device).bool()
        error = float((a[active] - b[active]).abs().max())
        loss_a = model(full['input_ids'].to(device), full['attention_mask'].to(device), full['label_ids'].to(device))
        loss_b = model(short['input_ids'].to(device), short['attention_mask'].to(device), short['label_ids'].to(device))
    result = {'full_width': 512, 'cropped_width': width, 'active_logit_max_abs_error': error,
              'loss_full': float(loss_a), 'loss_cropped': float(loss_b),
              'loss_abs_error': float((loss_a - loss_b).abs()), 'tolerance': 3e-5,
              'train_dropout_note': 'Cropping preserves effective-token computation and loss; train-mode dropout RNG tensor shape changes. Original code does not fix a seed.'}
    if error > 3e-5 or result['loss_abs_error'] > 3e-5:
        raise RuntimeError(f'Padding equivalence check failed: {result}')
    return result


def train(args, variant):
    start = time.perf_counter()
    root = args.output / variant
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'summary.json').exists() and not args.force:
        print(f'{variant}: complete summary already exists; leaving frozen result unchanged', flush=True)
        return json.loads((root / 'summary.json').read_text())
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    namespace, hashes = source_definitions(args.source, RobertaModel if variant == 'source_roberta' else BertModel)
    label2id, id2label, label_list = namespace['build_label_mapping'](str(args.data / 'train.txt'))
    tokenizer = load_tokenizer(args.pretrained)
    datasets = {split: namespace['NERDataset'](str(args.data / (split + '.txt')), tokenizer, 512, label2id)
                for split in ('train', 'dev', 'test')}
    rows = {split: read_bio(args.data / (split + '.txt')) for split in datasets}
    loaders = {split: DataLoader(dataset, batch_size=8, shuffle=split == 'train', num_workers=0, collate_fn=trim_collate)
               for split, dataset in datasets.items()}
    # Source constructor, freezing order, head initialization, loss are unchanged.
    model = namespace['RoBERTaOnlyNER'](str(args.pretrained), len(label_list), 0.2).to(device)
    config = {'variant': variant, 'device': str(device), 'seed_added_for_reproducibility': args.seed,
              'source_path': str(args.source), 'source_sha256': file_sha256(args.source), 'class_sha256': hashes,
              'pretrained_directory': str(args.pretrained), 'pretrained_manifest': json.loads((args.pretrained / 'asset_manifest.json').read_text()),
              'encoder_class': model.roberta.__class__.__name__, 'encoder_config': model.roberta.config.to_dict(),
              'label2id': label2id, 'data': {split: {'path': str(args.data / (split + '.txt')),
              'sha256': file_sha256(args.data / (split + '.txt')), 'samples': len(dataset)} for split, dataset in datasets.items()},
              'max_seq_len': 512, 'batch_size': 8, 'learning_rate': 2e-5, 'epochs': 15,
              'patience': 5, 'dropout': 0.2, 'weight_decay': 0.01, 'warmup_ratio': 0.1,
              'clip_grad_norm': 1.0, 'selection_metric': 'dev token entity micro F1, source definition',
              'parameters': sum(p.numel() for p in model.parameters()),
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'trainable_parameter_names': [name for name, p in model.named_parameters() if p.requires_grad],
              'implementation_difference': ['Local offline official HFL assets', 'MPS instead of CUDA/CPU auto',
                  'Fixed seed42 added', 'Only trailing all-PAD batch columns omitted; equivalence tested',
                  'Additional exact-span evaluation; source dev token metric selects best',
                  *(['RobertaModel constructor replaced with BertModel only'] if variant != 'source_roberta' else [])]}
    save_json(root / 'config.json', config)
    save_json(root / 'padding_equivalence.json', equivalence_check(model, datasets['dev'], device))
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5, weight_decay=0.01)
    total_steps = len(loaders['train']) * 15
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    best_f1, best_epoch, patience = -1., 0, 0
    history = []
    best = root / 'best_roberta_ner_model.pt'
    for epoch in range(1, 16):
        begin = time.perf_counter(); model.train(); losses = []
        for step, batch in enumerate(loaders['train'], 1):
            optimizer.zero_grad()
            loss = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label_ids'].to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); scheduler.step(); losses.append(float(loss.detach()))
            if step % 100 == 0:
                print(f'{variant} epoch={epoch} step={step}/{len(loaders["train"])} loss={np.mean(losses):.6f} elapsed={time.perf_counter()-begin:.1f}s', flush=True)
        metrics = evaluate(model, loaders['dev'], rows['dev'], id2label, device)
        f1 = metrics['token_entity_micro_f1']
        improved = f1 > best_f1
        if improved:
            best_f1, best_epoch, patience = f1, epoch, 0
            torch.save({'model_state_dict': model.state_dict(), 'label2id': label2id, 'id2label': id2label,
                        'epoch': epoch, 'best_f1': best_f1, 'variant': variant,
                        'config': config, 'selection_metric': config['selection_metric']}, best)
        else:
            patience += 1
        item = {'epoch': epoch, 'mean_train_loss': float(np.mean(losses)), 'dev': metrics,
                'best_by_source_metric': improved, 'elapsed_seconds': time.perf_counter() - begin,
                'learning_rate': scheduler.get_last_lr()[0], 'early_stop_counter': patience}
        history.append(item); save_json(root / 'epoch_history.json', history)
        print(f'{variant} epoch={epoch} dev_tokenF1={f1:.6f} dev_spanF1={metrics["exact_span_type"]["f1"]:.6f} time={item["elapsed_seconds"]:.1f}s best={best_epoch}', flush=True)
        if patience >= 5: break
    saved = torch.load(best, map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model_state_dict'], strict=True); del saved
    test = evaluate(model, loaders['test'], rows['test'], id2label, device, root / 'test_predictions.jsonl')
    dev = evaluate(model, loaders['dev'], rows['dev'], id2label, device, root / 'dev_predictions.jsonl')
    summary = {'status': 'completed', 'variant': variant, 'best_epoch': best_epoch, 'epochs_run': len(history),
               'best_dev_token_micro_f1': best_f1, 'dev': dev, 'test': test,
               'checkpoint_path': str(best), 'checkpoint_sha256': file_sha256(best),
               'elapsed_seconds': time.perf_counter() - start,
               'selection_boundary': 'Only provided dev token F1 selects checkpoint; test used once after selection.'}
    save_json(root / 'summary.json', summary)
    del model, optimizer, scheduler
    if device.type == 'mps': torch.mps.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--pretrained', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=['source_roberta', 'bert_identity_repair', 'both'], default='both')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(args.output / 'ner_training.log', encoding='utf-8'), logging.StreamHandler()])
    variants = ['source_roberta', 'bert_identity_repair'] if args.variant == 'both' else [args.variant]
    summaries = {variant: train(args, variant) for variant in variants}
    save_json(args.output / 'training_comparison.json', summaries)


if __name__ == '__main__': main()
