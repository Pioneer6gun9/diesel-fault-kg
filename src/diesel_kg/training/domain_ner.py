"""Post-hoc exploratory grouped-domain head adaptation; original gold preserved.

The existing 90 inspected domain records are split 60/15/15 by normalized body.
All 5000 original training records remain template training data. The encoder is
frozen at the completed corrected checkpoint, and its CPU features are cached.
Template/domain losses receive equal weights declared before training. Selection
uses only mean(template-dev strict F1, domain-dev strict F1). The 15 held-out
domain records are evaluated once after selection, alongside the same baseline.
This is not new independent industrial validation: all 90 occur in the corpus.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from diesel_kg.evaluation.ner_domain import decorate_bio, dump_json, dump_jsonl, key, metric, normalized_body, read_bio, sha256, spans


def grouped_split(rows, seed=42):
    groups = defaultdict(list)
    for r in rows:
        groups[normalized_body(r['text'])].append(r['sample_id'])
    bodies = sorted(groups)
    random.Random(seed).shuffle(bodies)
    n = len(bodies)
    n_train, n_dev = int(n*2/3), int(n/6)
    partitions = {'domain_train': bodies[:n_train], 'domain_dev': bodies[n_train:n_train+n_dev],
                  'domain_test': bodies[n_train+n_dev:]}
    result = {s: sorted(i for b in bs for i in groups[b]) for s, bs in partitions.items()}
    sets = [set(bs) for bs in partitions.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise AssertionError('Normalized body leak across domain partitions')
    return result


def protocol(data, checkpoint, tokenizer_dir, cases_path, output, seed):
    all_rows = {s: decorate_bio(read_bio(data/(s+'.txt'))) for s in ('train', 'dev', 'test')}
    domain = all_rows['test'][800:]
    if len(domain) != 90:
        raise ValueError('Exploratory protocol expects the existing final 90 domain records')
    partitions = grouped_split(domain, seed)
    by_id = {r['sample_id']: r for r in domain}
    cases = [json.loads(l) for l in cases_path.read_text().splitlines() if l.strip()]
    case_bodies = defaultdict(list)
    for case in cases:
        case_bodies[normalized_body(case['text'])].append(case['case_id'])
    manifest = {name: [{'sample_id': i, 'raw_text_sha256': hashlib.sha256(by_id[i]['text'].encode()).hexdigest(),
                        'normalized_body_sha256': hashlib.sha256(normalized_body(by_id[i]['text']).encode()).hexdigest(),
                        'matching_corpus_case_ids': case_bodies.get(normalized_body(by_id[i]['text']), [])}
                       for i in ids] for name, ids in partitions.items()}
    output.mkdir(parents=True, exist_ok=True)
    p = {'status': 'fixed_before_training', 'experiment': 'post_hoc_exploratory_existing_domain_labels',
         'evidence_boundary': 'The 90 original test records were previously inspected and all overlap the 1522-case corpus. This fixed internal resplit is exploratory, not newly independent industrial validation.',
         'seed': seed, 'split_rule': 'Shuffle sorted NFKC/casefold/punctuation-and-whitespace-removed body groups with Python random.Random(seed); first 2/3 train, next1/6 dev, remainder test.',
         'partition_samples': {s: len(ids) for s, ids in partitions.items()}, 'partitions': manifest,
         'base_training_samples': len(all_rows['train']), 'template_dev_samples': len(all_rows['dev']),
         'encoder': 'Entire corrected completed encoder frozen; unmodified raw-text features cached on CPU.',
         'classifier': 'Start from the completed corrected model classifier, train classifier only with dropout0.2.',
         'loss': '0.5 * mean cross-entropy of a template batch + 0.5 * mean cross-entropy of a domain batch; padding/special tokens excluded.',
         'batch_template_samples': 32, 'batch_domain_samples': 16,
         'optimizer': {'name': 'AdamW', 'lr': 0.0003, 'weight_decay': 0.01},
         'max_epochs': 15, 'patience': 5,
         'selection': 'First maximum of equally weighted template-dev and domain-dev strict raw span-and-type F1; domain-test never used.',
         'gold': 'Original BIO text, labels and half-open offsets preserved; first-character label alignment to wordpiece follows source convention.',
         'decoder': 'Original strict contiguous BIO span decoder; no test-selected boundary repair.',
         'test_evaluation': 'One final selected-checkpoint evaluation of domain-test15; unchanged baseline evaluated on precisely the same records afterwards.',
         'input_sha256': {str(p): sha256(p) for p in [*(data/(s+'.txt') for s in ('train','dev','test')), checkpoint, cases_path]},
         'tokenizer_files_sha256': {p.name: sha256(p) for p in sorted(tokenizer_dir.iterdir()) if p.suffix in ('.json','.txt')},
         'script_sha256': sha256(__file__)}
    dump_json(output/'fixed_protocol.json', p)
    dump_json(output/'domain_partition_ids.json', partitions)
    graph_split = {s: sorted({c for r in items for c in r['matching_corpus_case_ids']}) for s, items in manifest.items()}
    graph_split['scope'] = 'Force these normalized-body groups into matching graph train/dev/test; template and domain data remain unchanged. All matching duplicate numbered cases travel together.'
    dump_json(output/'domain_graph_exclusion_case_ids.json', graph_split)
    selections = {'template_train': all_rows['train'], 'template_dev': all_rows['dev'],
                  **{s: [by_id[i] for i in ids] for s, ids in partitions.items()}}
    for s, selected in selections.items():
        dump_jsonl(output/(s+'_frozen_rows.jsonl'), selected)
    return p, selections


def state_fingerprint(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def cache_features(rows, name, model, tokenizer, label2id, output, cache_manifest):
    import torch
    cache_dir = output/'feature_cache'/name
    cache_dir.mkdir(parents=True, exist_ok=True)
    started, all_cached, shards = time.monotonic(), [], []
    for first in range(0, len(rows), 8):
        batch = rows[first:first+8]
        encoded = tokenizer([r['text'] for r in batch], padding=True, truncation=True,
                            max_length=512, return_offsets_mapping=True, return_tensors='pt')
        mapping = encoded.pop('offset_mapping').tolist()
        with torch.inference_mode():
            hidden = model.roberta(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask']).last_hidden_state
        chunk = []
        for b, (row, offsets) in enumerate(zip(batch, mapping)):
            char_labels = ['O'] * len(row['text'])
            for label, (a,z) in zip(row['labels'], row['offsets']):
                char_labels[a:z] = [label] * (z-a)
            valid = [i for i, (a,z) in enumerate(offsets) if z>a]
            kept_offsets = [offsets[i] for i in valid]
            if kept_offsets and kept_offsets[-1][1] < len(row['text'].rstrip()):
                raise ValueError('Truncated raw BIO record; refusing silently incomplete features')
            item = {'sample_id': row['sample_id'], 'text': row['text'], 'gold_entities': row['gold_entities'],
                    'features': hidden[b, valid].clone().contiguous(),
                    'labels': torch.tensor([label2id[char_labels[offsets[i][0]]] for i in valid]),
                    'offsets': kept_offsets}
            if not bool(torch.isfinite(item['features']).all()):
                raise RuntimeError('Nonfinite frozen encoder features')
            chunk.append(item)
        shard = cache_dir/f'batch_{first:05d}.pt'
        torch.save(chunk, shard)
        shards.append({'path': str(shard), 'sha256': sha256(shard), 'samples': len(chunk), 'bytes': shard.stat().st_size})
        all_cached.extend(chunk)
        if first % 160 == 0:
            print(f'CPU feature cache {name}: {min(first+len(batch),len(rows))}/{len(rows)}; elapsed={time.monotonic()-started:.1f}s', flush=True)
    cache_manifest[name] = {'samples': len(rows), 'elapsed_seconds': time.monotonic()-started,
                            'input_frozen_rows_sha256': sha256(output/(name+'_frozen_rows.jsonl')),
                            'feature_dtype': 'float32', 'hidden_size': model.roberta.config.hidden_size,
                            'token_features': sum(len(x['features']) for x in all_cached), 'shards': shards}
    dump_json(output/'feature_cache_manifest.json', cache_manifest)
    return all_cached


def evaluate_head(head, cache, id2label):
    import torch
    head.eval()
    predictions = []
    with torch.inference_mode():
        for r in cache:
            values = head(r['features'])
            if not bool(torch.isfinite(values).all()):
                raise RuntimeError('Nonfinite classifier logits')
            labels = [id2label[i] for i in values.argmax(-1).tolist()]
            predictions.append({'sample_id': r['sample_id'], 'text': r['text'],
                                'gold_entities': r['gold_entities'], 'predicted_entities': spans(labels, r['offsets']),
                                'predicted_token_labels': labels, 'token_offsets': r['offsets']})
    return metric(predictions), predictions


def features_and_labels(batch):
    import torch
    return torch.cat([r['features'] for r in batch]), torch.cat([r['labels'] for r in batch])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    fixed, rows = protocol(args.data, args.checkpoint, args.tokenizer, args.cases, args.output, args.seed)
    print(json.dumps({'status': 'protocol_fixed', 'partition_samples': fixed['partition_samples']}, ensure_ascii=False), flush=True)
    if args.prepare_only:
        return
    import torch
    from torch import nn
    from diesel_kg.inference.ner import load_ner
    torch.set_num_threads(4)
    random.seed(args.seed); torch.manual_seed(args.seed)
    model, tokenizer, id2label, variant = load_ner(args.checkpoint, args.tokenizer, torch.device('cpu'))
    if variant != 'bert_identity_repair_rebuilt':
        raise ValueError('This control requires the verified corrected completed checkpoint')
    model.eval()
    for p in model.roberta.parameters():
        p.requires_grad_(False)
    frozen_before = state_fingerprint(model.roberta.state_dict())
    label2id = {label: i for i, label in id2label.items()}
    baseline_head = copy.deepcopy(model.classifier).eval()
    head = nn.Sequential(nn.Dropout(.2), copy.deepcopy(model.classifier))
    optimizer = torch.optim.AdamW(head.parameters(), lr=.0003, weight_decay=.01)
    caches, cache_manifest = {}, {}
    for name in ('template_train', 'domain_train', 'template_dev', 'domain_dev'):
        caches[name] = cache_features(rows[name], name, model, tokenizer, label2id, args.output, cache_manifest)
    baseline_dev = {name: evaluate_head(baseline_head, caches[name], id2label)[0] for name in ('template_dev', 'domain_dev')}
    dump_json(args.output/'baseline_dev_metrics.json', baseline_dev)
    selection_score = lambda metrics: .5*(metrics['template_dev']['f1']+metrics['domain_dev']['f1'])
    best_score, best_epoch, patience = selection_score(baseline_dev), 0, 0
    selected_state = copy.deepcopy(head.state_dict())
    history = [{'epoch': 0, 'status': 'unchanged_baseline', 'dev': baseline_dev, 'selection_score': best_score}]
    dump_json(args.output/'epoch_history.json', history)
    rng = random.Random(args.seed)
    for epoch in range(1, 16):
        head.train()
        template = list(caches['template_train']); domain = list(caches['domain_train'])
        rng.shuffle(template); rng.shuffle(domain)
        template_losses, domain_losses = [], []
        for step, first in enumerate(range(0, len(template), 32)):
            template_batch = template[first:first+32]
            domain_batch = [domain[(step*16+j) % len(domain)] for j in range(16)]
            tx, ty = features_and_labels(template_batch); dx, dy = features_and_labels(domain_batch)
            optimizer.zero_grad()
            template_loss = nn.functional.cross_entropy(head(tx), ty)
            domain_loss = nn.functional.cross_entropy(head(dx), dy)
            loss = .5*template_loss + .5*domain_loss
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite classifier training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); optimizer.step()
            template_losses.append(float(template_loss.detach())); domain_losses.append(float(domain_loss.detach()))
        dev = {name: evaluate_head(head, caches[name], id2label)[0] for name in ('template_dev', 'domain_dev')}
        score = selection_score(dev)
        epoch_data = {'epoch': epoch, 'template_loss': sum(template_losses)/len(template_losses),
                      'domain_loss': sum(domain_losses)/len(domain_losses), 'dev': dev, 'selection_score': score}
        history.append(epoch_data)
        if score > best_score:
            best_score, best_epoch, patience = score, epoch, 0
            selected_state = copy.deepcopy(head.state_dict())
            torch.save({'state_dict': selected_state, 'best_epoch': best_epoch, 'selection_score': score,
                        'protocol_sha256': sha256(args.output/'fixed_protocol.json')}, args.output/'best_classifier_head.pt')
            for name in ('template_dev','domain_dev'):
                dump_jsonl(args.output/(name+'_best_predictions.jsonl'), evaluate_head(head, caches[name], id2label)[1])
        else:
            patience += 1
        dump_json(args.output/'epoch_history.json', history)
        print(json.dumps({'epoch': epoch, 'template_dev_f1': dev['template_dev']['f1'],
                          'domain_dev_f1': dev['domain_dev']['f1'], 'selection_score': score,
                          'best_epoch': best_epoch, 'patience': patience}, ensure_ascii=False), flush=True)
        if patience >= 5:
            break
    head.load_state_dict(selected_state, strict=True); head.eval()
    if best_epoch == 0:
        torch.save({'state_dict': selected_state, 'best_epoch': 0, 'selection_score': best_score,
                    'protocol_sha256': sha256(args.output/'fixed_protocol.json')}, args.output/'best_classifier_head.pt')
    frozen_after = state_fingerprint(model.roberta.state_dict())
    if frozen_before != frozen_after:
        raise RuntimeError('Encoder mutated despite frozen-head protocol')
    selected_checkpoint = {'variant': 'bert_identity_repair_domain_head_exploratory', 'label2id': label2id,
                           'model_state_dict': model.state_dict(),
                           'config': {'encoder_class': 'BertModel', 'encoder_config': model.roberta.config.to_dict()},
                           'domain_protocol_sha256': sha256(args.output/'fixed_protocol.json'), 'best_epoch': best_epoch}
    selected_checkpoint['model_state_dict']['classifier.weight'] = head[1].weight.detach().clone()
    selected_checkpoint['model_state_dict']['classifier.bias'] = head[1].bias.detach().clone()
    torch.save(selected_checkpoint, args.output/'domain_adapted_ner_checkpoint.pt')
    # First and only evaluation of the fixed domain-test partition in this run.
    test_cache = cache_features(rows['domain_test'], 'domain_test', model, tokenizer, label2id, args.output, cache_manifest)
    test_metrics, test_predictions = evaluate_head(head, test_cache, id2label)
    baseline_test, baseline_predictions = evaluate_head(baseline_head, test_cache, id2label)
    dump_jsonl(args.output/'domain_test_selected_predictions.jsonl', test_predictions)
    dump_jsonl(args.output/'domain_test_baseline_predictions.jsonl', baseline_predictions)
    summary = {'status': 'completed', 'experiment': fixed['experiment'], 'evidence_boundary': fixed['evidence_boundary'],
               'protocol_path': str(args.output/'fixed_protocol.json'), 'protocol_sha256': sha256(args.output/'fixed_protocol.json'),
               'execution': {'argv': sys.argv, 'python': sys.executable, 'script_sha256': sha256(__file__),
                             'device': 'cpu', 'torch_version': str(torch.__version__), 'elapsed_seconds': time.monotonic()-started},
               'best_epoch': best_epoch, 'epochs_run': len(history)-1, 'best_dev_selection_score': best_score,
               'baseline_dev': baseline_dev, 'selected_dev': {name: evaluate_head(head,caches[name],id2label)[0] for name in ('template_dev','domain_dev')},
               'domain_test_baseline': baseline_test, 'domain_test_selected': test_metrics,
               'frozen_encoder_fingerprint_before': frozen_before, 'frozen_encoder_fingerprint_after': frozen_after,
               'encoder_unchanged': frozen_before == frozen_after, 'gold_changed': False,
               'domain_test_evaluations_per_model': 1, 'domain_test_used_for_selection': False,
               'partition_samples': fixed['partition_samples'],
               'base_checkpoint_sha256': sha256(args.checkpoint),
               'classifier_head_sha256': sha256(args.output/'best_classifier_head.pt'),
               'selected_checkpoint_sha256': sha256(args.output/'domain_adapted_ner_checkpoint.pt'),
               'feature_cache_manifest_sha256': sha256(args.output/'feature_cache_manifest.json'),
               'model_scope': 'Same frozen encoder, adapt only classifier to supplied domain-label conventions. Improvement, if any, supports this small internal resplit only.'}
    dump_json(args.output/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
