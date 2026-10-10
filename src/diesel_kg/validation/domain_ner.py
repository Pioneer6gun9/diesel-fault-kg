"""Independently verify fixed splits, frozen weights and saved NER predictions.

No model forward, retraining, checkpoint selection, or new test evaluation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from diesel_kg.evaluation.ner_domain import key, metric, normalized_body, sha256, spans
from diesel_kg.paths import resolve_recorded_path


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adaptation', type=Path, required=True)
    parser.add_argument('--primary-predictions', type=Path, required=True)
    args = parser.parse_args()
    root = args.adaptation
    summary = json.loads((root/'summary.json').read_text())
    protocol = json.loads((root/'fixed_protocol.json').read_text())
    checks = []

    def check(name, passed, details=None):
        checks.append({'name': name, 'passed': bool(passed), 'details': details})

    check('completed_actual_training', summary['status'] == 'completed')
    check('protocol_sha256', summary['protocol_sha256'] == sha256(root/'fixed_protocol.json'))
    check('protocol_inputs_unchanged', all(sha256(resolve_recorded_path(p)) == digest for p,digest in protocol['input_sha256'].items()))
    groups = {s: load_rows(root/(s+'_frozen_rows.jsonl')) for s in ('domain_train','domain_dev','domain_test')}
    sets = {s: {normalized_body(r['text']) for r in rows} for s,rows in groups.items()}
    check('domain_groups_do_not_overlap', not (sets['domain_train'] & sets['domain_dev'] or sets['domain_train'] & sets['domain_test'] or sets['domain_dev'] & sets['domain_test']))
    check('fixed_60_15_15_samples', {s:len(rows) for s,rows in groups.items()} == {'domain_train':60,'domain_dev':15,'domain_test':15})
    check('all_90_original_sample_ids_preserved', {r['sample_id'] for rows in groups.values() for r in rows} == set(range(800,890)))
    history = json.loads((root/'epoch_history.json').read_text())
    first_best = max(history, key=lambda x:x['selection_score'])
    check('best_is_first_maximum_dev_score', summary['best_epoch'] == first_best['epoch'] and summary['best_dev_selection_score'] == first_best['selection_score'])
    check('selection_score_equal_dev_weights', all(abs(r['selection_score'] - .5*(r['dev']['template_dev']['f1']+r['dev']['domain_dev']['f1'])) < 1e-12 for r in history))
    check('test_unused_for_selection', all('test' not in r and not any('test' in k for k in r['dev']) for r in history) and not summary['domain_test_used_for_selection'])
    cache = json.loads((root/'feature_cache_manifest.json').read_text())
    check('cache_manifest_sha256', sha256(root/'feature_cache_manifest.json') == summary['feature_cache_manifest_sha256'])
    check('feature_shard_inventory_sha256', all(sha256(resolve_recorded_path(item['path'])) == item['sha256'] for group in cache.values() for item in group['shards']),
          {'shards': sum(len(group['shards']) for group in cache.values()), 'bytes': sum(item['bytes'] for group in cache.values() for item in group['shards'])})
    check('frozen_cache_rows_sha256', all(sha256(root/(name+'_frozen_rows.jsonl')) == group['input_frozen_rows_sha256'] for name,group in cache.items()))
    check('base_5000_and_dev800_retained', len(load_rows(root/'template_train_frozen_rows.jsonl')) == 5000 and len(load_rows(root/'template_dev_frozen_rows.jsonl')) == 800)
    gold_by_id = {r['sample_id']: r for r in groups['domain_test']}
    final_predictions = {}
    for model in ('baseline','selected'):
        predictions = load_rows(root/('domain_test_'+model+'_predictions.jsonl'))
        final_predictions[model] = predictions
        check(model+'_same_fixed_test_ids', {r['sample_id'] for r in predictions} == set(gold_by_id))
        check(model+'_raw_gold_preserved', all(r['text'] == gold_by_id[r['sample_id']]['text'] and {key(e) for e in r['gold_entities']} == {key(e) for e in gold_by_id[r['sample_id']]['gold_entities']} for r in predictions))
        check(model+'_strict_decoder_preserved', all({key(e) for e in spans(r['predicted_token_labels'],r['token_offsets'])} == {key(e) for e in r['predicted_entities']} for r in predictions))
        check(model+'_metrics_independent_saved_prediction_recount', metric(predictions) == summary['domain_test_'+model])
    prior = {r['sample_id']: r for r in load_rows(args.primary_predictions)}
    prior_exact = all({key(e) for e in r['predicted_entities']} == {key(e) for e in prior[r['sample_id']]['predicted_entities']} for r in final_predictions['baseline'])
    check('baseline_matches_original_completed_checkpoint_saved_spans', prior_exact)
    head_path = resolve_recorded_path(root/'best_classifier_head.pt')
    checkpoint = resolve_recorded_path(root/'domain_adapted_ner_checkpoint.pt')
    check('classifier_head_sha256', summary['classifier_head_sha256'] == sha256(head_path))
    check('selected_checkpoint_sha256', summary['selected_checkpoint_sha256'] == sha256(checkpoint))
    import torch
    base_path = next(resolve_recorded_path(p) for p in protocol['input_sha256'] if p.endswith('.pt'))
    original = torch.load(base_path, map_location='cpu', weights_only=True, mmap=True)
    selected = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    enc = [k for k in original['model_state_dict'] if k.startswith('roberta.')]
    check('all_encoder_state_tensors_bitwise_unchanged', all(torch.equal(original['model_state_dict'][k],selected['model_state_dict'][k]) for k in enc), {'encoder_tensors': len(enc)})
    check('label_mapping_unchanged', original['label2id'] == selected['label2id'])
    result = {'status': 'passed' if all(c['passed'] for c in checks) else 'failed', 'checks': checks,
              'counts': {'passed': sum(c['passed'] for c in checks), 'total': len(checks)},
              'execution': {'argv':sys.argv,'script_sha256':sha256(__file__)},
              'evidence_boundary': protocol['evidence_boundary'],
              'note': 'Verification re-counted saved predictions and compared state tensors. No new test forward or selection occurred.'}
    (root/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
