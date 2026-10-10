"""Independent positive-only RE audit and template-group-heldout control.

Original sources and previous artifacts are read-only. Groups are exact sentence
skeletons inferred from unique, nonoverlapping annotated mentions, not verified
document provenance. Only original train/dev supplies grouped train/dev/test.
Original225 is an additional previously evaluated benchmark, never new gold.
No negative examples or no_relation labels are generated.
"""
from __future__ import annotations
from diesel_kg.paths import module_path, resolve_recorded_path
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import shutil
import sys
import time
import unicodedata

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizerFast


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def read_rows(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.suffix == '.jsonl' else json.loads(path.read_text())


def normalize(text):
    return ''.join(unicodedata.normalize('NFKC', text).split())


def query_key(row):
    return tuple(normalize(row[name]) for name in ('text', 'subject', 'object'))


def all_occurrences(text, entity):
    if not entity: return []
    starts = []; start = 0
    while True:
        found = text.find(entity, start)
        if found < 0: return starts
        starts.append((found, found + len(entity))); start = found + 1


def strict_pair(row, tokenizer, width=128):
    text = row['text']; offsets = tokenizer(text, truncation=True, max_length=width, return_offsets_mapping=True)['offset_mapping']
    result = {'valid': True, 'reasons': [], 'entities': {}}
    for name in ('subject', 'object'):
        spans = all_occurrences(text, row[name]); info = {'occurrences': spans}
        if len(spans) != 1:
            result['reasons'].append(name + ('_absent' if not spans else '_ambiguous_occurrence'))
        else:
            start, end = spans[0]
            positions = [i for i, (left, right) in enumerate(offsets) if right > left and left < end and right > start]
            exact = bool(positions and offsets[positions[0]][0] == start and offsets[positions[-1]][1] == end)
            info.update({'char_span': [start, end], 'token_span_inclusive': [positions[0], positions[-1]] if positions else None,
                         'complete_exact_token_boundaries': exact})
            if not exact: result['reasons'].append(name + '_truncated_or_partial_token_boundary')
        result['entities'][name] = info
    if all(len(result['entities'][name]['occurrences']) == 1 for name in ('subject', 'object')):
        a, b = (result['entities'][name]['occurrences'][0] for name in ('subject', 'object'))
        if max(a[0], b[0]) < min(a[1], b[1]): result['reasons'].append('overlapping_or_same_mention_pair')
        result['textual_entity_order'] = 'subject_before_object' if a[0] < b[0] else 'object_before_subject'
    result['valid'] = not result['reasons']
    return result


def skeleton(row, pair):
    if not pair['valid']: return None
    text = row['text']
    spans = [(pair['entities'][name]['char_span'][0], pair['entities'][name]['char_span'][1], marker)
             for name, marker in (('subject', '<S>'), ('object', '<O>'))]
    for start, end, marker in sorted(spans, reverse=True): text = text[:start] + marker + text[end:]
    return text


def metric_arrays(truth, predicted, classes):
    matrix = np.bincount(np.asarray(truth, dtype=int) * classes + np.asarray(predicted, dtype=int), minlength=classes*classes).reshape(classes, classes)
    support = matrix.sum(1); pp = matrix.sum(0); tp = matrix.diagonal()
    denom = support + pp
    f1 = np.divide(2*tp, denom, out=np.zeros(classes, dtype=float), where=denom > 0)
    return matrix, float(tp.sum()/matrix.sum()) if matrix.sum() else None, float(f1.mean()) if matrix.sum() else None


def metrics(rows, labels, bootstrap=2000, seed=42):
    if not rows: return {'n': 0, 'accuracy': None, 'macro_f1_all_nine_classes': None, 'reason': 'No examples satisfy this subset definition.'}
    mapping = {name: i for i, name in enumerate(labels)}; nclass = len(labels)
    y = np.array([mapping[row['relation']] for row in rows]); p = np.array([mapping[row['predicted_relation']] for row in rows])
    cm, accuracy, f1 = metric_arrays(y, p, nclass)
    rng = np.random.default_rng(seed); values = []
    for _ in range(bootstrap):
        indices = rng.integers(0, len(rows), len(rows))
        _, a, f = metric_arrays(y[indices], p[indices], nclass); values.append([a, f])
    low, high = np.quantile(values, [0.025, 0.975], axis=0)
    per_class = {}
    for i, name in enumerate(labels):
        tp = int(cm[i, i]); ng = int(cm[i].sum()); npred = int(cm[:, i].sum())
        pr, re = tp/npred if npred else 0., tp/ng if ng else 0.
        per_class[name] = {'gold': ng, 'predicted': npred, 'matched': tp, 'precision': pr, 'recall': re, 'f1': 2*pr*re/(pr+re) if pr+re else 0.}
    result = {'n': len(rows), 'correct': int(cm.diagonal().sum()), 'accuracy': accuracy,
              'macro_f1_all_nine_classes': f1, 'macro_f1_gold_supported_classes': float(np.mean([per_class[x]['f1'] for x in labels if per_class[x]['gold']])),
              'confusion_matrix': cm.tolist(), 'confusion_matrix_order': labels, 'rows_gold_columns_prediction': True, 'per_class': per_class,
              'bootstrap_ci95': {'accuracy': [float(low[0]), float(high[0])], 'macro_f1_all_nine_classes': [float(low[1]), float(high[1])],
                    'resamples': bootstrap, 'seed': seed, 'method': 'Percentile iid sample bootstrap; conditional on supplied annotations, not annotation validity or population representativeness.'}}
    groups = defaultdict(list)
    for i, row in enumerate(rows): groups[row.get('_template_group') or 'raw:' + normalize(row['text'])].append(i)
    keys = sorted(groups)
    if len(keys) > 1:
        values = []; rng = np.random.default_rng(seed + 1)
        for _ in range(bootstrap):
            indices = np.concatenate([groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
            _, a, f = metric_arrays(y[indices], p[indices], nclass); values.append([a, f])
        low, high = np.quantile(values, [0.025, 0.975], axis=0)
        result['template_cluster_bootstrap_ci95'] = {'accuracy': [float(low[0]), float(high[0])], 'macro_f1_all_nine_classes': [float(low[1]), float(high[1])],
            'clusters': len(keys), 'resamples': bootstrap, 'seed': seed+1, 'method': 'Resample inferred template clusters with replacement, retain all member examples; all nine macro classes fixed, including classes absent from a replicate.'}
    return result


def source_classes(path, device='cpu'):
    source = path.read_text(encoding='utf-8-sig'); tree = ast.parse(source)
    wanted = {'RelationModel', 'REDataset', 'compute_class_weights', 'oversample_minority'}
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted]
    assert {x.name for x in selected} == wanted
    namespace = {'torch': torch, 'nn': nn, 'Dataset': Dataset, 'BertModel': BertModel, 'json': json,
                 'logger': logging.getLogger('re_clean'), 'random': random, 'DEVICE': torch.device(device)}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace, {node.name: hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest() for node in selected}


def offset_locator(self, text, entity, offsets):
    spans = all_occurrences(text, entity)
    if len(spans) != 1: raise ValueError('Unique gold mention required in grouped control')
    start, end = spans[0]
    positions = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
    if not positions or offsets[positions[0]][0] != start or offsets[positions[-1]][1] != end: raise ValueError('Exact complete gold mention required')
    return positions[0], positions[-1]


def external_first_occurrence_locator(self, text, entity, offsets):
    """Original locator-repair policy, only for previously evaluated original225."""
    start = text.find(entity)
    if start < 0: raise ValueError('external_entity_not_in_text')
    end = start + len(entity)
    positions = [i for i, (a,b) in enumerate(offsets) if b > a and a < end and b > start]
    if not positions or offsets[positions[0]][0] > start or offsets[positions[-1]][1] < end:
        raise ValueError('external_entity_truncated_or_unaligned')
    return positions[0], positions[-1]


def provenance(args):
    return {'sys_argv': sys.argv, 'sys_executable': sys.executable, 'script_sha256': sha(Path(__file__)),
            'source_sha256': sha(args.source), 'data_sha256': {s: sha(args.data_dir/(s+'.json')) for s in ('train','dev','test')},
            'tokenizer_sha256': {p.name: sha(p) for p in sorted(args.pretrained.iterdir()) if p.suffix in ('.json','.txt')},
            'original_re_training_sha256': sha(module_path('diesel_kg.training.source_re')),
            'scope': 'Independent continued validation. Nine provided positive relations only; no relation-existence gold or no_relation generated.'}


def audit(args):
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    tokenizer = BertTokenizerFast.from_pretrained(str(args.pretrained), local_files_only=True)
    assert not tokenizer.do_lower_case
    source, ast_sha = source_classes(args.source)
    old_dataset = source['REDataset'].__new__(source['REDataset']); old_dataset.tokenizer = tokenizer
    rows = {s: json.loads((args.data_dir/(s+'.json')).read_text()) for s in ('train','dev','test')}
    query_splits = defaultdict(set); text_splits = defaultdict(set); label_sets = defaultdict(set)
    for split, part in rows.items():
        for row in part:
            query_splits[query_key(row)].add(split); text_splits[normalize(row['text'])].add(split); label_sets[query_key(row)].add(row['relation'])
    train_dev_labels = sorted({row['relation'] for s in ('train','dev') for row in rows[s]})
    assert len(train_dev_labels) == 9 and set(row['relation'] for row in rows['test']) <= set(train_dev_labels)
    details = {}; good = {}; prior_queries = {query_key(row) for s in ('train','dev') for row in rows[s]}
    prior_texts = {normalize(row['text']) for s in ('train','dev') for row in rows[s]}
    for split, part in rows.items():
        details[split] = []; good[split] = []; seen = set()
        for index, row in enumerate(part):
            pair = strict_pair(row, tokenizer); key = query_key(row); template = skeleton(row, pair)
            duplicate = key in seen; seen.add(key)
            offsets = tokenizer(row['text'], truncation=True, max_length=128, padding='max_length', return_offsets_mapping=True)['offset_mapping']
            original_positions = {}
            for name in ('subject','object'):
                p, q = old_dataset._locate_entity(row['text'], row[name], offsets)
                original_positions[name] = {'token_span_inclusive': [p,q], 'raw_text_roundtrip': row['text'][offsets[p][0]:offsets[q][1]],
                                            'roundtrip_equals_entity': row['text'][offsets[p][0]:offsets[q][1]] == row[name]}
            item = {'original_split': split, 'original_index': index, **row, 'strict_pair': pair, 'inferred_template': template,
                    'within_split_duplicate_query': duplicate, 'cross_split_query_splits': sorted(query_splits[key]),
                    'cross_split_normalized_text_splits': sorted(text_splits[normalize(row['text'])]),
                    'query_has_conflicting_relation_labels': len(label_sets[key]) > 1, 'original_locator': original_positions}
            reverse = (key[0],key[2],key[1])
            item['reverse_order_same_text_query_present'] = reverse != key and reverse in query_splits
            if reverse in label_sets: item['reverse_order_gold_relations'] = sorted(label_sets[reverse])
            details[split].append(item)
            if pair['valid'] and not duplicate and len(label_sets[key]) == 1: good[split].append(item)
    # Group development uses original train/dev only. No original test row enters
    # these groups or the class mapping. Conflicts are determined in train/dev.
    grouped = {}; dev_label_sets = defaultdict(set)
    for s in ('train','dev'):
        for row in rows[s]: dev_label_sets[query_key(row)].add(row['relation'])
    duplicate_removed = []
    for s in ('train','dev'):
        for item in details[s]:
            key = query_key(item)
            if not item['strict_pair']['valid'] or len(dev_label_sets[key]) != 1: continue
            if key in grouped: duplicate_removed.append({'split':s,'index':item['original_index'],'same_as':grouped[key]['_source_row_id']}); continue
            copied = dict(rows[s][item['original_index']]); copied['_source_row_id'] = f'{s}:{item["original_index"]}'
            copied['_template_group'] = item['inferred_template']; grouped[key] = copied
    by_relation = defaultdict(set)
    for row in grouped.values(): by_relation[row['relation']].add(row['_template_group'])
    assert all(len(by_relation[name]) == 3 for name in train_dev_labels), 'Expected audited three inferred templates per positive class'
    assignments = {}; sets = {'train':[],'dev':[],'test':[]}; group_plan = []
    for relation in train_dev_labels:
        templates = sorted(by_relation[relation], key=lambda t: hashlib.sha256(json.dumps([42,relation,t],ensure_ascii=False).encode()).hexdigest())
        for split, template in zip(('train','dev','test'),templates):
            assert template not in assignments
            assignments[template] = split; group_plan.append({'relation':relation,'template':template,'assigned_split':split,
                'assignment_hash':hashlib.sha256(json.dumps([42,relation,template],ensure_ascii=False).encode()).hexdigest()})
    for row in grouped.values(): sets[assignments[row['_template_group']]].append(row)
    for split, part in sets.items():
        part.sort(key=lambda row: row['_source_row_id'])
        assert set(x['relation'] for x in part) == set(train_dev_labels)
        dump(out/'grouped_data'/f'{split}.json', part)
    group_sets = {s:{r['_template_group'] for r in part} for s,part in sets.items()}
    sentence_sets = {s:{normalize(r['text']) for r in part} for s,part in sets.items()}
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        assert not(group_sets[a]&group_sets[b]) and not(sentence_sets[a]&sentence_sets[b])
    assert sum(map(len,sets.values())) == len(grouped)
    # Seal original225 as read-only additional benchmark. It was evaluated by
    # previous checkpoints; sealing now does not make it a new independent test.
    dump(out/'sealed_original225.json', rows['test'])
    dump(out/'row_audit.json', details)
    all_prior_templates = set().union(*(set(x['inferred_template'] for x in details[s] if x['inferred_template']) for s in ('train','dev')))
    results = {}; references = {}
    for variant in ('original_re','re_locator_fixed'):
        directory = args.legacy_root/variant; predictions_path = directory/'test_predictions.jsonl'
        predictions = read_rows(predictions_path); index_map = {x['original_index']:x for x in predictions}
        assert len(index_map) == len(predictions) == len(rows['test']) == 225
        joined = []
        for item in details['test']:
            pred = index_map[item['original_index']]
            assert all(pred[k] == item[k] for k in ('text','subject','object','relation'))
            joined.append({**pred,'_template_group':item['inferred_template'], '_audit':item})
        clean = [r for r in joined if r['_audit']['strict_pair']['valid'] and not r['_audit']['within_split_duplicate_query']
                 and not r['_audit']['query_has_conflicting_relation_labels'] and query_key(r) not in prior_queries
                 and normalize(r['text']) not in prior_texts]
        subsets = {'all_original225':joined, 'strict_unique_nonoverlap_no_prior_text_or_query_no_test_duplicate':clean,
            'clean_seen_inferred_template':[r for r in clean if r['_template_group'] in all_prior_templates],
            'clean_unseen_inferred_template':[r for r in clean if r['_template_group'] not in all_prior_templates],
            'clean_explicit_subject_object_types':[r for r in clean if r.get('subject_type') and r.get('object_type')]}
        results[variant] = {name:{**metrics(part,train_dev_labels,args.bootstrap), 'original_indices':[r['original_index'] for r in part]} for name,part in subsets.items()}
        checkpoint = args.checkpoint_root/variant/'best_re_model/best_model.pt'
        configuration = json.loads((directory/'run_configuration.json').read_text())
        summary = json.loads((directory/'run_summary.json').read_text())
        assert sha(checkpoint) == summary['best_checkpoint_sha256']
        assert configuration['data']['test']['sha256'] == sha(args.data_dir/'test.json')
        references[variant] = {'predictions_path':str(predictions_path),'predictions_sha256':sha(predictions_path),
            'checkpoint_path':str(checkpoint),'checkpoint_sha256':sha(checkpoint), 'configuration_sha256':sha(directory/'run_configuration.json'),
            'summary_sha256':sha(directory/'run_summary.json'), 'source_sha256':configuration['source_sha256']}
    report = {'execution':provenance(args), 'original_classes_ast_sha256':ast_sha, 'labels':train_dev_labels,
        'subset_definition':'Strict complete token-aligned unique disjoint gold mentions; remove normalized query conflicts, duplicate test queries, and any normalized text/query present in original train/dev. No schema/type-rule flag is treated as proved label error.',
        'original_splits':{s:{'n':len(part),'strict_pair_valid':sum(x['strict_pair']['valid'] for x in details[s]),
            'duplicate_query_extra_rows':sum(x['within_split_duplicate_query'] for x in details[s]),
            'query_conflict_rows':sum(x['query_has_conflicting_relation_labels'] for x in details[s]),
            'cross_split_query_rows':sum(len(x['cross_split_query_splits'])>1 for x in details[s]),
            'cross_split_text_rows':sum(len(x['cross_split_normalized_text_splits'])>1 for x in details[s]),
            'original_locator_both_entities_roundtrip_correct':sum(all(v['roundtrip_equals_entity'] for v in x['original_locator'].values()) for x in details[s]),
            'textual_entity_order':dict(Counter(x['strict_pair'].get('textual_entity_order','ambiguous') for x in details[s])),
            'reverse_order_same_text_query_rows':sum(x['reverse_order_same_text_query_present'] for x in details[s])} for s,part in rows.items()},
        'legacy_checkpoint_references':references, 'existing_prediction_metrics':results,
        'limits':['Existing predictions reused; no new forward for this audit.','Gold mention relation classification only; all labels positive.','Template groups are inferred weak provenance; generator/document source IDs absent.','Original225 has already been evaluated historically and is not newly independent gold.','Explicit type/direction metadata is not independent annotation validation.']}
    # Direction audit binds the existing declared ontology only as flags.
    schema_path=args.legacy_root/'re_locator_fixed/data_schema_audit.json'
    schema=json.loads(schema_path.read_text())
    report['declared_type_direction_audit']={'schema_sha256':sha(schema_path), 'rules':schema['rules'], 'splits':schema['splits'],
        'interpretation':'Declared-rule incompatibility is a review flag, never an automatic label correction or clean-subset exclusion.'}
    dump(out/'clean_evaluation.json',report)
    plan={'status':'frozen','seed':42,'group_key':'Exact masked <S>/<O> sentence from strictly unique gold mentions; inferred, not verified original source provenance.',
        'assignment_rule':'Per nine positive relations sort three templates by SHA256(JSON[42,relation,template]); first train, second dev, third final heldout.',
        'class_mapping_from':'Original train/dev only','original225_role':'Additional previously evaluated benchmark sealed; no split/fit/model selection influence.',
        'group_assignments':group_plan,'duplicate_development_rows_removed':duplicate_removed,
        'excluded_ambiguous_or_invalid_development_rows':[{'split':s,'index':x['original_index'],'reasons':x['strict_pair']['reasons']} for s in ('train','dev') for x in details[s] if not x['strict_pair']['valid']],
        'splits':{s:{'n':len(p),'groups':len(group_sets[s]),'relations':dict(Counter(x['relation'] for x in p)), 'sha256':sha(out/'grouped_data'/f'{s}.json')} for s,p in sets.items()},
        'original225_sha256':sha(out/'sealed_original225.json'),'original225_source_sha256':sha(args.data_dir/'test.json'),
        'checks':{'group_overlap':0,'normalized_text_overlap':0,'query_overlap':0,'nine_class_coverage_all_splits':True,'unique_development_queries_conserved':len(grouped), 'original225_rows_in_grouped_data':0}}
    dump(out/'split_manifest.json',plan)
    print(json.dumps({'original_splits':report['original_splits'],'legacy_clean':{k:v['strict_unique_nonoverlap_no_prior_text_or_query_no_test_duplicate'] for k,v in results.items()}, 'grouped_splits':plan['splits']},ensure_ascii=False),flush=True)
    return plan


def evaluate_model(model, loader, labels, device):
    model.eval(); result=[]
    with torch.inference_mode():
        for batch in loader:
            logits=model(*(batch[name].to(device) for name in ('input_ids','attention_mask','sub_pos','obj_pos')))
            assert bool(torch.isfinite(logits).all()),'Nonfinite RE evaluation logits'
            predicted=logits.argmax(-1).cpu().tolist(); gold=batch['rel_id'].tolist()
            for p,g in zip(predicted,gold):result.append({'relation':labels[g],'predicted_relation':labels[p]})
    return result


def reload_check(model, cls, pretrained, dataset, device):
    model.eval(); batch={name:torch.stack([dataset[i][name] for i in range(min(8,len(dataset)))]) for name in dataset[0]}
    with torch.inference_mode(): live=model(*(batch[n].to(device) for n in ('input_ids','attention_mask','sub_pos','obj_pos'))).cpu()
    with torch.random.fork_rng(devices=[]): copy=cls(str(pretrained),model.classifier[-1].out_features,0.2)
    copy.load_state_dict({n:t.detach().cpu().clone() for n,t in model.state_dict().items()},strict=True);copy.eval()
    with torch.inference_mode():ref=copy(*(batch[n] for n in ('input_ids','attention_mask','sub_pos','obj_pos')))
    error=float((live-ref).abs().max());finite=bool(torch.isfinite(live).all() and torch.isfinite(ref).all());same=bool(torch.equal(live.argmax(-1),ref.argmax(-1)))
    result={'samples':len(live),'device':str(device),'cpu_max_logit_abs_error':error,'argmax_identical':same,'finite':finite,'strict_state_load':True,'tolerance':1e-4}
    assert finite and same and error<1e-4, result
    return result


def train(args):
    out=args.output;run=out/'grouped_training';run.mkdir(exist_ok=True)
    execution_path=run/'execution.json'
    if execution_path.exists(): raise RuntimeError('Do not overwrite recorded training; use a new output directory.')
    record={'status':'running','launch_observed_unix':time.time(),'execution':provenance(args),'device':args.device,'test_evaluations':[]}
    dump(execution_path,record);start=time.perf_counter()
    try:
        manifest=json.loads((out/'split_manifest.json').read_text())
        for split in ('train','dev','test'):assert sha(out/'grouped_data'/f'{split}.json')==manifest['splits'][split]['sha256']
        assert sha(out/'sealed_original225.json')==manifest['original225_sha256']
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        if args.device=='mps':torch.mps.manual_seed(42)
        torch.set_num_threads(4);device=torch.device(args.device)
        namespace,ast_sha=source_classes(args.source,device)
        cls=namespace['REDataset'];cls._locate_entity=offset_locator
        tokenizer=BertTokenizerFast.from_pretrained(str(args.pretrained),local_files_only=True)
        development={s:read_rows(out/'grouped_data'/f'{s}.json') for s in ('train','dev')}
        labels=sorted({r['relation'] for r in development['train']});assert len(labels)==9
        rel2id={label:i for i,label in enumerate(labels)}
        datasets={s:cls(str(out/'grouped_data'/f'{s}.json'),tokenizer,rel2id,128) for s in ('train','dev')}
        original_train_count=len(datasets['train']);datasets['train'].data=namespace['oversample_minority'](datasets['train'].data,rel2id)
        loaders={s:DataLoader(d,batch_size=32 if s=='train' else 64,shuffle=s=='train',num_workers=0) for s,d in datasets.items()}
        model_class=namespace['RelationModel'];model=model_class(str(args.pretrained),9,0.2)
        assert all(p.is_contiguous() for p in model.bert.parameters()),'Unexpected noncontiguous encoder; stop before MPS'
        model.to(device)
        configuration={'execution':provenance(args),'class_ast_sha256':ast_sha,'split_manifest_sha256':sha(out/'split_manifest.json'),'labels':labels,
            'train_before_oversampling':original_train_count,'train_after_oversampling':len(datasets['train']),
            'protocol':{'batch_size':32,'evaluation_batch_size':64,'max_seq_len':128,'learning_rate':1e-5,'max_epochs':15,'early_stop_patience':3,'dropout':0.2,'weight_decay':1e-4,'clip_grad_norm':1.,
                'optimizer':'AdamW eps1e-8','scheduler':'CosineAnnealingWarmRestarts T_0=3 T_mult=2 eta_min=1e-7 stepped every minibatch; original WARMUP_RATIO unused','selection_metric':'dev accuracy == single-label micro F1'},
            'source_settings_preserved':['Original RelationModel AST, entity interaction/MLP/inclusive pooling','Original last12 parameter tensors trainable','Original class-weight/oversampling functions','Original optimizer/scheduler/gradient clipping/15 upper limit/patience3'],
            'differences':['Inferred-template groups separated and development queries deduplicated','Strict unique full token-aligned offset locator; no ambiguous gold occurrence guessed','Class vocabulary from train only; test loaded only after checkpoint frozen','Offline official assets and added seed42/MPS device','Finite logits + CPU strict reload control each epoch before selection'],
            'pretrained_sha256':{p.name:sha(p) for p in sorted(args.pretrained.iterdir()) if p.is_file()},
            'trainable_parameter_names':[n for n,p in model.named_parameters() if p.requires_grad]}
        dump(run/'config.json',configuration);dump(run/'initial_reload_check.json',reload_check(model,model_class,args.pretrained,datasets['dev'],device))
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4,eps=1e-8)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=3,T_mult=2,eta_min=1e-7)
        criterion=nn.CrossEntropyLoss(weight=namespace['compute_class_weights'](datasets['train'].data,rel2id))
        best=-1.;best_epoch=0;patience=0;history=[];checkpoint=run/'best_model.pt'
        for epoch in range(1,16):
            began=time.perf_counter();model.train();losses=[]
            for step,batch in enumerate(loaders['train'],1):
                logits=model(*(batch[n].to(device) for n in ('input_ids','attention_mask','sub_pos','obj_pos')))
                assert bool(torch.isfinite(logits).all()),'Nonfinite train logits'
                loss=criterion(logits,batch['rel_id'].to(device));assert bool(torch.isfinite(loss))
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step();optimizer.zero_grad();losses.append(float(loss.detach()))
                if step%5==0:print(f'RE_GROUP epoch={epoch} step={step}/{len(loaders["train"])} meanloss={np.mean(losses):.6f} elapsed={time.perf_counter()-began:.1f}s',flush=True)
            dev_predictions=evaluate_model(model,loaders['dev'],labels,device)
            acc=sum(x['relation']==x['predicted_relation'] for x in dev_predictions)/len(dev_predictions)
            check=reload_check(model,model_class,args.pretrained,datasets['dev'],device)
            improved=acc>best
            if improved:
                best=acc;best_epoch=epoch;patience=0
                torch.save({'epoch':epoch-1,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'rel2id':rel2id,'best_dev_accuracy':best,'config':configuration},checkpoint)
            else:patience+=1
            item={'epoch':epoch,'mean_train_loss':float(np.mean(losses)),'dev_accuracy_micro_f1':acc,'best_by_dev':improved,'early_stop_counter':patience,'elapsed_seconds':time.perf_counter()-began,'reload_check':check}
            history.append(item);dump(run/'epoch_history.json',history);print(json.dumps(item),flush=True)
            if patience>=3:break
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True);model.load_state_dict(saved['model_state_dict'],strict=True);del saved
        checkpoint_sha=sha(checkpoint);record['frozen_checkpoint_before_test_sha256']=checkpoint_sha;dump(execution_path,record)
        # Heldout labels/data first loaded here, after the checkpoint is frozen.
        final={}
        for name,path in [('group_heldout',out/'grouped_data/test.json'),('original225_previously_evaluated',out/'sealed_original225.json')]:
            source_rows=read_rows(path)
            if name.startswith('original225'):
                # Full external benchmark retains the documented first-mention
                # locator repair. Its strict/clean submetrics use the same
                # single forward and do not pretend ambiguous mentions are gold.
                evaluation_class=type('ExternalFirstOccurrenceDataset',(cls,),{'_locate_entity':external_first_occurrence_locator})
            else:evaluation_class=cls
            kept=list(enumerate(source_rows));data_path=path
            dataset=evaluation_class(str(data_path),tokenizer,rel2id,128);loader=DataLoader(dataset,batch_size=64,shuffle=False,num_workers=0)
            predictions=evaluate_model(model,loader,labels,device)
            for pred,(index,row) in zip(predictions,kept):pred.update({**row,'original_index':index})
            if name.startswith('original225'):
                row_audit=json.loads((out/'row_audit.json').read_text())['test']
                for row in predictions:row['_template_group']=row_audit[row['original_index']]['inferred_template']
            prediction_path=run/(name+'_predictions.jsonl')
            with prediction_path.open('w') as stream:
                for row in predictions:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            final[name]=metrics(predictions,labels,args.bootstrap)
            final[name].update({'source_rows':len(source_rows),'strict_scored_rows':len(predictions),'excluded_ambiguous_or_invalid_indices':[i for i in range(len(source_rows)) if i not in {j for j,_ in kept}], 'predictions_sha256':sha(prediction_path)})
            if name.startswith('original225'):
                final[name]['strict_scored_rows']=sum(x['strict_pair']['valid'] for x in row_audit)
                final[name]['ambiguous_occurrence_policy']='First complete occurrence as previous re_locator_fixed; full225 supplemental only. Strict/clean subsets exclude unlocated or ambiguous annotations.'
                clean_evaluation=json.loads((out/'clean_evaluation.json').read_text())['existing_prediction_metrics']['re_locator_fixed']
                final[name]['subsets_from_same_single_forward']={}
                for sub in ('strict_unique_nonoverlap_no_prior_text_or_query_no_test_duplicate','clean_seen_inferred_template','clean_unseen_inferred_template'):
                    allowed=set(clean_evaluation[sub]['original_indices'])
                    final[name]['subsets_from_same_single_forward'][sub]=metrics([p for p in predictions if p['original_index'] in allowed],labels,args.bootstrap)
            record['test_evaluations'].append({'name':name,'count':1,'checkpoint_sha256':checkpoint_sha,'prediction_path':str(prediction_path),'predictions_sha256':sha(prediction_path)});dump(execution_path,record)
        dump(run/'final_cpu_reload_check.json',reload_check(model,model_class,args.pretrained,datasets['dev'],device))
        assert sha(checkpoint)==checkpoint_sha
        summary={'status':'completed','valid_for_model_selection':True,'best_epoch':best_epoch,'epochs_run':len(history),'best_dev_accuracy_micro_f1':best,'checkpoint_path':str(checkpoint),'checkpoint_sha256':checkpoint_sha,'test':final,'elapsed_seconds':time.perf_counter()-start,
            'qualification':'Positive-only gold-mention forced-choice classification. Inferred-template heldout only; external original225 already evaluated historically, not independent industrial annotation.'}
        dump(run/'summary.json',summary)
        record.update({'status':'completed','exit_code':0,'completion_observed_unix':time.time(),'checkpoint_sha256':checkpoint_sha,'elapsed_seconds':time.perf_counter()-start});dump(execution_path,record)
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    except BaseException as error:
        record.update({'status':'failed','exit_code':1,'completion_observed_unix':time.time(),'exception':repr(error)});dump(execution_path,record);raise


def verify(args):
    out=args.output;manifest=json.loads((out/'split_manifest.json').read_text());parts={s:read_rows(out/'grouped_data'/f'{s}.json') for s in ('train','dev','test')}
    for s,rows in parts.items():
        assert len(rows)==manifest['splits'][s]['n'] and sha(out/'grouped_data'/f'{s}.json')==manifest['splits'][s]['sha256']
        assert len({query_key(r) for r in rows})==len(rows) and len({r['relation'] for r in rows})==9
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        assert not({r['_template_group'] for r in parts[a]}&{r['_template_group'] for r in parts[b]})
        assert not({query_key(r) for r in parts[a]}&{query_key(r) for r in parts[b]})
        assert not({normalize(r['text']) for r in parts[a]}&{normalize(r['text']) for r in parts[b]})
    summary_path=out/'grouped_training/summary.json'; checks={'split_hashes_and_coverage':True,'group_query_sentence_overlap_zero':True,'total_rows':sum(map(len,parts.values()))}
    if summary_path.exists():
        summary=json.loads(summary_path.read_text());assert sha(resolve_recorded_path(summary['checkpoint_path']))==summary['checkpoint_sha256']
        for name,metric in summary['test'].items():
            path=out/'grouped_training'/f'{name}_predictions.jsonl';assert sha(path)==metric['predictions_sha256']
            result=metrics(read_rows(path),sorted({r['relation'] for r in parts['train']}),args.bootstrap)
            for key in ('n','correct','accuracy','macro_f1_all_nine_classes','confusion_matrix'):assert result[key]==metric[key],key
        checks.update({'checkpoint_hash_verified':True,'prediction_hashes_verified':True,'metric_recomputation_identical':True})
    dump(out/'verification.json',{'execution':provenance(args),'status':'passed','checks':checks});print(json.dumps(checks),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['audit','train','verify'],default='audit')
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--pretrained',type=Path,required=True);parser.add_argument('--legacy-root',type=Path,required=True)
    parser.add_argument('--checkpoint-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','mps'],default='mps');parser.add_argument('--bootstrap',type=int,default=2000)
    args=parser.parse_args();torch.set_num_threads(4);logging.basicConfig(level=logging.INFO)
    {'audit':audit,'train':train,'verify':verify}[args.phase](args)


if __name__=='__main__':main()
