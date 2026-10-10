"""Actual complete corpus NER using the selected exploratory adapted head.

Uses the unchanged occurrence-preserving loader/long-case decoder; verifies the
adapted checkpoint equals base encoder plus selected classifier before inference.
Does not change training selection or any supplied gold.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adaptation', type=Path, required=True)
    parser.add_argument('--base-checkpoint', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    import torch
    from diesel_kg.inference.ner import load_ner
    from diesel_kg.models.loading import extract_long_cases, file_sha256
    import diesel_kg.inference.ner as loader_module
    import diesel_kg.models.loading as decoder_module
    torch.set_num_threads(4)
    started = time.monotonic()
    summary = json.loads((args.adaptation/'summary.json').read_text())
    verifier = json.loads((args.adaptation/'verification.json').read_text())
    if summary['status'] != 'completed' or verifier['status'] != 'passed':
        raise ValueError('Adapted model must have completed and passed independent verification')
    checkpoint = args.adaptation/'domain_adapted_ner_checkpoint.pt'
    head_path = args.adaptation/'best_classifier_head.pt'
    if file_sha256(args.base_checkpoint) != summary['base_checkpoint_sha256'] or file_sha256(checkpoint) != summary['selected_checkpoint_sha256'] or file_sha256(head_path) != summary['classifier_head_sha256']:
        raise ValueError('Base/head/selected checkpoint SHA differs from completed experiment')
    base = torch.load(args.base_checkpoint, map_location='cpu', weights_only=True, mmap=True)
    selected = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    head = torch.load(head_path, map_location='cpu', weights_only=True, mmap=True)
    enc = [k for k in base['model_state_dict'] if k.startswith('roberta.')]
    if not all(torch.equal(base['model_state_dict'][k], selected['model_state_dict'][k]) for k in enc):
        raise ValueError('Adapted full encoder differs from frozen base')
    if not torch.equal(selected['model_state_dict']['classifier.weight'],head['state_dict']['1.weight']) or not torch.equal(selected['model_state_dict']['classifier.bias'],head['state_dict']['1.bias']):
        raise ValueError('Adapted full checkpoint does not contain selected classifier head')
    del base, selected, head
    model, tokenizer, labels, variant = load_ner(checkpoint,args.tokenizer,torch.device(args.device))
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    if len(cases) != 1522:
        raise ValueError('This corpus stage requires all supplied 1522 numbered cases')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    counts = {'cases':0,'mentions':0,'one_character_mentions':0,'offset_checks':0,'unique_occurrence_ids':0}
    seen = set()
    digest = file_sha256(checkpoint)
    with args.output.open('w') as handle:
        for first in range(0,len(cases),8):
            batch = cases[first:first+8]
            entities = extract_long_cases(model,tokenizer,[c['text'] for c in batch],labels,torch.device(args.device),0.,8)
            for case,mentions in zip(batch,entities):
                counts['cases'] += 1
                for item in mentions:
                    if not 0 <= item['start'] < item['end'] <= len(case['text']) or case['text'][item['start']:item['end']] != item['text']:
                        raise ValueError('Mention is not grounded at its saved half-open character offsets')
                    signature = f'{case["case_id"]}:{item["start"]}:{item["end"]}:{item["type"]}'
                    item['occurrence_id'] = 'mention_'+hashlib.sha256(signature.encode()).hexdigest()[:20]
                    if item['occurrence_id'] in seen:
                        raise ValueError('Duplicate occurrence identity within complete corpus')
                    seen.add(item['occurrence_id'])
                    counts['mentions'] += 1; counts['offset_checks'] += 1
                    counts['one_character_mentions'] += len(item['text']) == 1
                handle.write(json.dumps({**case,'entities':mentions,'status':'model_predicted_mentions_not_gold',
                                          'ner_variant':variant,'ner_checkpoint_sha256':digest},ensure_ascii=False)+'\n')
            if first % 160 == 0:
                print(f'Adapted occurrence NER {min(first+len(batch),len(cases))}/{len(cases)}',flush=True)
    counts['unique_occurrence_ids'] = len(seen)
    metadata = {'status':'completed','counts':counts,'variant':variant,
                'execution':{'argv':sys.argv,'python':sys.executable,'device':args.device,
                             'script_path':str(Path(__file__).resolve()),'script_sha256':file_sha256(Path(__file__)),
                             'loader_path':loader_module.__file__,'loader_sha256':file_sha256(Path(loader_module.__file__)),
                             'decoder_path':decoder_module.__file__,'decoder_sha256':file_sha256(Path(decoder_module.__file__))},
                'base_checkpoint_path':str(args.base_checkpoint),'base_checkpoint_sha256':summary['base_checkpoint_sha256'],
                'head_path':str(head_path),'head_sha256':summary['classifier_head_sha256'],
                'checkpoint_path':str(checkpoint),'checkpoint_sha256':digest,
                'frozen_encoder_tensors_verified_equal':len(enc),'selected_head_verified_equal':True,
                'cases_path':str(args.cases),'cases_sha256':file_sha256(args.cases),
                'output_sha256':file_sha256(args.output),'elapsed_seconds':time.monotonic()-started,
                'offset_convention':'zero-based half-open character [start,end)',
                'mention_deduplication':'none by surface; occurrence identities preserve case and offsets',
                'domain_protocol_sha256':summary['protocol_sha256'],
                'evidence_boundary':summary['evidence_boundary']+' Full corpus includes the internal held-out domain cases; downstream graph fitting must enforce the saved train/dev/test exclusion groups.'}
    args.output.with_suffix('.summary.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(metadata,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
