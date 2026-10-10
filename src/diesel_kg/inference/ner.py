"""NER corpus inference with occurrence spans retained until canonical aggregation."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import torch
from torch import nn
from transformers import BertConfig, BertModel, RobertaConfig, RobertaModel
from diesel_kg.models.loading import extract_long_cases, file_sha256, load_tokenizer
import diesel_kg.models.loading as model_pipeline_module


class OccurrenceNER(nn.Module):
    def __init__(self, config, labels, encoder):
        super().__init__()
        self.roberta = encoder(config)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(config.hidden_size, labels)
    def forward(self, input_ids, attention_mask):
        return self.classifier(self.dropout(self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state))


def load_ner(checkpoint, tokenizer_dir, device):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    label2id = saved['label2id']
    config_values = saved.get('config', {}).get('encoder_config') or json.loads((tokenizer_dir / 'config.json').read_text())
    variant = saved.get('variant', 'source_roberta')
    is_bert = saved.get('config', {}).get('encoder_class') == 'BertModel' or variant.startswith('bert_identity_repair')
    config_cls, encoder = (BertConfig, BertModel) if is_bert else (RobertaConfig, RobertaModel)
    model = OccurrenceNER(config_cls.from_dict(config_values), len(label2id), encoder)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.eval().to(device)
    return model, load_tokenizer(tokenizer_dir), {int(i): label for label, i in label2id.items()}, variant


def run(args):
    start = time.perf_counter(); torch.set_num_threads(4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    execution = {'sys_executable': sys.executable, 'sys_argv': list(sys.argv),
                 'script_path': str(Path(__file__).resolve()), 'script_sha256': file_sha256(Path(__file__)),
                 'model_pipeline_path': str(Path(model_pipeline_module.__file__).resolve()),
                 'model_pipeline_sha256': file_sha256(Path(model_pipeline_module.__file__)),
                 'tokenizer_files_sha256': {path.name: file_sha256(path) for path in sorted(args.tokenizer.iterdir())
                                            if path.is_file() and path.suffix in ('.json', '.txt')},
                 'python_version': sys.version, 'torch_version': str(torch.__version__),
                 'device': args.device, 'batch_size': args.batch_size}
    device = torch.device(args.device)
    model, tokenizer, id2label, variant = load_ner(args.checkpoint, args.tokenizer, device)
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    digest = file_sha256(args.checkpoint); counts = {'cases': len(cases), 'mentions': 0, 'one_character_mentions': 0, 'offset_checks': 0}
    with args.output.open('w', encoding='utf-8') as handle:
        for first in range(0, len(cases), args.batch_size):
            batch = cases[first:first + args.batch_size]
            predictions = extract_long_cases(model, tokenizer, [row['text'] for row in batch], id2label, device, 0., args.batch_size)
            for row, entities in zip(batch, predictions):
                for item in entities:
                    assert row['text'][item['start']:item['end']] == item['text']
                    signature = f'{row["case_id"]}:{item["start"]}:{item["end"]}:{item["type"]}'
                    item['occurrence_id'] = 'mention_' + hashlib.sha256(signature.encode()).hexdigest()[:20]
                    counts['mentions'] += 1; counts['one_character_mentions'] += int(len(item['text']) == 1); counts['offset_checks'] += 1
                handle.write(json.dumps({**row, 'entities': entities, 'status': 'model_predicted_mentions_not_gold',
                                         'ner_variant': variant, 'ner_checkpoint_sha256': digest}, ensure_ascii=False) + '\n')
            if first % 160 == 0: print(f'Occurrence NER {min(first+len(batch),len(cases))}/{len(cases)}', flush=True)
    metadata = {'status': 'completed', 'counts': counts, 'variant': variant,
                'execution': execution,
                'checkpoint_path': str(args.checkpoint), 'checkpoint_sha256': digest,
                'cases_sha256': file_sha256(args.cases), 'output_sha256': file_sha256(args.output),
                'offset_convention': 'zero-based half-open character [start,end)',
                'mention_deduplication': 'none by surface text; only overflow tokens reconciled using most-context window',
                'elapsed_seconds': time.perf_counter() - start,
                'evidence_boundary': 'Unlabelled original 1522-case corpus; these are predictions.'}
    args.output.with_suffix('.summary.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


def fixture_checks(tokenizer_directory, output):
    tokenizer = load_tokenizer(tokenizer_directory)
    pump = tokenizer.convert_tokens_to_ids('泵')
    class Fixture(nn.Module):
        def forward(self, input_ids, attention_mask):
            values = torch.full((*input_ids.shape, 3), -50.)
            values[:, :, 0] = 50.
            values[input_ids == pump] = torch.tensor([-50., 50., -50.])
            return values
    texts = ['泵故障，更换泵。', '泵' + ('发动机。' * 180) + '泵']
    predictions = extract_long_cases(Fixture(), tokenizer, texts,
                    {0: 'O', 1: 'B-故障部位', 2: 'I-故障部位'}, torch.device('cpu'), 0., 8)
    checks = {'duplicate_mentions_kept': len(predictions[0]) == 2,
              'duplicate_start_offsets': [item['start'] for item in predictions[0]],
              'overflow_mentions_kept': len(predictions[1]) == 2,
              'overflow_exact_end': predictions[1][-1]['end'] == len(texts[1]),
              'all_spans_grounded': all(text[item['start']:item['end']] == item['text']
                                        for text, items in zip(texts, predictions) for item in items)}
    if not all(checks[key] for key in ('duplicate_mentions_kept', 'overflow_mentions_kept',
                                     'overflow_exact_end', 'all_spans_grounded')):
        raise AssertionError(checks)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(checks, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(checks, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--cases', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        fixture_checks(args.tokenizer, args.output)
    else:
        if args.checkpoint is None or args.cases is None:
            parser.error('--checkpoint and --cases are required for actual model inference')
        run(args)


if __name__ == '__main__': main()
