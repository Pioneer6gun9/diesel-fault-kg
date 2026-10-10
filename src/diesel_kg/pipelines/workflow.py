"""Complete experiment workflow, with all stages sharing one output directory."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import sys


def step(name, module, *args):
    return {'name': name, 'module': module,
            'command': [sys.executable, '-m', module, *map(str, args)]}


def reuse_models(saved: Path, output: Path):
    """Copy mutable model records; share immutable weights within the project.

    These remain records of earlier training, not training performed by a replay.
    The model-specific verification stages check the copied records and weights.
    """
    saved, output = Path(saved).resolve(), Path(output).resolve()
    runs = ['ner/source_roberta', 'ner/bert_identity_repair_rebuilt',
            'ner_domain/adaptation', 're_clean']
    records = []
    for run in runs:
        source, target = saved / run, output / run
        if not source.is_dir():
            raise FileNotFoundError(f'Missing selected model run: {source}')
        if target.exists():
            raise ValueError(f'Refuse to replace a model run: {target}')
    for run in runs:
        source, target = saved / run, output / run
        for path in sorted(source.rglob('*')):
            if not path.is_file() or path.suffix == '.log':
                continue
            if path.name in ('corpus_entities.jsonl', 'corpus_entities.summary.json'):
                continue
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix in ('.pt', '.bin', '.safetensors'):
                destination.symlink_to(os.path.relpath(path.resolve(), destination.parent))
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(8 << 20), b''):
                        digest.update(block)
                records.append({'path': str(destination), 'source': str(path),
                                'sha256': digest.hexdigest(), 'mode': 'recorded_checkpoint'})
            else:
                shutil.copy2(path, destination)
    return records


def build_plan(args, config):
    output = Path(args.output)
    saved = config['results_dir']
    assets = config['assets_dir']
    cases = output / 'inputs/cases_all.jsonl'
    source = output / 'inputs/original_source/RelationExtraction.py'
    pretrained = assets / 'pretrained/bert-base-chinese'
    tokenizer = assets / 'pretrained/chinese-roberta-wwm-ext'
    adaptation = output / 'ner_domain/adaptation'
    re_output = output / 're_clean'
    legacy = (output if args.train else saved) / 'recorded_RE_training'
    flags = ['--data-dir', config['data_dir'], '--paper-source', config['paper_source'],
             '--assets-dir', assets, '--topic-entities', config['topic_entities'],
             '--output', output, '--phase', 'all', '--device', args.device,
             '--inference-device', config['inference_device'],
             '--recorded-results', saved / 'recorded_RE_training',
             '--ner-mode', 'train' if args.train else 'recorded',
             '--re-mode', 'train' if args.train else 'recorded', '--resume']
    if not args.train:
        flags += ['--ner-results', output / 'ner']
    plan = [step('data_extraction_and_graph_analysis', 'diesel_kg.pipelines.core', *flags)]
    checkpoint = output / 'ner/bert_identity_repair_rebuilt/best_roberta_ner_model.pt'
    if args.train:
        plan.append(step('NER_training', 'diesel_kg.training.domain_ner',
                         '--data', output / 'inputs/ner_data', '--checkpoint', checkpoint,
                         '--tokenizer', tokenizer, '--cases', cases, '--output', adaptation))
    plan += [
        step('NER_verification', 'diesel_kg.validation.domain_ner', '--adaptation', adaptation,
             '--primary-predictions', output / 'ner/bert_identity_repair_rebuilt/test_predictions.jsonl'),
        step('NER_corpus', 'diesel_kg.inference.domain_ner', '--adaptation', adaptation,
             '--base-checkpoint', checkpoint, '--tokenizer', tokenizer, '--cases', cases,
             '--output', adaptation / 'corpus_entities.jsonl', '--device', config['inference_device']),
    ]
    re_flags = ['--source', source, '--data-dir', output / 'inputs/re_data',
                '--pretrained', pretrained, '--legacy-root', legacy,
                '--checkpoint-root', legacy, '--output', re_output, '--device', args.device]
    plan.append(step('RE_split_and_audit', 'diesel_kg.training.grouped_re', '--phase', 'audit', *re_flags))
    if args.train:
        plan.append(step('RE_training', 'diesel_kg.training.grouped_re', '--phase', 'train', *re_flags))
    plan += [
        step('RE_verification', 'diesel_kg.training.grouped_re', '--phase', 'verify', *re_flags),
        step('RE_uncertainty', 'diesel_kg.evaluation.re_uncertainty', '--output', re_output),
        step('RE_checkpoint_replay', 'diesel_kg.evaluation.re_checkpoint', '--output', re_output,
             '--source', source, '--pretrained', pretrained),
        step('reference_graph_ablation', 'diesel_kg.graph.ablation', '--cases', cases,
             '--entities', output / 'occurrence_bert_identity_repair_rebuilt/corpus_entities.jsonl',
             '--triples', output / 'relation_threshold_bert_identity_repair_rebuilt/threshold0_qualified_triples.jsonl',
             '--candidate-graph', output / 'relation_threshold_bert_identity_repair_rebuilt/threshold0_qualified_predicted_graph.json',
             '--forced-splits', adaptation / 'domain_graph_exclusion_case_ids.json', '--output', output / 'graph_ablation'),
        step('reference_chain_qualification', 'diesel_kg.evaluation.chain_qualification',
             '--graph-ablation', output / 'graph_ablation', '--output', output / 'graph_ablation/case_role_qualification'),
        step('reference_graph_verification', 'diesel_kg.validation.grouped_graph', '--graph-ablation', output / 'graph_ablation'),
        step('model_graph', 'diesel_kg.pipelines.model_graph', '--results-dir', output,
             '--assets-dir', assets, '--output', output / 'adapted_model_graph', '--device', config['inference_device']),
        step('model_graph_verification', 'diesel_kg.validation.grouped_graph',
             '--graph-ablation', output / 'adapted_model_graph/graph_ablation'),
        step('model_comparison', 'diesel_kg.evaluation.cross_model',
             '--old-root', output / 'graph_ablation', '--new-root', output / 'adapted_model_graph/graph_ablation',
             '--output', output / 'cross_model_silver_evaluation'),
        step('comparison_verification', 'diesel_kg.validation.cross_model',
             '--comparison', output / 'cross_model_silver_evaluation'),
        step('graph_figures', 'diesel_kg.visualization.ablation',
             '--graph-ablation', output / 'graph_ablation', '--output', output / 'figures/reference_graph'),
        step('model_graph_figures', 'diesel_kg.visualization.adapted_graph',
             '--graph-ablation', output / 'adapted_model_graph/graph_ablation', '--output', output / 'figures/model_graph'),
        step('NER_figures', 'diesel_kg.visualization.ner', '--adaptation', adaptation,
             '--output', output / 'figures/ner'),
        step('measured_summary', 'diesel_kg.evaluation.summary', '--results-dir', output, '--replace'),
        step('manuscript_figures_and_tables', 'diesel_kg.visualization.manuscript',
             '--results-dir', output, '--output', output / 'manuscript', '--compile'),
    ]
    return plan
