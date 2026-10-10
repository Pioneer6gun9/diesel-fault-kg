#!/usr/bin/env python3
"""Unified command line for the diesel-engine KG experiments."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
from diesel_kg.paths import source_dependencies
from diesel_kg.pipelines.workflow import build_plan, reuse_models, step


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_config(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    for key in ('data_dir', 'paper_source', 'topic_entities', 'assets_dir', 'results_dir'):
        data[key] = project_path(data[key])
    return data


def environment():
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def check_output(output, config, simulation=False, resume=False):
    output = output.resolve()
    results = (ROOT / 'results').resolve()
    if output == results or not output.is_relative_to(results):
        raise ValueError('实验输出目录必须位于本项目的 results/ 下。')
    protected = [config['results_dir'], ROOT / 'results/.provenance']
    if not simulation:
        protected.append(ROOT / 'results/simulated')
    for saved in protected:
        saved = saved.resolve()
        if output == saved or output.is_relative_to(saved) or saved.is_relative_to(output):
            raise ValueError(f'已有实验目录受保护，请选择新的输出目录：{saved}')
    if output.exists() and not output.is_dir():
        raise ValueError(f'输出路径不是目录：{output}')
    if output.exists() and any(output.iterdir()) and not (resume and (output / 'pipeline.json').is_file()):
        raise ValueError(f'输出目录已有文件，请选择空目录：{output}')
    return output


def execute_plan(plan, output, dry_run=False, prepare=None, mode='inference', resume=False):
    print(f'输出目录：{output}', flush=True)
    for index, record in enumerate(plan, 1):
        print(f'{index:02d}. {record["name"]}: {record["module"]}', flush=True)
    if dry_run:
        print('仅显示运行计划，未执行训练、推理或写入结果。')
        return
    output.mkdir(parents=True, exist_ok=True)
    logs = output / 'execution_logs'
    logs.mkdir(exist_ok=True)
    manifest = output / 'pipeline.json'
    previous = json.loads(manifest.read_text()) if resume and manifest.is_file() else {}
    if previous and previous.get('mode') != mode:
        raise ValueError('续跑模式与原运行不一致')
    prior_steps = {item['name']: item for item in previous.get('steps', [])}
    record = {**previous, 'status': 'running', 'mode': mode,
              'started_unix': previous.get('started_unix', time.time()), 'steps': []}
    record.pop('error', None)
    if previous.get('status') == 'failed':
        record.setdefault('attempts', []).append({'status': 'failed',
            'finished_unix': previous.get('finished_unix'), 'error': previous.get('error')})
    elif previous.get('error'):
        record.setdefault('attempts', []).append({'status': 'failed',
            'error': previous['error'], 'observed_unix': time.time(),
            'recorded_from': 'stale_error_before_successful_resume'})
    markers = {
        'data_extraction_and_graph_analysis': ['stage_manifest.json', 'model_references.json', 'full_verification.json'],
        'NER_verification': ['ner_domain/adaptation/verification.json'],
        'NER_corpus': ['ner_domain/adaptation/corpus_entities.jsonl', 'ner_domain/adaptation/corpus_entities.summary.json'],
        'RE_split_and_audit': ['re_clean/split_manifest.json', 're_clean/clean_evaluation.json'],
        'RE_verification': ['re_clean/verification.json'],
        'RE_uncertainty': ['re_clean/uncertainty_audit.json'],
        'RE_checkpoint_replay': ['re_clean/grouped_training/full_cpu_checkpoint_replay.json'],
        'reference_graph_ablation': ['graph_ablation/analysis_summary.json', 'graph_ablation/run_config.json'],
        'reference_chain_qualification': ['graph_ablation/case_role_qualification/analysis_summary.json'],
        'reference_graph_verification': ['graph_ablation/independent_result_checks.json'],
        'model_graph': ['adapted_model_graph/execution.json'],
        'model_graph_verification': ['adapted_model_graph/graph_ablation/independent_result_checks.json'],
        'model_comparison': ['cross_model_silver_evaluation/summary.json'],
        'comparison_verification': ['cross_model_silver_evaluation/independent_verification.json'],
        'measured_summary': ['measured_metrics.csv', 'metrics_provenance.json'],
        'manuscript_figures_and_tables': ['manuscript/data_sources.json', 'manuscript/paper.pdf', 'manuscript/overleaf.zip'],
    }

    def fingerprint(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                digest.update(block)
        return digest.hexdigest()

    def save():
        manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')

    save()
    try:
        if prepare and not previous:
            record['recorded_checkpoints'] = prepare()
            save()
        for item in record.get('recorded_checkpoints', []):
            if fingerprint(item['path']) != item['sha256']:
                raise ValueError('已复用权重或特征分片发生变化：' + item['path'])
        for item in plan:
            log = logs / (item['name'] + '.log')
            stage = {**item, 'code_sha256': source_dependencies(item['module']),
                     'status': 'running', 'started_unix': time.time(), 'log': str(log)}
            prior = prior_steps.get(item['name'], {})
            artifacts = prior.get('artifacts_sha256', {})
            if (prior.get('status') == 'completed' and prior.get('command') == item['command']
                    and prior.get('code_sha256') == stage['code_sha256'] and artifacts
                    and all(Path(p).is_file() and fingerprint(p) == h for p, h in artifacts.items())):
                record['steps'].append({**prior, 'resume_verified_unix': time.time()})
                save()
                print(f'已校验续跑阶段：{item["name"]}', flush=True)
                continue
            record['steps'].append(stage)
            save()
            print(f'执行 {item["name"]}；日志：{log}', flush=True)
            with log.open('w') as stream:
                completed = subprocess.run(item['command'], cwd=ROOT, env=environment(),
                                           stdout=stream, stderr=subprocess.STDOUT)
            stage.update(exit_code=completed.returncode, finished_unix=time.time(),
                         status='completed' if completed.returncode == 0 else 'failed')
            paths = [output / name for name in markers.get(item['name'], [])]
            stage['artifacts_sha256'] = {str(p): fingerprint(p) for p in paths if p.is_file()}
            stage['code_unchanged_during_execution'] = source_dependencies(item['module']) == stage['code_sha256']
            if not stage['code_unchanged_during_execution']:
                stage['status'] = 'failed'
                save()
                raise RuntimeError('阶段执行期间源码发生变化：' + item['name'])
            save()
            if completed.returncode:
                raise RuntimeError(f'{item["name"]} 失败，请查看 {log}')
        record.update(status='completed', finished_unix=time.time())
        save()
    except Exception as error:
        record.update(status='failed', error=str(error), finished_unix=time.time())
        save()
        raise


def run_project(args, config):
    output = check_output(project_path(args.output), config, resume=args.resume)
    if args.resume:
        doctor(config)
    args.output = output
    plan = build_plan(args, config)
    prepare = None if args.train else lambda: reuse_models(config['results_dir'], output)
    execute_plan(plan, output, args.dry_run, prepare,
                 mode='train_all_models' if args.train else 'recorded_checkpoints', resume=args.resume)


def graph(args, config):
    output = check_output(project_path(args.output), config)
    selected = project_path(args.experiment) if args.experiment else config['results_dir']
    plan = [step('graph', 'diesel_kg.pipelines.model_graph', '--results-dir', selected,
                 '--assets-dir', config['assets_dir'],
                 '--output', output, '--device', args.device)]
    execute_plan(plan, output, args.dry_run)


def plot(args, config):
    output = check_output(project_path(args.output), config)
    root = project_path(args.experiment) if args.experiment else config['results_dir']
    plan = [step('reference_graph_figures', 'diesel_kg.visualization.ablation',
                 '--graph-ablation', root / 'graph_ablation', '--output', output / 'reference_graph'),
            step('adapted_graph_figures', 'diesel_kg.visualization.adapted_graph',
                 '--graph-ablation', root / 'adapted_model_graph/graph_ablation', '--output', output / 'adapted_graph'),
            step('NER_figures', 'diesel_kg.visualization.ner', '--adaptation', root / 'ner_domain/adaptation',
                 '--output', output / 'ner'),
            step('manuscript_figures_and_tables', 'diesel_kg.visualization.manuscript',
                 '--results-dir', root, '--output', output / 'manuscript')]
    execute_plan(plan, output, args.dry_run)


def manuscript(args, config):
    output = check_output(project_path(args.output), config)
    root = project_path(args.experiment) if args.experiment else config['results_dir']
    flags = ['--results-dir', root, '--output', output]
    if args.compile_pdf:
        flags.append('--compile')
    execute_plan([step('manuscript_figures_and_tables', 'diesel_kg.visualization.manuscript', *flags)],
                 output, args.dry_run)


def simulate(args, config):
    output = check_output(project_path(args.output), config, simulation=True)
    if args.dry_run:
        print(f'SIMULATED：七系统 × {args.cases_per_system} 案例，seed={args.seed}；输出：{output}')
        print('只生成验证流程用的 CSV/JSON，不训练模型、不写入实测目录。')
        return
    from diesel_kg.data.simulation import build, write_bundle
    manifest = write_bundle(output, build(args.seed, args.cases_per_system), config['paper_source'])
    print(json.dumps({'data_status': 'SIMULATED', 'output': str(output),
                      'summary': manifest['summary'], 'validation': manifest['validation']}, ensure_ascii=False, indent=2))


def status(config):
    print(f'代码：{ROOT / "src/diesel_kg"}')
    checkpoints = [
        config['results_dir'] / 'ner/source_roberta/best_roberta_ner_model.pt',
        config['results_dir'] / 'ner/bert_identity_repair_rebuilt/best_roberta_ner_model.pt',
        config['results_dir'] / 'recorded_RE_training/original_re/best_re_model/best_model.pt',
        config['results_dir'] / 'recorded_RE_training/re_locator_fixed/best_re_model/best_model.pt',
        config['results_dir'] / 'ner_domain/adaptation/domain_adapted_ner_checkpoint.pt',
        config['results_dir'] / 're_clean/grouped_training/best_model.pt',
    ]
    print(f'已有模型权重：{sum(p.is_file() for p in checkpoints)}/{len(checkpoints)}')
    print(f'完整实验：{config["results_dir"]}')
    metrics = config['results_dir'] / 'measured_metrics.csv'
    print(f'已有实测汇总：{metrics}')
    if metrics.is_file():
        with metrics.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                print(f'  {row["experiment"]} | {row["metric"]}: {row["value"]}')
    else:
        print('  尚未找到实测汇总文件。')
    simulation = ROOT / 'results/simulated/manifest.json'
    if simulation.is_file():
        print(f'正文验证模拟数据（SIMULATED）：{simulation.parent}')
        workbook = simulation.parent / '论文验证模拟数据_SIMULATED.xlsx'
        if workbook.is_file():
            print(f'模拟表格（SIMULATED）：{workbook}')


def assets(args, config):
    flags = ['--assets-dir', config['assets_dir']]
    if not args.download:
        flags.append('--verify-only')
    completed = subprocess.run([sys.executable, '-m', 'diesel_kg.data.assets', *map(str, flags)],
                               cwd=ROOT, env=environment())
    raise SystemExit(completed.returncode)


def doctor(config):
    """Check local input integrity and the active environment without experiments."""
    manifest = json.loads((ROOT / 'data/manifest.json').read_text(encoding='utf-8'))
    errors = []
    for name, expected in manifest['files'].items():
        path = ROOT / name
        if not path.is_file():
            errors.append(f'缺少文件：{name}')
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                digest.update(block)
        if digest.hexdigest() != expected['sha256']:
            errors.append(f'文件内容变化：{name}')
    for key in ('data_dir', 'paper_source', 'topic_entities', 'assets_dir', 'results_dir'):
        if not config[key].is_relative_to(ROOT):
            errors.append(f'配置仍指向项目外部：{key}')
    import torch, transformers, gensim, numpy, pandas, scipy, sklearn, networkx, jieba, matplotlib, safetensors
    versions = {module.__name__: module.__version__ for module in
                (torch, transformers, gensim, numpy, pandas, scipy, sklearn, networkx, jieba, matplotlib, safetensors)}
    print(json.dumps({'status': 'passed' if not errors else 'failed', 'input_files_checked': len(manifest['files']),
                      'python': sys.executable, 'python_base': sys.base_prefix,
                      'versions': versions, 'errors': errors}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Diesel Fault Knowledge Graph — 柴油机故障知识图谱',
        epilog='直接运行 ./run 启动完整实验流程；./run --train 重新训练模型。')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/default.json')
    parser.add_argument('--assets-dir', type=Path, help='覆盖默认预训练模型和 tokenizer 目录')
    workflow = parser.add_argument_group('实验运行')
    workflow.add_argument('--output', default='results/run', help='本次实验输出目录，默认 results/run')
    workflow.add_argument('--train', action='store_true', help='重新训练全部模型后执行完整实验')
    workflow.add_argument('--resume', action='store_true', help='校验已有本次运行记录，继续未完成阶段')
    workflow.add_argument('--device', choices=['cpu', 'mps'], default=None, help='训练设备，默认自动选择')
    workflow.add_argument('--dry-run', action='store_true', help='显示执行步骤，不启动实验')
    sub = parser.add_subparsers(dest='command', title='其他操作')
    sub.add_parser('status', help='查看已有模型和实测结果，不执行实验')
    sub.add_parser('doctor', help='校验项目内数据、预训练模型和运行环境')
    p = sub.add_parser('graph', help='使用已有域适配 NER 和分组 RE，重跑关系推理和图谱实验')
    p.add_argument('--output', default='results/graph_rerun')
    p.add_argument('--experiment', help='包含输入、模型和预测的一套完整实验目录')
    p.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('plot', help='从已有真实数据重新出图')
    p.add_argument('--output', default='results/figures')
    p.add_argument('--experiment', help='指定完整实验目录，默认使用已有实测实验')
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('simulate', help='生成明确标注 SIMULATED 的正文验证流程数据，不替代实测结果')
    p.add_argument('--output', default='results/simulated')
    p.add_argument('--seed', type=int, default=20261008)
    p.add_argument('--cases-per-system', type=int, default=12)
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('manuscript', help='从实测结果生成当前投稿稿的彩色图、左对齐表格和源码包')
    p.add_argument('--output', default='results/manuscript_run')
    p.add_argument('--experiment', help='指定完整实验目录，默认使用已有实测实验')
    p.add_argument('--compile', action='store_true', dest='compile_pdf', help='使用 latexmk 编译 PDF')
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('assets', help='校验预训练模型资源；--download 下载缺少的固定版本资源')
    p.add_argument('--download', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.assets_dir:
        config['assets_dir'] = args.assets_dir.resolve()
    if args.command is None and args.device is None:
        args.device = config['training_device']
        if args.device == 'auto':
            import torch
            args.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    if args.command == 'status':
        status(config)
    elif args.command == 'doctor':
        doctor(config)
    else:
        try:
            if args.command is None:
                run_project(args, config)
            else:
                globals()[args.command](args, config)
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))


if __name__ == '__main__':
    main()
