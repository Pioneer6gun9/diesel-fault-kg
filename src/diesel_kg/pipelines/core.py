"""Execute the completed paper framework with frozen inputs and verifiable stages.

Original algorithms and recorded model reuse are distinguished from disclosed
repairs. Historical graph analysis and newly predicted end-to-end graphs are
separate branches. Missing expert, causal gold or event data never becomes a
numerical paper result. Run --help for training and recorded-checkpoint modes.
"""
from __future__ import annotations
from diesel_kg.paths import PROJECT_ROOT, CONFIG_DIR, resolve_script, source_dependencies, resolve_recorded_path
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = PROJECT_ROOT
NER_VARIANTS = ('source_roberta', 'bert_identity_repair_rebuilt')
NER_OPTIONS = (*NER_VARIANTS, 'bert_identity_repair', 'bert_identity_repair_eager')
RE_VARIANTS = ('original_re', 're_locator_fixed')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read(path): return json.loads(Path(path).read_text())


def local_dependencies(script):
    return source_dependencies(script, HERE)


def artifact_hashes(paths):
    files=[]
    for value in paths:
        path=Path(value)
        if path.is_dir(): files.extend(p for p in path.rglob('*') if p.is_file())
        else: files.append(path)
    return {str(path):sha(path) for path in sorted(set(files))}


class Runner:
    """Resume only when the command, inputs, code and saved artifacts agree."""
    def __init__(self, output, resume):
        self.output = output; self.resume = resume
        self.path = output / 'stage_manifest.json'
        self.manifest = read(self.path) if self.path.exists() else {}
        self.logs = output / 'execution_logs'; self.logs.mkdir(exist_ok=True)

    def run(self, name, script, args, inputs, artifacts, external_completion=None):
        command = [sys.executable, str(resolve_script(script, HERE)), *map(str, args)]
        signature = {'command': command, 'local_code_sha256': local_dependencies(script),
                     'inputs': artifact_hashes(inputs), 'required_artifacts':sorted(str(Path(p)) for p in artifacts)}
        prior = self.manifest.get(name, {})
        if self.resume and prior.get('status') == 'completed' and prior.get('signature') == signature:
            expected = prior.get('artifacts_sha256', {})
            if expected and all(Path(p).is_file() and str(p) in expected for p in artifacts) and all(Path(p).is_file() and sha(p) == h for p,h in expected.items()):
                print(f'{name}: verified resume', flush=True)
                return {'mode':'verified_stage_resume','executed_this_invocation':False}
        if external_completion:
            if external_completion['command'] != command or external_completion['status'] != 'completed' or external_completion['exit_code'] != 0:
                raise ValueError('External completion does not match the requested successful command')
            verified_files=external_completion.get('verified_files_sha256',{})
            if not verified_files or any(not Path(p).is_file() or sha(p)!=h for p,h in verified_files.items()):
                raise ValueError('Previously verified external inference evidence has changed before acceptance')
            current_completion=completed_ner_execution(self.output,args) if resolve_script(script,HERE)==resolve_script('diesel_kg.inference.ner',HERE) else None
            if not current_completion or current_completion['metadata_sha256']!=external_completion['metadata_sha256'] or current_completion['verified_files_sha256']!=verified_files:
                raise ValueError('External execution metadata or complete tokenizer inventory has changed before acceptance')
            item={'status':'completed','signature':signature,'exit_code':0,
                  'execution_mode':'verified_external_completed_execution',
                  'external_execution':external_completion,
                  'artifacts_sha256':artifact_hashes(artifacts),'missing_artifacts':[]}
            self.manifest[name]=item; write(self.path,self.manifest)
            print(f'{name}: verified already completed external execution',flush=True)
            return {'mode':'verified_external_completed_execution','executed_this_invocation':False}
        log = self.logs / (name + '.log'); began = time.time()
        item = {'status': 'running', 'signature': signature, 'log': str(log), 'started_unix': began}
        self.manifest[name] = item; write(self.path, self.manifest)
        print(f'{name}: executing; log={log}', flush=True)
        with log.open('w') as f:
            result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'TOKENIZERS_PARALLELISM': 'false'})
        item.update(exit_code=result.returncode, elapsed_seconds=time.time()-began)
        missing = [str(p) for p in artifacts if not Path(p).is_file()]
        item['status'] = 'failed' if result.returncode or missing else 'completed'
        item['missing_artifacts'] = missing
        item['artifacts_sha256'] = {str(p): sha(p) for p in artifacts if Path(p).is_file()}
        write(self.path, self.manifest)
        if item['status'] != 'completed': raise RuntimeError(f'{name} failed; inspect {log}')
        print(f'{name}: completed in {item["elapsed_seconds"]:.1f}s', flush=True)
        return {'mode':'executed_this_invocation','executed_this_invocation':True}


def copy_frozen(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha(source) != sha(destination): raise ValueError(f'Frozen input has changed: {destination}')
    else: shutil.copy2(source, destination)
    return {'source_path': str(source), 'frozen_path': str(destination), 'bytes': source.stat().st_size, 'sha256': sha(source)}


def prepare(a, runner):
    kg = a.data_dir; inputs = a.output / 'inputs'
    inputs.mkdir(exist_ok=True)
    records = []
    # Copy only explicit experiment inputs; never import/copy database builders.
    paths = {'案例语料.txt': 'raw/cases.txt', 'neo4j_query_table_data.csv': 'raw/historical_export.csv',
             'NER/EntityExtraction.py': 'original_source/EntityExtraction.py',
             'RE/RelationExtraction.py': 'original_source/RelationExtraction.py'}
    paths.update({f'NER/ner_data_all_generate/{s}.txt': f'ner_data/{s}.txt' for s in ('train','dev','test')})
    paths.update({f'RE/re_data_generate/{s}.json': f're_data/{s}.json' for s in ('train','dev','test')})
    for rel, target in paths.items(): records.append(copy_frozen(kg / rel, inputs / target))
    records.append(copy_frozen(a.topic_entities, inputs / 'topic_dictionary_entities.jsonl'))
    paper = a.paper_source
    records.append(copy_frozen(paper, inputs / 'original_source/paper2.tex'))
    write(inputs / 'frozen_manifest.json', records)
    runner.run('data_audit','diesel_kg.data.audit',['--kg-build',kg,'--output-dir',inputs],
               [kg / p for p in paths], [inputs/'data_audit.json',inputs/'cases_all.jsonl',inputs/'source_manifest.json'])
    runner.run('parse_graph','diesel_kg.data.neo4j',[inputs/'raw/historical_export.csv','--output',inputs/'historical_graph.json'],
               [inputs/'raw/historical_export.csv'], [inputs/'historical_graph.json'])
    packages = {}
    for name in ('torch','transformers','numpy','gensim','networkx','scikit-learn','matplotlib','jieba'):
        packages[name] = importlib.metadata.version(name)
    write(a.output/'runtime_environment.json', {'python':sys.version,'executable':sys.executable,'packages':packages})
    runner.run('topics_frozen_dictionary','diesel_kg.graph.topics',
               ['--cases',inputs/'cases_all.jsonl','--entities',inputs/'topic_dictionary_entities.jsonl','--output-dir',a.output/'topics'],
               [inputs/'cases_all.jsonl',inputs/'topic_dictionary_entities.jsonl'],
               [a.output/'topics/analysis_summary.json',a.output/'topics/paper_fixed_K07/document_topic.csv',
                a.output/'topics/selected_best_K10/document_topic.csv'])


def analyses(a, runner):
    inputs = a.output/'inputs'; topics = a.output/'topics'
    for k in (7,10):
        model = topics/('paper_fixed_K07' if k==7 else 'selected_best_K10')
        out = a.output/f'graph_k{k}'
        runner.run(f'graph_K{k}','diesel_kg.graph.analysis',
          ['--topics-dir',topics,'--topic-model-dir',model,'--export-json',inputs/'historical_graph.json',
           '--cases',inputs/'cases_all.jsonl','--output',out],
          [inputs/'historical_graph.json',inputs/'cases_all.jsonl',model/'document_topic.csv',model/'case_system_probabilities.csv'],
          [out/'analysis_summary.json',out/'base_graph.json',out/'enhanced_graph.json',out/'enhanced_propagation_candidate_paths.jsonl'])
    runner.run('topic_graph_sensitivity','diesel_kg.evaluation.topic_sensitivity',
          ['--k7-dir',a.output/'graph_k7','--k10-dir',a.output/'graph_k10','--reference-k7-dir',a.recorded_results/'graph',
           '--output',a.output/'graph_topic_sensitivity.json'],
          [a.output/'graph_k7/enhanced_graph.json',a.output/'graph_k10/enhanced_graph.json',a.recorded_results/'graph/analysis_summary.json'],
          [a.output/'graph_topic_sensitivity.json'])
    for name,script in [('patterns','diesel_kg.graph.patterns'),('hub_hierarchy','diesel_kg.graph.hubs')]:
        runner.run(name,script,['--graph-dir',a.output/'graph_k7','--cases',inputs/'cases_all.jsonl','--output',a.output/name],
          [a.output/'graph_k7/base_graph.json',a.output/'graph_k7/enhanced_graph.json',inputs/'cases_all.jsonl'],
          [a.output/name/'analysis_summary.json'])
    system_args=['--graph-dir',a.output/'graph_k7','--output',a.output/'system_validation']
    system_inputs=[a.output/'graph_k7/enhanced_graph.json',a.output/'graph_k7/automatic_communities.json']
    if a.physical_system_labels:
        system_args += ['--labels',a.physical_system_labels]; system_inputs.append(a.physical_system_labels)
    runner.run('physical_system_validation','diesel_kg.evaluation.systems',system_args,system_inputs,
               [a.output/'system_validation/analysis_summary.json',a.output/'system_validation/node_annotation_inventory.csv'])
    extra=[]; evidence=[]
    for flag,path in [('--gold',a.gold),('--expert-scores',a.expert_scores),('--split-manifest',a.split_manifest)]:
        if path: extra += [flag,path]; evidence.append(path)
    runner.run('engineering','diesel_kg.evaluation.engineering',
          ['--graph-dir',a.output/'graph_k7','--cases',inputs/'cases_all.jsonl','--output',a.output/'engineering',
           '--repeats',3,*extra],
          [a.output/'graph_k7/base_graph.json',a.output/'graph_k7/enhanced_graph.json',inputs/'cases_all.jsonl',*evidence],
          [a.output/'engineering/analysis_summary.json',a.output/'engineering/structural_descriptive_metrics.json',
           a.output/'engineering/case_queries_and_inference.jsonl'])
    temporal_args=['--output',a.output/'temporal']; temporal_inputs=[]
    if a.events: temporal_args += ['--events',a.events]; temporal_inputs.append(a.events)
    if a.temporal_links: temporal_args += ['--links',a.temporal_links]; temporal_inputs.append(a.temporal_links)
    runner.run('temporal','diesel_kg.evaluation.temporal',temporal_args,temporal_inputs,[a.output/'temporal/analysis_summary.json'])
    runner.run('table1_original_chains','diesel_kg.evaluation.paper_chains',
          ['--cases',inputs/'cases_all.jsonl','--paper-targets',CONFIG_DIR / 'paper_targets.json','--output',a.output/'table1_literal_check'],
          [inputs/'cases_all.jsonl',CONFIG_DIR / 'paper_targets.json'],[a.output/'table1_literal_check/run_summary.json'])
    runner.run('graph_quality','diesel_kg.evaluation.graph_quality',
          ['--graph-dir',a.output/'graph_k7','--source-export',inputs/'historical_graph.json',
           '--cases',inputs/'cases_all.jsonl','--output',a.output/'graph_quality_audit'],
          [a.output/'graph_k7/enhanced_graph.json',inputs/'historical_graph.json',inputs/'cases_all.jsonl'],
          [a.output/'graph_quality_audit/quality_audit.json'])
    runner.run('relation_existence_audit','diesel_kg.evaluation.relation_existence',
          ['--audit-source-dir',inputs/'re_data','--output',a.output/'relation_existence'],
          [inputs/f're_data/{s}.json' for s in ('train','dev','test')],
          [a.output/'relation_existence/run_summary.json',a.output/'relation_existence/source_positive_only_audit.json'])
    runner.run('graph_modes_verification','diesel_kg.validation.graph_modes',
          ['--graph-dir',a.output/'graph_k7','--cases',inputs/'cases_all.jsonl','--patterns-dir',a.output/'patterns',
           '--hub-dir',a.output/'hub_hierarchy','--output',a.output/'graph_modes_verification.json'],
          [a.output/'graph_k7/enhanced_graph.json',a.output/'patterns/enhanced_pattern_instances.jsonl',
           a.output/'hub_hierarchy/analysis_summary.json',inputs/'cases_all.jsonl'],
          [a.output/'graph_modes_verification.json'])


def model_reference(checkpoint, summary, execution=None):
    data=read(summary)
    if data.get('status') not in ('complete','completed'):
        raise ValueError(f'Model training was not completed: {summary}')
    if data.get('valid_for_model_selection') is False:
        raise ValueError(f'Model selection was invalidated by its saved audit: {summary}')
    if 'best_dev_token_micro_f1' in data and abs(data['best_dev_token_micro_f1']-data['dev']['token_entity_micro_f1'])>1e-10:
        raise ValueError(f'Selected NER checkpoint does not reproduce its development score: {summary}')
    expected=data.get('checkpoint_sha256') or data.get('best_checkpoint_sha256')
    if not expected: raise ValueError(f'Training summary lacks checkpoint SHA256: {summary}')
    actual=sha(checkpoint)
    if expected and expected != actual: raise ValueError(f'Checkpoint hash mismatch: {checkpoint}')
    execution=execution or {'mode':'recorded_completed_training','executed_this_invocation':False}
    return {**execution,'checkpoint_path':str(checkpoint), 'checkpoint_sha256':actual,
            'summary_path':str(summary),'summary_sha256':sha(summary),'summary_status':data.get('status'),
            'not_retrained_in_this_invocation':not execution['executed_this_invocation']}


def verify_model_inputs(config_path, inputs, source_name, data_dir, suffix):
    config=read(config_path)
    checks={'source':config['source_sha256']==sha(inputs/'original_source'/source_name)}
    for split in ('train','dev','test'):
        checks[split]=config['data'][split]['sha256']==sha(inputs/data_dir/(split+suffix))
    if not all(checks.values()): raise ValueError(f'Recorded training inputs differ from frozen inputs: {config_path}: {checks}')
    return {'configuration_path':str(config_path),'configuration_sha256':sha(config_path),'source_and_split_hash_checks':checks}


def completed_ner_execution(output_root, args):
    """Reuse the agent's real completed inference only with exact command/SHA evidence."""
    flags={str(args[i]):str(args[i+1]) for i in range(0,len(args),2)}
    target=Path(flags['--output']); summary=target.with_suffix('.summary.json')
    record_files=[output_root/'ner/followthrough_commands.json',
                  output_root/'ner/bert_identity_repair_eager/execution.json',
                  output_root/'ner/bert_identity_repair_rebuilt/execution.json']
    if not (target.is_file() and summary.is_file()): return None
    records=[]
    for path in record_files:
        if path.is_file():
            value=read(path); records.extend(value if isinstance(value,list) else value.get('commands',[]))
    if not records: return None
    meta=read(summary); execution=meta.get('execution',{})
    command=[sys.executable,str(resolve_script('diesel_kg.inference.ner', HERE)),*map(str,args)]
    first_source=output_root/'ner/source_corpus_first_execution_summary.json'
    historical=read(first_source) if first_source.is_file() else {}
    def recorded_script_version(record):
        if record.get('script_sha256'): return record['script_sha256']
        # Older followthrough records did not include script SHA. Bind them
        # only to their preserved first-execution metadata, never a later rerun.
        if record.get('stage')=='source_roberta_corpus_inference' and historical.get('execution',{}).get('sys_argv')==command[1:]:
            return historical['execution'].get('script_sha256')
        return None
    match=[r for r in records if r.get('command')==command and r.get('status')=='completed' and r.get('exit_code')==0
           and recorded_script_version(r)==execution.get('script_sha256')]
    if not match or meta.get('status')!='completed': return None
    checks=[execution.get('sys_executable')==sys.executable,
            execution.get('sys_argv')==command[1:],
            execution.get('script_sha256')==sha(resolve_script('diesel_kg.inference.ner', HERE)),
            execution.get('model_pipeline_sha256')==sha(resolve_script('diesel_kg.models.loading', HERE)),
            execution.get('tokenizer_files_sha256')=={p.name:sha(p) for p in sorted(Path(flags['--tokenizer']).iterdir()) if p.is_file() and p.suffix in ('.json','.txt')},
            meta.get('checkpoint_sha256')==sha(flags['--checkpoint']),
            meta.get('cases_sha256')==sha(flags['--cases']),meta.get('output_sha256')==sha(target)]
    if not all(checks): return None
    verified_paths=[resolve_script('diesel_kg.inference.ner', HERE),resolve_script('diesel_kg.models.loading', HERE),
                    Path(flags['--checkpoint']),Path(flags['--cases']),target,summary]
    verified_paths.extend(p for p in sorted(Path(flags['--tokenizer']).iterdir()) if p.is_file() and p.suffix in ('.json','.txt'))
    return {**match[-1],'metadata_path':str(summary),'metadata_sha256':sha(summary),
            'verified_files_sha256':artifact_hashes(verified_paths),
            'exact_command_and_execution_hashes_verified':True}


def models_and_inference(a, runner):
    inputs=a.output/'inputs'
    re_pre=a.assets_dir/'tokenizers/bert-base-chinese'; ner_pre=a.assets_dir/'tokenizers/chinese-roberta-wwm-ext'
    ner_executions={}; re_executions={}
    if a.ner_mode=='train':
        for variant in a.variants:
            script='diesel_kg.training.verified_ner' if variant=='bert_identity_repair_rebuilt' else 'diesel_kg.training.source_ner'
            if variant=='bert_identity_repair_eager':
                raise ValueError('The failed eager diagnostic is preserved; use its executed snapshot for audit, or the verified rebuilt branch for training.')
            ner_executions[variant]=runner.run(f'NER_training_{variant}',script,
              ['--source',inputs/'original_source/EntityExtraction.py','--data',inputs/'ner_data','--pretrained',ner_pre,
               '--output',a.output/'ner','--device',a.device,'--variant',variant,'--force'],
              [inputs/'original_source/EntityExtraction.py',ner_pre,*[inputs/f'ner_data/{s}.txt' for s in ('train','dev','test')]],
              [a.output/f'ner/{variant}/summary.json',a.output/f'ner/{variant}/epoch_history.json',
               a.output/f'ner/{variant}/best_roberta_ner_model.pt',a.output/f'ner/{variant}/test_predictions.jsonl',a.output/f'ner/{variant}/config.json'])
        ner_root=a.output/'ner'
    else: ner_root=a.ner_results or a.output/'ner'
    re_root=a.recorded_results
    if a.re_mode=='train':
        re_root=a.output/'recorded_RE_training'
        for variant,locator in [('original_re','original'),('re_locator_fixed','offset')]:
            re_executions[variant]=runner.run(variant,'diesel_kg.training.source_re',
              ['--source',inputs/'original_source/RelationExtraction.py','--data-dir',inputs/'re_data','--pretrained-dir',re_pre,
               '--output-dir',re_root/variant,'--work-dir',a.output/'runtime'/variant,'--device',a.device,'--locator',locator],
              [inputs/'original_source/RelationExtraction.py',*[inputs/f're_data/{s}.json' for s in ('train','dev','test')],re_pre],
              [re_root/variant/'run_summary.json',re_root/variant/'run_configuration.json',re_root/variant/'best_re_model/best_model.pt'])
    references=read(a.output/'model_references.json') if (a.output/'model_references.json').exists() else {}
    for variant in RE_VARIANTS:
        ckpt=re_root/variant/'best_re_model/best_model.pt'
        references[variant]=model_reference(ckpt,re_root/variant/'run_summary.json',re_executions.get(variant))
        references[variant]['input_audit']=verify_model_inputs(re_root/variant/'run_configuration.json',inputs,'RelationExtraction.py','re_data','.json')
    for variant in a.variants:
        inference_device=a.bert_inference_device or a.inference_device if variant.startswith('bert_identity_repair') else a.inference_device
        ckpt=resolve_recorded_path(ner_root/variant/'best_roberta_ner_model.pt')
        references[variant]=model_reference(ckpt,ner_root/variant/'summary.json',ner_executions.get(variant))
        references[variant]['input_audit']=verify_model_inputs(ner_root/variant/'config.json',inputs,'EntityExtraction.py','ner_data','.txt')
        runner.run(f'NER_final_audit_{variant}','diesel_kg.evaluation.ner',
          ['--summarize-trained',ner_root/variant,'--pretrained',ner_pre],
          [ckpt,ner_root/variant/'summary.json',ner_root/variant/'test_predictions.jsonl',
           ner_root/variant/'epoch_history.json',ner_pre],
          [ner_root/variant/'test_strata.json',ner_root/variant/'checkpoint_reload_checks.json',
           ner_root/variant/'training_curve.png',ner_root/variant/'training_curve.svg'])
        # The audit appends deterministic file-stratum metrics to the summary;
        # record its final hash rather than the pre-audit summary hash.
        references[variant]=model_reference(ckpt,ner_root/variant/'summary.json',ner_executions.get(variant))
        references[variant]['input_audit']=verify_model_inputs(ner_root/variant/'config.json',inputs,'EntityExtraction.py','ner_data','.txt')
        write(a.output/'model_references.json',references)
        entities=a.output/f'occurrence_{variant}/corpus_entities.jsonl'
        ner_args=['--checkpoint',ckpt,'--tokenizer',ner_pre,'--cases',inputs/'cases_all.jsonl','--output',entities,'--device',inference_device,'--batch-size',8]
        runner.run(f'NER_occurrences_{variant}','diesel_kg.inference.ner',
          ner_args,[ckpt,inputs/'cases_all.jsonl',ner_pre],[entities,entities.with_suffix('.summary.json')],
          external_completion=completed_ner_execution(a.output,ner_args))
        out=a.output/f'occurrence_{variant}/re_locator_fixed'
        re_ckpt=re_root/'re_locator_fixed/best_re_model/best_model.pt'
        if inference_device!='cpu':
            runner.run(f'RE_device_consistency_{variant}','diesel_kg.evaluation.re_device',
              ['--source',inputs/'original_source/RelationExtraction.py','--checkpoint',re_ckpt,
               '--pretrained',re_pre,'--tokenizer',a.assets_dir/'tokenizers/bert-base-chinese',
               '--entities',entities,'--device',inference_device,
               '--output',out/'device_consistency_check.json'],
              [entities,re_ckpt,inputs/'original_source/RelationExtraction.py',re_pre,
               a.assets_dir/'tokenizers/bert-base-chinese'],[out/'device_consistency_check.json'])
        runner.run(f'RE_occurrences_{variant}','diesel_kg.inference.relations',
          ['--entities',entities,'--checkpoint',re_ckpt,'--source',inputs/'original_source/RelationExtraction.py',
           '--tokenizer',a.assets_dir/'tokenizers/bert-base-chinese','--pretrained',re_pre,'--output',out,'--device',inference_device],
          [entities,re_ckpt,inputs/'original_source/RelationExtraction.py',re_pre,a.assets_dir/'tokenizers/bert-base-chinese'],
          [out/'run_summary.json',out/'qualified_predicted_graph.json',out/'case_counts.json',out/'qualified_triples.jsonl',out/'all_occurrence_pair_predictions.jsonl.gz'])
        fresh_topics=a.output/f'topics_{variant}_dictionary'
        runner.run(f'LDA_new_NER_dictionary_{variant}','diesel_kg.graph.topics',
          ['--cases',inputs/'cases_all.jsonl','--entities',entities,'--output-dir',fresh_topics],
          [inputs/'cases_all.jsonl',entities],
          [fresh_topics/'analysis_summary.json',fresh_topics/'paper_fixed_K07/document_topic.csv'])
        summary=read(out/'run_summary.json')
        if summary['qualified_graph_edges']:
            graph_out=a.output/f'fresh_graph_{variant}'
            runner.run(f'fresh_graph_{variant}','diesel_kg.graph.analysis',
              ['--topics-dir',fresh_topics,'--export-json',out/'qualified_predicted_graph.json','--support-mode','model_occurrences',
               '--cases',inputs/'cases_all.jsonl','--output',graph_out],
              [out/'qualified_predicted_graph.json',inputs/'cases_all.jsonl',fresh_topics/'paper_fixed_K07/document_topic.csv',
               fresh_topics/'paper_fixed_K07/case_system_probabilities.csv'],
              [graph_out/'analysis_summary.json',graph_out/'base_graph.json',graph_out/'enhanced_graph.json',
               graph_out/'enhanced_propagation_candidate_paths.jsonl'])
            for name,script in [('patterns','diesel_kg.graph.patterns'),('hub_hierarchy','diesel_kg.graph.hubs')]:
                runner.run(f'fresh_{name}_{variant}',script,
                  ['--graph-dir',graph_out,'--cases',inputs/'cases_all.jsonl','--output',graph_out/name],
                  [graph_out/'base_graph.json',graph_out/'enhanced_graph.json',inputs/'cases_all.jsonl'],
                  [graph_out/name/'analysis_summary.json'])
            runner.run(f'fresh_engineering_{variant}','diesel_kg.evaluation.engineering',
              ['--graph-dir',graph_out,'--cases',inputs/'cases_all.jsonl','--output',graph_out/'engineering','--repeats',3],
              [graph_out/'base_graph.json',graph_out/'enhanced_graph.json',inputs/'cases_all.jsonl'],
              [graph_out/'engineering/analysis_summary.json',graph_out/'engineering/case_queries_and_inference.jsonl',
               graph_out/'engineering/structural_descriptive_metrics.json'])
            runner.run(f'fresh_graph_modes_verification_{variant}','diesel_kg.validation.graph_modes',
              ['--graph-dir',graph_out,'--cases',inputs/'cases_all.jsonl','--patterns-dir',graph_out/'patterns',
               '--hub-dir',graph_out/'hub_hierarchy','--output',graph_out/'graph_modes_verification.json'],
              [graph_out/'enhanced_graph.json',graph_out/'patterns/enhanced_pattern_instances.jsonl',
               graph_out/'hub_hierarchy/analysis_summary.json',inputs/'cases_all.jsonl'],
              [graph_out/'graph_modes_verification.json'])
        else:
            write(a.output/f'fresh_graph_{variant}/analysis_summary.json',
                  {'status':'not_evaluable_empty_qualified_model_graph','qualified_graph_edges':0,
                   'historical_export_inserted':False,'upstream_summary':str(out/'run_summary.json')})
        # The paper does not specify the added 0.8 confidence gate. Preserve
        # that result and execute the saved-argmax comparison independently.
        threshold_out=a.output/f'relation_threshold_{variant}'
        relmap=re_root/'re_locator_fixed/best_re_model/rel2id.json'
        raw=out/'all_occurrence_pair_predictions.jsonl.gz'
        threshold_artifacts=[threshold_out/name for name in (
            'analysis_summary.json','run_config.json','threshold_comparison.csv',
            'case_counts_and_rejections.jsonl','pair_filter_audit.jsonl.gz',
            'threshold0_qualified_predicted_graph.json','threshold0_qualified_triples.jsonl')]
        runner.run(f'RE_threshold_sensitivity_{variant}','diesel_kg.evaluation.thresholds',
          ['--raw-predictions',raw,'--entities',entities,'--rel2id',relmap,'--output',threshold_out,
           '--thresholds',0,.2,.4,.6,.8,'--min-entity-length',2,'--export-argmax-graph'],
          [raw,entities,relmap,out/'run_summary.json',resolve_script('diesel_kg.models.loading', HERE)],threshold_artifacts)
        argmax_out=threshold_out/'argmax_graph_analysis'
        argmax_artifacts=[argmax_out/'execution.json',argmax_out/'analysis_summary.json']
        if read(threshold_out/'analysis_summary.json')['thresholds']['0.0']['n_typed_canonical_edges']:
            argmax_artifacts.extend(argmax_out/name for name in (
                'run_config.json','base_graph.json','enhanced_graph.json',
                'base_case_path_completion.csv','enhanced_case_path_completion.csv',
                'patterns/analysis_summary.json','hub_hierarchy/analysis_summary.json',
                'engineering/analysis_summary.json','engineering/run_config.json',
                'engineering/case_queries_and_inference.jsonl',
                'graph_modes_verification.json','argmax_evidence_verification.json'))
        runner.run(f'argmax_graph_comparison_{variant}','diesel_kg.pipelines.argmax',
          ['--threshold-dir',threshold_out,'--topics-dir',fresh_topics,
           '--cases',inputs/'cases_all.jsonl','--output',argmax_out],
          [*threshold_artifacts,raw,entities,relmap,out/'run_summary.json',
           out/'qualified_predicted_graph.json',out/'qualified_triples.jsonl',
           inputs/'cases_all.jsonl',fresh_topics/'analysis_summary.json',
           fresh_topics/'paper_fixed_K07/document_topic.csv',
           fresh_topics/'paper_fixed_K07/case_system_probabilities.csv'],argmax_artifacts)
    write(a.output/'model_references.json',references)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=HERE/'data/kg_build')
    p.add_argument('--paper-source',type=Path,default=HERE/'data/paper/paper2.tex')
    p.add_argument('--assets-dir',type=Path,default=HERE/'assets')
    p.add_argument('--output',type=Path,default=HERE/'results/experiments')
    p.add_argument('--recorded-results',type=Path,default=HERE/'results/experiments/recorded_RE_training')
    p.add_argument('--topic-entities',type=Path,default=HERE/'data/topic_dictionary_entities.jsonl')
    p.add_argument('--phase',choices=['prepare','analyses','inference','all'],default='all')
    p.add_argument('--ner-mode',choices=['train','recorded'],default='train'); p.add_argument('--re-mode',choices=['train','recorded'],default='recorded')
    p.add_argument('--ner-results',type=Path); p.add_argument('--device',choices=['cpu','mps','cuda'],default='mps')
    p.add_argument('--variants',nargs='+',choices=NER_OPTIONS,default=list(NER_VARIANTS),help='Primary source and verified contiguous-weight repair; failed identity/eager arms remain diagnostic records')
    p.add_argument('--inference-device',choices=['cpu','mps','cuda'],default='cpu'); p.add_argument('--resume',action='store_true')
    p.add_argument('--bert-inference-device',choices=['cpu','mps','cuda'],help='Explicit device override for BERT repair branches, used only after their training has finished')
    p.add_argument('--gold',type=Path); p.add_argument('--expert-scores',type=Path); p.add_argument('--events',type=Path)
    p.add_argument('--split-manifest',type=Path); p.add_argument('--temporal-links',type=Path)
    p.add_argument('--physical-system-labels',type=Path)
    
    a=p.parse_args()
    for name,value in vars(a).items():
        if isinstance(value,Path): setattr(a,name,value.resolve())
    if HERE==a.output:
        p.error('Output cannot replace the project root')
    if HERE in a.output.parents and not a.output.is_relative_to(HERE/'results'):
        p.error('Project-local experiment outputs must be under results/')
    a.output.mkdir(parents=True,exist_ok=True); runner=Runner(a.output,a.resume)
    write(a.output/'execution_configuration.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
    if a.phase in ('prepare','all'): prepare(a,runner)
    if a.phase in ('analyses','all'): analyses(a,runner)
    if a.phase in ('inference','all'): models_and_inference(a,runner)
    if a.phase in ('inference','all') and all((a.output/f'occurrence_{v}/re_locator_fixed/run_summary.json').exists() for v in NER_VARIANTS):
        with (a.output/'execution_logs/full_verification.log').open('w') as f:
            subprocess.run([sys.executable,str(resolve_script('diesel_kg.validation.framework', HERE)),'--output-root',str(a.output)],
                           stdout=f,stderr=subprocess.STDOUT,check=True)
    
    print(f'Phase {a.phase} finished. Actual stage evidence: {runner.path}',flush=True)


if __name__=='__main__': main()
