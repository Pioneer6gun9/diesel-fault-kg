"""Run actual adapted-NER/group-heldout-RE candidate inference and train-only graphs.

Requires completed selected models in the same experiment as the frozen inputs.
No relation-existence, expert, diagnostic or temporal gold is made.
"""
from __future__ import annotations
from diesel_kg.paths import PROJECT_ROOT, resolve_script
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from diesel_kg.pipelines.argmax import dependencies
from diesel_kg.evaluation.phase import sha, dump

HERE = PROJECT_ROOT

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir',type=Path,default=HERE/'results/experiments')
    p.add_argument('--assets-dir',type=Path,required=True)
    p.add_argument('--re-checkpoint',type=Path)
    p.add_argument('--output',type=Path,default=HERE/'results/experiments/adapted_model_graph')
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.re_checkpoint is None:
        a.re_checkpoint=a.results_dir/'re_clean/grouped_training/best_model.pt'
    ner=a.results_dir/'ner_domain/adaptation';entities=ner/'corpus_entities.jsonl'
    summary=json.loads(entities.with_suffix('.summary.json').read_text())
    if summary['status']!='completed' or summary['counts']['cases']!=1522:raise ValueError('Complete adapted NER corpus required')
    if sha(entities)!=summary['output_sha256']:raise ValueError('NER corpus changed')
    re_execution=json.loads((a.results_dir/'re_clean/grouped_training/execution.json').read_text())
    if re_execution.get('status')!='completed':raise ValueError('Completed actual grouped RE training required')
    if sha(a.re_checkpoint)!=re_execution['checkpoint_sha256']:raise ValueError('RE checkpoint differs from completed grouped training')
    verify=json.loads((ner/'verification.json').read_text())
    if verify['status']!='passed':raise ValueError('Verified adapted NER required')
    cases=a.results_dir/'inputs/cases_all.jsonl';source=a.results_dir/'inputs/original_source/RelationExtraction.py'
    pretrained=a.assets_dir/'pretrained/bert-base-chinese'
    forced=ner/'domain_graph_exclusion_case_ids.json'
    bound=[cases,source,entities,entities.with_suffix('.summary.json'),a.re_checkpoint,forced,
           ner/'summary.json',ner/'verification.json',a.results_dir/'re_clean/grouped_training/execution.json']
    bound += [f for f in pretrained.iterdir() if f.is_file() and f.suffix in ('.json','.txt','.safetensors','.bin')]
    inputs={str(f.resolve()):sha(f) for f in bound}
    record={'status':'running','command':[sys.executable,*sys.argv],'started_unix':time.time(),
        'input_sha256':inputs,'stages':[],
        'scope':'exploratory_adapted_NER_and_groupheldout_RE_model_candidates_train_only_LDA_graph_silver_retrieval',
        'human_gold':False,'paper_numeric_claims_reproduced':False}
    save=lambda:dump(a.output/'execution.json',record);save()
    def run(name,script,args):
        log=a.output/(name+'.log');cmd=[sys.executable,str(resolve_script(script, HERE)),*map(str,args)]
        r={'stage':name,'command':cmd,'cwd':str(HERE),'code_sha256':dependencies(script),
           'status':'running','log':str(log.resolve())};record['stages'].append(r);save();beg=time.perf_counter()
        print(name,'executing',flush=True)
        with log.open('w') as f:result=subprocess.run(cmd,cwd=HERE,stdout=f,stderr=subprocess.STDOUT)
        r.update(status='completed' if result.returncode==0 else 'failed',exit_code=result.returncode,
                 elapsed_seconds=time.perf_counter()-beg,log_sha256=sha(log));save()
        if result.returncode:raise RuntimeError(name+' failed: '+str(log))
        if any(sha(f)!=v for f,v in r['code_sha256'].items()):raise ValueError('Code changed during actual stage')
    try:
        import torch
        saved=torch.load(a.re_checkpoint,map_location='cpu',weights_only=True,mmap=True)
        dump(a.output/'rel2id.json',saved['rel2id']);del saved
        if a.device!='cpu':
            run('RE_device_check','diesel_kg.evaluation.re_device',['--checkpoint',a.re_checkpoint,'--pretrained',pretrained,
                '--source',source,'--tokenizer',pretrained,'--entities',entities,'--device',a.device,'--output',a.output/'re_device_check.json'])
        re_output=a.output/'re_occurrences'
        run('RE_occurrences','diesel_kg.inference.relations',['--entities',entities,'--checkpoint',a.re_checkpoint,
            '--source',source,'--tokenizer',pretrained,'--pretrained',pretrained,'--device',a.device,'--output',re_output])
        threshold=a.output/'relation_threshold'
        run('threshold_sensitivity','diesel_kg.evaluation.thresholds',['--raw-predictions',re_output/'all_occurrence_pair_predictions.jsonl.gz',
            '--entities',entities,'--rel2id',a.output/'rel2id.json','--output',threshold,
            '--thresholds','0','.2','.4','.6','.8','--min-entity-length','2','--export-argmax-graph'])
        run('grouped_graph_ablation','diesel_kg.graph.ablation',['--cases',cases,'--entities',entities,
            '--triples',threshold/'threshold0_qualified_triples.jsonl','--candidate-graph',threshold/'threshold0_qualified_predicted_graph.json',
            '--forced-splits',forced,'--output',a.output/'graph_ablation'])
        run('case_local_chain_audit','diesel_kg.evaluation.chain_qualification',['--graph-ablation',a.output/'graph_ablation',
            '--output',a.output/'graph_ablation/case_role_qualification'])
        run('phase_direction_audit','diesel_kg.evaluation.phase',['--cases',cases,'--entities',entities,
            '--triples',threshold/'threshold0_qualified_triples.jsonl','--graph',threshold/'threshold0_qualified_predicted_graph.json',
            '--output',a.output/'phase_evidence'])
        run('physical_reference_audit','diesel_kg.evaluation.physical',['--graph',threshold/'threshold0_qualified_predicted_graph.json',
            '--output',a.output/'physical_references'])
        if any(sha(f)!=v for f,v in inputs.items()):raise ValueError('Input/model changed during pipeline')
        outputs={str(f.relative_to(a.output)):sha(f) for f in a.output.rglob('*') if f.is_file() and f!=a.output/'execution.json'}
        record.update(status='completed_actual_model_candidate_pipeline_not_gold_validation',finished_unix=time.time(),output_sha256=outputs);save()
        print(json.dumps({'status':record['status'],'actual_stages':len(record['stages'])}),flush=True)
    except Exception as e:
        record.update(status='failed',error=str(e),finished_unix=time.time());save();raise

if __name__=='__main__':main()
