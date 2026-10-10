"""Compare saved graph rankings against identical frozen old-NER silver targets.

Never reruns extraction/retrieval, changes a graph, or selects a model. New-model
silver targets are ignored. References are biased toward the old extraction and
are not human gold or verified causal roots. Empty-target cases are excluded,
while eligible target cases with zero candidates retain zero recall and MRR.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time

METRICS = ('recall_at_1','recall_at_3','recall_at_5','mrr',
           'candidate_target_recall','candidate_coverage','reference_nodes_in_graph_fraction')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def write_rows(path, rows):
    with Path(path).open('w') as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')


def ranking_metrics(ranked, targets, node_ids):
    if len(ranked) != len(set(ranked)):
        raise ValueError('Saved candidate ranking contains duplicate IDs')
    target_set = set(targets)
    if not target_set:
        return {k:None for k in METRICS}
    ranks = [i+1 for i,n in enumerate(ranked) if n in target_set]
    return {**{f'recall_at_{k}':len(set(ranked[:k]) & target_set)/len(target_set) for k in (1,3,5)},
            'mrr':1/min(ranks) if ranks else 0.,
            'candidate_target_recall':len(set(ranked) & target_set)/len(target_set),
            'candidate_coverage':float(bool(ranked)),
            'reference_nodes_in_graph_fraction':len(target_set & set(node_ids))/len(target_set)}


def freeze_references(rows):
    result = []
    for r in rows:
        refs = r['silver_reference']['references']
        if r['split'] != 'test' or not refs:
            continue
        if any(x.get('is_human_gold') is not False or x.get('is_verified_causal_root') is not False for x in refs):
            raise ValueError('This comparison requires explicitly non-gold silver references')
        result.append({'case_id':r['case_id'],'group_id':r['group_id'],'split':'test',
                       'query_text':r['query_text'],'boundary_span':r['boundary_span'],
                       'boundary_quote':r['boundary_quote'],
                       'query_boundary_available':r['query_boundary_available'],
                       'target_ids':sorted({x['node_id'] for x in refs}),
                       'frozen_silver_reference':r['silver_reference']})
    if len({r['case_id'] for r in result}) != len(result):
        raise ValueError('Duplicate eligible reference case identity')
    return result


def group_means(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['group_id']].append(r)
    output = []
    for gid, records in sorted(grouped.items()):
        item = {'group_id':gid,'case_ids':sorted(r['case_id'] for r in records),'numbered_cases':len(records)}
        for side in ('baseline','new'):
            item[side] = {k:sum(r[side][k] for r in records)/len(records) for k in METRICS}
        item['paired_delta'] = {k:item['new'][k]-item['baseline'][k] for k in METRICS}
        output.append(item)
    return output


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered)-1)*probability
    lo = int(position); hi = min(lo+1,len(ordered)-1)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)


def paired_bootstrap(groups, n_boot=2000, seed=42):
    if not groups:
        raise ValueError('No paired eligible body groups')
    n = len(groups);rng=random.Random(seed)
    arrays = {side:{k:[] for k in METRICS} for side in ('baseline','new','paired_delta')}
    # Identical group draw for all branches/metrics preserves their pairing.
    for _ in range(n_boot):
        drawn=[groups[rng.randrange(n)] for _ in range(n)]
        for side in arrays:
            for k in METRICS:
                arrays[side][k].append(sum(r[side][k] for r in drawn)/n)
    return {side:{k:{'mean':sum(r[side][k] for r in groups)/n,
                       'ci95':[quantile(arrays[side][k],.025),quantile(arrays[side][k],.975)],
                       'n_groups':n} for k in METRICS} for side in arrays}


def compare_variant(refs, before, after, old_nodes, new_nodes):
    old = {r['case_id']:r for r in before}; new = {r['case_id']:r for r in after}
    if len(old)!=len(before) or len(new)!=len(after) or set(old)!=set(new):
        raise ValueError('Unpaired saved query cases between model branches')
    for cid in old:
        for field in ('group_id','split','query_text','boundary_span','boundary_quote','query_boundary_available'):
            if old[cid][field] != new[cid][field]:
                raise ValueError(f'Different heldout query/split/diagnosis boundary: {cid}/{field}')
    for node_id in old_nodes.keys() & new_nodes.keys():
        if (old_nodes[node_id]['name'],old_nodes[node_id]['entity_type']) != (new_nodes[node_id]['name'],new_nodes[node_id]['entity_type']):
            raise ValueError('Cross-model node ID identity collision')
    result=[]
    for ref in refs:
        a,b=old[ref['case_id']],new[ref['case_id']]
        if a['group_id']!=ref['group_id'] or a['query_text']!=ref['query_text'] or a['split']!='test':
            raise ValueError('Frozen reference does not bind to saved query identity')
        if sorted({r['node_id'] for r in a['silver_reference']['references']}) != ref['target_ids']:
            raise ValueError('Original silver target set differs across old ablations')
        for query,nodes in ((a,old_nodes),(b,new_nodes)):
            if set(query['ranked_candidate_ids'])-nodes.keys():
                raise ValueError('Saved candidate ID missing from corresponding graph')
            if query['n_root_candidates'] != len(query['ranked_candidate_ids']):
                raise ValueError('Saved ranking length does not match candidate count')
        result.append({'case_id':ref['case_id'],'group_id':ref['group_id'],'target_ids':ref['target_ids'],
                       'query_text':ref['query_text'],'baseline_ranked_ids':a['ranked_candidate_ids'],
                       'new_ranked_ids':b['ranked_candidate_ids'],
                       'baseline':ranking_metrics(a['ranked_candidate_ids'],ref['target_ids'],old_nodes),
                       'new':ranking_metrics(b['ranked_candidate_ids'],ref['target_ids'],new_nodes)})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-root',type=Path,required=True)
    parser.add_argument('--new-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();started=time.monotonic();args.output.mkdir(parents=True,exist_ok=True)
    reference_source=args.old_root/'base_unweighted__no_mechanisms'/'heldout_queries.jsonl'
    original=read_rows(reference_source);refs=freeze_references(original)
    groups=sorted({r['group_id'] for r in refs})
    if len(groups)!=69 or len(refs)!=69:
        raise ValueError('Expected the frozen 69 old-silver test case/body groups')
    write_rows(args.output/'fixed_old_silver_test_references.jsonl',refs)
    scope={'reference_source_path':str(reference_source),'reference_source_sha256':digest(reference_source),
           'fixed_reference_file':str(args.output/'fixed_old_silver_test_references.jsonl'),
           'fixed_reference_sha256':digest(args.output/'fixed_old_silver_test_references.jsonl'),
           'numbered_test_cases':len(refs),'unique_test_body_groups':len(groups),'group_ids':groups,
           'all_original_test_numbered_cases':sum(r['split']=='test' for r in original),
           'empty_target_test_cases_excluded':sum(r['split']=='test' and not r['silver_reference']['references'] for r in original),
           'targets':sum(len(r['target_ids']) for r in refs),'target_matching':'Exact canonical typed node ID; no fuzzy/name-only/substring substitution.',
           'reference_origin':'Old frozen corrected-NER occurrences within the disclosed diagnosis-before-intervention text window.',
           'bias':'References originate from the old extraction branch and favor its entity boundaries/types. This comparison measures retrieval of these fixed old-silver targets, not unbiased new extraction quality.',
           'is_human_gold':False,'is_verified_causal_or_industrial_accuracy':False,
           'empty_target_scoring':'Exclude cases without old-silver targets; cases with targets but zero candidates retain zero Recall/MRR.',
           'new_branch_silver_references_used_as_targets':False}
    write_json(args.output/'fixed_reference_scope.json',scope)
    if args.prepare_only:
        print(json.dumps({'status':'fixed_reference_prepared','groups':len(groups),'cases':len(refs)},ensure_ascii=False),flush=True)
        return
    inputs={str(reference_source):digest(reference_source)}
    for root in (args.old_root,args.new_root):
        for name in ('analysis_summary.json','result_checks.json','case_group_split.jsonl'):
            p=root/name;inputs[str(p)]=digest(p)
        if not json.loads((root/'analysis_summary.json').read_text())['status'].startswith('completed') or json.loads((root/'result_checks.json').read_text())['status']!='passed':
            raise ValueError('Both graph branches must have actually completed and passed producer checks')
    if read_rows(args.old_root/'case_group_split.jsonl') != read_rows(args.new_root/'case_group_split.jsonl'):
        raise ValueError('Complete case-body split manifests differ across branches')
    variants=sorted(p.parent.name for p in args.old_root.glob('*/heldout_queries.jsonl'))
    new_variants=sorted(p.parent.name for p in args.new_root.glob('*/heldout_queries.jsonl'))
    if len(variants)!=10 or variants!=new_variants:
        raise ValueError('Require the same ten completed graph ablations')
    comparisons={};csv_rows=[];all_case_metrics=[];all_group_metrics=[]
    for variant in variants:
        paths={}
        data={}
        nodes={}
        for side,root in (('baseline',args.old_root),('new',args.new_root)):
            query=root/variant/'heldout_queries.jsonl';graph=root/variant/'graph.json'
            inputs[str(query)]=digest(query);inputs[str(graph)]=digest(graph)
            data[side]=read_rows(query);g=json.loads(graph.read_text())
            nodes[side]={n['id']:n for n in g['nodes']}
            if len(nodes[side])!=len(g['nodes']):raise ValueError('Duplicate node ID in graph')
            paths[side]={'ranking_path':str(query),'ranking_sha256':inputs[str(query)],
                         'graph_path':str(graph),'graph_sha256':inputs[str(graph)]}
        case_metrics=compare_variant(refs,data['baseline'],data['new'],nodes['baseline'],nodes['new'])
        grouped=group_means(case_metrics);stats=paired_bootstrap(grouped)
        comparisons[variant]={**stats,'ranking_sources':paths,'paired_numbered_cases':len(case_metrics),'paired_body_groups':len(grouped)}
        all_case_metrics.extend({'variant':variant,**r} for r in case_metrics)
        all_group_metrics.extend({'variant':variant,**r} for r in grouped)
        for k in METRICS:
            csv_rows.append({'variant':variant,'metric':k,'baseline_mean':stats['baseline'][k]['mean'],
                             'new_mean':stats['new'][k]['mean'],'mean_paired_delta':stats['paired_delta'][k]['mean'],
                             'paired_delta_ci95_low':stats['paired_delta'][k]['ci95'][0],
                             'paired_delta_ci95_high':stats['paired_delta'][k]['ci95'][1],
                             'paired_groups':len(grouped),'paired_numbered_cases':len(case_metrics)})
    write_rows(args.output/'fixed_reference_paired_case_metrics.jsonl',all_case_metrics)
    write_rows(args.output/'fixed_reference_paired_group_metrics.jsonl',all_group_metrics)
    with (args.output/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    if any(digest(p)!=h for p,h in inputs.items()):raise ValueError('Frozen ranking/graph/reference inputs changed during comparison')
    definitions={'recall_at_k':'Fraction of distinct frozen old-silver typed target IDs present in topk candidates.',
                 'mrr':'Reciprocal rank of first matching fixed target; zero when absent.',
                 'candidate_target_recall':'Fraction of fixed target IDs anywhere in complete saved candidate list.',
                 'candidate_coverage':'Fraction of fixed-reference queries returning any candidates, regardless of correctness.',
                 'reference_nodes_in_graph_fraction':'Fraction of fixed old-silver target IDs existing in each training graph.'}
    summary={'status':'completed','fixed_reference_scope':scope,'comparisons':comparisons,
             'metric_definitions':definitions,'bootstrap':{'samples':2000,'seed':42,'ci95':'Percentile, linear interpolation.',
             'resampling_unit':'Unique normalized case-body group; average numbered cases inside a group first; both models retain the same sampled groups.',
             'multiple_comparison_adjustment':'None; ten fixed exploratory comparisons.'},
             'input_sha256':inputs,'execution':{'argv':sys.argv,'python':sys.executable,'script_sha256':digest(__file__),
             'elapsed_seconds':time.monotonic()-started},
             'scope':'Saved rankings only; original graph/model metrics retained. Identical old-silver references do not establish human diagnostic accuracy or causal truth.'}
    write_json(args.output/'summary.json',summary)
    print(json.dumps({'status':'completed','variants':len(comparisons),'fixed_test_groups':len(groups),'paired_metric_rows':len(csv_rows)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
