"""Independent saved-ranking recount and paired bootstrap verification."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
from diesel_kg.paths import resolve_recorded_path


def read_rows(path):
    return [json.loads(l) for l in resolve_recorded_path(path).read_text().splitlines() if l.strip()]


def sha(path):
    return hashlib.sha256(resolve_recorded_path(path).read_bytes()).hexdigest()


def independent_metrics(query, targets, graph_ids):
    ranking=query['ranked_candidate_ids'];gold=set(targets)
    assert gold and len(ranking)==len(set(ranking))
    result={}
    for cutoff in (1,3,5):
        result['recall_at_'+str(cutoff)]=sum(t in ranking[:cutoff] for t in gold)/len(gold)
    first=next((i for i,n in enumerate(ranking,1) if n in gold),None)
    result['mrr']=0. if first is None else 1./first
    result['candidate_target_recall']=sum(t in ranking for t in gold)/len(gold)
    result['candidate_coverage']=float(len(ranking)>0)
    result['reference_nodes_in_graph_fraction']=sum(t in graph_ids for t in gold)/len(gold)
    return result


def near(a,b):
    return abs(a-b)<1e-12


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,required=True)
    args=parser.parse_args();root=args.comparison
    summary=json.loads((root/'summary.json').read_text());scope=summary['fixed_reference_scope']
    checks={}
    checks['actual_comparison_completed']=summary['status']=='completed'
    checks['all_original_inputs_sha_unchanged']=all(sha(p)==h for p,h in summary['input_sha256'].items())
    refs=read_rows(root/'fixed_old_silver_test_references.jsonl')
    original=read_rows(scope['reference_source_path'])
    original_targets={r['case_id']:sorted({e['node_id'] for e in r['silver_reference']['references']}) for r in original if r['split']=='test' and r['silver_reference']['references']}
    checks['fixed_reference_file_sha']=sha(root/'fixed_old_silver_test_references.jsonl')==scope['fixed_reference_sha256']
    checks['exact_original69_test_targets_retained']=len(refs)==len({r['group_id'] for r in refs})==69 and {r['case_id']:r['target_ids'] for r in refs}==original_targets
    checks['empty_targets_excluded_and_bias_disclosed']=scope['empty_target_test_cases_excluded']==156 and not scope['is_human_gold'] and not scope['new_branch_silver_references_used_as_targets'] and bool(scope['bias'])
    case_output=read_rows(root/'fixed_reference_paired_case_metrics.jsonl')
    group_output=read_rows(root/'fixed_reference_paired_group_metrics.jsonl')
    case_lookup={(r['variant'],r['case_id']):r for r in case_output}
    group_lookup={(r['variant'],r['group_id']):r for r in group_output}
    checks['all_ten_variants_have_complete69_pair_outputs']=len(summary['comparisons'])==10 and len(case_lookup)==len(case_output)==690 and len(group_lookup)==len(group_output)==690
    cases_correct=True;groups_correct=True;means_correct=True;intervals_correct=True;queries_correct=True
    metric_names=None
    for variant,comparison in summary['comparisons'].items():
        queries={};graphs={}
        for side in ('baseline','new'):
            source=comparison['ranking_sources'][side]
            queries[side]={r['case_id']:r for r in read_rows(source['ranking_path'])}
            graphs[side]={n['id'] for n in json.loads(resolve_recorded_path(source['graph_path']).read_text())['nodes']}
        by_group=defaultdict(lambda:defaultdict(list))
        for ref in refs:
            cid,gid=ref['case_id'],ref['group_id']
            a,b=queries['baseline'][cid],queries['new'][cid]
            queries_correct &= all(a[k]==b[k] for k in ('case_id','group_id','split','query_text','boundary_span','boundary_quote'))
            for side in ('baseline','new'):
                expected=independent_metrics(queries[side][cid],ref['target_ids'],graphs[side])
                metric_names=list(expected)
                cases_correct &= all(near(case_lookup[variant,cid][side][k],v) for k,v in expected.items())
                by_group[gid][side].append(expected)
        ordered=[]
        for gid in sorted(by_group):
            item={}
            for side in ('baseline','new'):
                item[side]={k:sum(r[k] for r in by_group[gid][side])/len(by_group[gid][side]) for k in metric_names}
            item['paired_delta']={k:item['new'][k]-item['baseline'][k] for k in metric_names}
            groups_correct &= all(near(group_lookup[variant,gid][side][k],item[side][k]) for side in item for k in metric_names)
            ordered.append(item)
        n=len(ordered)
        for side in ('baseline','new','paired_delta'):
            for k in metric_names:
                means_correct &= near(sum(r[side][k] for r in ordered)/n,comparison[side][k]['mean'])
                means_correct &= comparison[side][k]['n_groups']==n==69
        rng=random.Random(42)
        distributions={side:{k:[] for k in metric_names} for side in ('baseline','new','paired_delta')}
        # Recompute as a histogram-weighted dot product, distinct from the
        # producer's list-of-drawn-record mean; same RNG supports exact CI check.
        for _ in range(2000):
            multiplicity=Counter(rng.randrange(n) for _ in range(n))
            for side in distributions:
                for k in metric_names:
                    distributions[side][k].append(sum(ordered[i][side][k]*count for i,count in multiplicity.items())/n)
        for side in distributions:
            for k,values in distributions[side].items():
                values.sort()
                quantiles=[]
                for p in (.025,.975):
                    index=(len(values)-1)*p;lower=int(index);upper=min(lower+1,len(values)-1)
                    quantiles.append(values[lower]*(upper-index)+values[upper]*(index-lower) if upper!=lower else values[lower])
                intervals_correct &= all(near(a,b) for a,b in zip(quantiles,comparison[side][k]['ci95']))
    checks.update({'all_saved_query_identities_equal':queries_correct,
                   'every_case_metric_independently_recounted_from_source_ranking':cases_correct,
                   'every_body_group_average_independently_recomputed':groups_correct,
                   'all_baseline_new_paired_difference_means_and_counts_recomputed':means_correct,
                   'all2000_paired_group_bootstrap_CI_independently_recomputed':intervals_correct})
    result={'status':'passed' if all(checks.values()) else 'failed','checks':checks,'n_passed':sum(checks.values()),
            'n_checks':len(checks),'summary_sha256':sha(root/'summary.json'),
            'execution':{'argv':sys.argv,'python':sys.executable,'script_sha256':sha(__file__)},
            'scope':'Read-only source rankings and fixed old-silver targets; no graph/model rerun or causal/human validation.'}
    (root/'independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
