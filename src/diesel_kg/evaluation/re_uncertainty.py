"""Saved-prediction coverage-conditioned template cluster uncertainty audit.

No model call, fitting or new label. A single heldout template per class cannot
estimate across-template variance. Unstratified template resampling can omit
classes and downward-bias fixed-nine-class macro F1; it remains diagnostic.
"""
from diesel_kg.paths import module_path, resolve_recorded_path
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
from diesel_kg.training.grouped_re import sha, dump, read_rows, normalize, metric_arrays


def cluster_audit(rows,labels,reps=2000):
    mapping={label:i for i,label in enumerate(labels)}
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[row.get('_template_group') or 'raw:'+normalize(row['text'])].append(i)
    strata=defaultdict(list);mixed=[]
    for key,indices in groups.items():
        gold={rows[i]['relation'] for i in indices}
        if len(gold)>1:mixed.append({'template':key,'gold_relations':sorted(gold),'n':len(indices)})
        for label in gold:strata[label].append(key)
    counts={label:len(keys) for label,keys in strata.items()}
    result={'n':len(rows),'inferred_clusters':len(groups),'clusters_containing_each_gold_class':counts,'mixed_gold_class_clusters':mixed,
            'group_identity_limit':'Inferred masked sentence, not verified original document/case source.',
            'unstratified_macro_ci_warning':'Resampling templates without preserving gold class coverage can omit whole classes; fixed-nine-class macro then receives artificial zero classes and is downward biased. Do not treat that naive interval as a cross-template confidence statement.'}
    if any(n<2 for n in counts.values()):
        result.update({'coverage_conditioned_cluster_bootstrap_ci95':None,'reason':'At least one supported gold class has only one template group. Across-template uncertainty within that class is not identifiable from these heldout groups.',
                       'sample_bootstrap_limit':'The saved iid sample interval is conditional on these specific templates and annotations and ignores template dependence.'})
        return result
    y=np.array([mapping[row['relation']] for row in rows]);p=np.array([mapping[row['predicted_relation']] for row in rows])
    rng=np.random.default_rng(44);values=[];keys=sorted(groups);required=set(y.tolist());rejected=0;attempts=0
    while len(values)<reps:
        attempts+=1
        if attempts>100*reps:raise RuntimeError('Insufficient coverage-complete cluster draws; do not invent a confidence interval.')
        indices=np.concatenate([groups[keys[i]] for i in rng.integers(0,len(keys),len(keys))])
        if set(y[indices].tolist())!=required:rejected+=1;continue
        _,acc,f1=metric_arrays(y[indices],p[indices],len(labels));values.append([acc,f1])
    lo,hi=np.quantile(values,[.025,.975],axis=0)
    result['coverage_conditioned_cluster_bootstrap_ci95']={'accuracy':[float(lo[0]),float(hi[0])],'macro_f1_all_nine_classes':[float(lo[1]),float(hi[1])],
        'accepted_resamples':reps,'rejected_missing_class_draws':rejected,'seed':44,'method':'Resample intact inferred template clusters with replacement; retain only draws containing every originally supported gold class. Mixed-class clusters remain intact. All nine macro labels fixed. Exploratory interval conditional on class coverage, inferred groups and supplied annotations; not a source-population generalization interval.'}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output;report=json.loads((out/'clean_evaluation.json').read_text());labels=report['labels']
    row_audit=json.loads((out/'row_audit.json').read_text())['test'];result={'execution':{'sys_argv':sys.argv,'script_sha256':sha(Path(__file__)),
        'evaluation_module_sha256':sha(module_path('diesel_kg.training.grouped_re'))},'no_new_model_forward':True,'runs':{}}
    for variant in ('original_re','re_locator_fixed'):
        ref=report['legacy_checkpoint_references'][variant];path=resolve_recorded_path(ref['predictions_path']);assert sha(path)==ref['predictions_sha256']
        rows=read_rows(path)
        for row in rows:row['_template_group']=row_audit[row['original_index']]['inferred_template']
        clean=set(report['existing_prediction_metrics'][variant]['strict_unique_nonoverlap_no_prior_text_or_query_no_test_duplicate']['original_indices'])
        result['runs'][variant+'_clean211']=cluster_audit([r for r in rows if r['original_index'] in clean],labels)
        result['runs'][variant+'_clean211']['predictions_sha256']=sha(path)
    summary=json.loads((out/'grouped_training/summary.json').read_text())
    for name in ('group_heldout','original225_previously_evaluated'):
        path=out/'grouped_training'/f'{name}_predictions.jsonl';assert sha(path)==summary['test'][name]['predictions_sha256']
        result['runs']['grouped_training_'+name]=cluster_audit(read_rows(path),labels)
        result['runs']['grouped_training_'+name]['predictions_sha256']=sha(path)
    manifest=json.loads((out/'split_manifest.json').read_text());expected={x['template']:x['relation'] for x in manifest['group_assignments']}
    flags=[{'original_index':x['original_index'],'inferred_template':x['inferred_template'],'source_gold_relation':x['relation'],'observed_train_dev_template_relation':expected[x['inferred_template']]} for x in row_audit if x['inferred_template'] in expected and x['relation']!=expected[x['inferred_template']]]
    result['test_template_label_consistency_review']={'flag_count':len(flags),'flags':flags,'scope':'Weak inferred-template agreement with observed train/dev labels. A flag is not independent proof that the original label is wrong; no gold, prediction or primary metric is changed.'}
    dump(out/'uncertainty_audit.json',result);print(json.dumps({k:v.get('coverage_conditioned_cluster_bootstrap_ci95') for k,v in result['runs'].items()}),flush=True)


if __name__=='__main__':main()
