"""Render the completed fixed-split domain adaptation using saved metrics only."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from collections import Counter
from diesel_kg.evaluation.ner_domain import key, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adaptation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); root = args.adaptation
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((root/'summary.json').read_text())
    verification = json.loads((root/'verification.json').read_text())
    if summary['status'] != 'completed' or verification['status'] != 'passed':
        raise ValueError('Report only completed, verified actual experiments')
    history = json.loads((root/'epoch_history.json').read_text())
    comparisons = []
    for split, baseline, selected in [
        ('template_dev800',summary['baseline_dev']['template_dev'],summary['selected_dev']['template_dev']),
        ('domain_dev15',summary['baseline_dev']['domain_dev'],summary['selected_dev']['domain_dev']),
        ('domain_test15_once_after_selection',summary['domain_test_baseline'],summary['domain_test_selected'])]:
        comparisons.append({'split':split,'baseline_f1':baseline['f1'],'selected_f1':selected['f1'],
                            'delta_f1':selected['f1']-baseline['f1'],'gold':selected['gold'],
                            'baseline_predicted':baseline['predicted'],'selected_predicted':selected['predicted'],
                            'baseline_matched':baseline['matched'],'selected_matched':selected['matched']})
    with (output/'baseline_adaptation_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    per_type=[]
    for model in ('baseline','selected'):
        rows=[json.loads(line) for line in (root/('domain_test_'+model+'_predictions.jsonl')).read_text().splitlines() if line.strip()]
        gold={};predicted={}
        for r in rows:
            for e in r['gold_entities']:gold[(r['sample_id'],)+key(e)]=e['type']
            for e in r['predicted_entities']:predicted[(r['sample_id'],)+key(e)]=e['type']
        gc,pc,mc=Counter(gold.values()),Counter(predicted.values()),Counter(gold[k] for k in gold.keys() & predicted.keys())
        for kind in sorted(gc.keys() | pc.keys()):
            p=mc[kind]/pc[kind] if pc[kind] else 0.;rec=mc[kind]/gc[kind] if gc[kind] else 0.
            per_type.append({'model':model,'type':kind,'gold':gc[kind],'predicted':pc[kind],'matched':mc[kind],
                             'precision':p,'recall':rec,'f1':2*p*rec/(p+rec) if p+rec else 0.})
    with (output/'domain_test_per_type_descriptive.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_type[0]));writer.writeheader();writer.writerows(per_type)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    epochs=[r['epoch'] for r in history]
    for name,label,color in [('template_dev','Template dev800','#2d719b'),('domain_dev','Domain dev15','#da7b32')]:
        axes[0].plot(epochs,[r['dev'][name]['f1'] for r in history],marker='o',markersize=3,label=label,color=color)
    axes[0].axvline(summary['best_epoch'],color='#555',linestyle=':',label=f'Selected epoch {summary["best_epoch"]}')
    axes[0].set(xlabel='Classifier training epoch',ylabel='Strict span + type F1',ylim=(0,1.03))
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    xs=list(range(len(comparisons)));w=.36
    axes[1].bar([x-w/2 for x in xs],[r['baseline_f1'] for r in comparisons],w,label='Unchanged baseline',color='#9cacb8')
    bars=axes[1].bar([x+w/2 for x in xs],[r['selected_f1'] for r in comparisons],w,label='Selected adapted head',color='#2d719b')
    for bar,r in zip(bars,comparisons):axes[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+.025,f'{r["selected_f1"]:.3f}',ha='center',fontsize=8)
    axes[1].set(xticks=xs,xticklabels=['Template\ndev800','Domain\ndev15','Domain\ntest15'],ylabel='Strict span + type F1',ylim=(0,1.08))
    axes[1].legend(fontsize=8);axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('Post-hoc exploratory domain adaptation: frozen encoder, fixed grouped split',fontsize=11)
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(output/('domain_adaptation_comparison.'+ext),dpi=180)
    plt.close(fig)
    report={'status':'completed','summary_sha256':sha256(root/'summary.json'),'verification_sha256':sha256(root/'verification.json'),
            'script_sha256':sha256(__file__),'artifact_sha256':{name:sha256(output/name) for name in ('baseline_adaptation_comparison.csv','domain_test_per_type_descriptive.csv','domain_adaptation_comparison.png','domain_adaptation_comparison.svg')}}
    (output/'plot_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'completed','comparisons':comparisons},ensure_ascii=False),flush=True)


if __name__ == '__main__':main()
