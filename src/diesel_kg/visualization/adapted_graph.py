#!/usr/bin/env python3
"""Plot completed adapted-model ablations with nonnegative metric axes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from diesel_kg.graph.ablation import WEIGHT_MODES, sha, dump


def plot(graph_ablation, output):
    root = Path(graph_ablation).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = root / 'analysis_summary.json'
    data = json.loads(source.read_text())
    if data['status'] != 'completed_grouped_heldout_silver_retrieval_ablation':
        raise ValueError('Actual ablation run is incomplete')
    names = ['Unweighted', 'Semantic', 'Cooccurrence', 'Relation\nproxy', 'Fusion']
    x = np.arange(5)
    fields = [('silver_root_recall_at_3', 'Silver Recall@3 (%)', True, 100),
              ('silver_root_mrr', 'Silver MRR', True, 1),
              ('n_root_candidates', 'Candidate roots per query', False, 1),
              ('machine_query_ms', 'Local CPU query time (ms)', False, 1)]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.1))
    for ax, (field, label, silver, scale) in zip(axes.ravel(), fields):
        for offset, level, legend, color in [(-.18, 'no_mechanisms', 'Without mechanism nodes', '#4E79A7'),
                                              (.18, 'mechanisms', 'With mechanism nodes', '#E09643')]:
            values, low, high = [], [], []
            for mode in WEIGHT_MODES:
                test = data['retrieval'][mode + '__' + level]['test']
                metric = test['silver_reference_subset']['metrics'][field] if silver else test['metrics'][field]
                values.append(metric['mean'] * scale)
                low.append(max(0., (metric['mean'] - metric['ci95'][0]) * scale))
                high.append(max(0., (metric['ci95'][1] - metric['mean']) * scale))
            ax.bar(x + offset, values, width=.34, color=color, label=legend)
            ax.errorbar(x + offset, values, yerr=np.array([low, high]), fmt='none',
                        ecolor='#333333', capsize=3, linewidth=1)
        ax.set_xticks(x, names)
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        if not any(values):
            ax.text(.5,.55,'All measured values = 0',transform=ax.transAxes,ha='center',fontsize=11)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    first = data['retrieval']['base_unweighted__no_mechanisms']['test']
    fig.suptitle(f"Adapted-model grouped graph test: {first['n_case_groups']} body groups; "
                 f"{first['silver_reference_subset']['n_case_groups']} groups with silver references", fontsize=14)
    axes[0, 0].legend(loc='upper left', fontsize=9)
    fig.text(.5, .025, 'Bars: equal body-group means. Whiskers: body-group bootstrap 95% intervals.\n'
        'References are text/NER silver labels, not human causal gold. Frozen NER/RE training-source independence is unproven.\n'
        'Identical unweighted/relation-proxy graphs expose a constant proxy; timing differences include local run-order and CPU noise.',
        ha='center', fontsize=9, linespacing=1.5)
    fig.tight_layout(rect=[0, .095, 1, .94])
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(output / ('grouped_test_ablation.' + extension), dpi=180)
    plt.close(fig)
    manifest = {'status': 'generated_from_completed_saved_metrics', 'source': str(source),
        'source_sha256': sha(source), 'script_sha256': sha(__file__),
        'plots': {path.name: {'bytes': path.stat().st_size, 'sha256': sha(path)}
                  for path in sorted(output.glob('grouped_test_ablation.*'))},
        'human_gold_accuracy': None, 'engineer_localization_time': None,
        'timing_scope': 'same-process CPU retrieval; order fixed; concurrent system load uncontrolled'}
    dump(output / 'plot_manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph-ablation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.graph_ablation, args.output)
