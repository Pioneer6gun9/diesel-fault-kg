"""Build the manuscript's color figures and aligned tables from measured results.

Publisher styles, conceptual illustrations, authors and non-table prose come
from data/manuscript. No online archive or external workspace is required.
"""
from __future__ import annotations
import argparse
import collections
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unicodedata
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from diesel_kg.paths import PROJECT_ROOT, source_dependencies, resolve_recorded_path


def build(results_dir: Path, output: Path, compile_pdf: bool = False):
    EXP, OUT = Path(results_dir).resolve(), Path(output).resolve()
    if EXP.is_relative_to(PROJECT_ROOT / 'results/simulated'):
        raise ValueError('Manuscript measurements must not use SIMULATED inputs')
    TEMPLATE = PROJECT_ROOT / 'data/manuscript'
    TEX, FIG = OUT / 'overleaf', OUT / 'overleaf/figures'
    FIG.mkdir(parents=True, exist_ok=True)
    used_files = set()
    for name in ['main.rty', 'maik.rty', 'maik.bst', 'ref.bib',
                 'figures/framework.png']:
        source = TEMPLATE / name
        used_files.add(source)
        shutil.copy2(source, TEX / name)
    source = PROJECT_ROOT / 'data/paper/LDA模型.png'
    used_files.add(source)
    shutil.copy2(source, FIG / 'lda_illustration.png')
    template = TEMPLATE / 'main.tex'
    used_files.add(template)
    original = template.read_text(encoding='utf-8-sig')
    def load(name):
        path = EXP / name
        used_files.add(path)
        return json.loads(path.read_text(encoding='utf-8-sig'))


    def rows(name):
        path = EXP / name
        used_files.add(path)
        return list(csv.DictReader(path.open(encoding='utf-8-sig')))


    def predictions(name):
        path = EXP / name
        used_files.add(path)
        return [json.loads(line) for line in path.open() if line.strip()]


    def sha(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()


    def span_metrics(part):
        gold, predicted = set(), set()
        for i, row in enumerate(part):
            for entity in row['gold_entities']:
                gold.add((i, entity['start'], entity['end'], entity['type']))
            for entity in row['predicted_entities']:
                predicted.add((i, entity['start'], entity['end'], entity['type']))
        matched = len(gold & predicted)
        return dict(n=len(part), gold=len(gold), predicted=len(predicted), matched=matched,
                    precision=matched / len(predicted), recall=matched / len(gold),
                    f1=2 * matched / (len(gold) + len(predicted)))


    def percentage(value):
        return f'{100 * value:.2f}'


    def save_plot(name):
        plt.savefig(FIG / name, bbox_inches='tight', pad_inches=0.08, facecolor='white')
        plt.close()


    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .65, 'axes.labelsize': 10.5,
                         'axes.titlesize': 10.5, 'axes.titlepad': 12,
                         'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
                         'legend.fontsize': 9.5, 'axes.axisbelow': True,
                         'grid.color': '#dbe2e7', 'grid.linewidth': .6,
                         'pdf.fonttype': 42, 'savefig.dpi': 300})
    # Retain the visual conventions of the manuscript's original color figures.
    BASE_COLOR = '#1f77b4'
    ENHANCED_COLOR = '#ff7f0e'
    WITHIN_COLOR = '#e74c3c'
    BETWEEN_COLOR = '#3498db'

    cases = predictions('inputs/cases_all.jsonl')
    assert len(cases) == 1522
    body_keys = [''.join(unicodedata.normalize('NFKC', x['text']).split()) for x in cases]
    assert len(set(body_keys)) == 1360

    ner = {}
    for variant in ['source_roberta', 'bert_identity_repair_rebuilt']:
        pred = predictions(f'ner/{variant}/test_predictions.jsonl')
        summary = load(f'ner/{variant}/summary.json')
        ner[variant] = {name: span_metrics(part) for name, part in
                        [('all', pred), ('template', pred[:800]), ('domain', pred[800:])]}
        assert math.isclose(ner[variant]['all']['f1'], summary['test']['exact_span_type']['f1'])
        checkpoint = resolve_recorded_path(EXP / f'ner/{variant}' / Path(summary['checkpoint_path']).name)
        assert sha(checkpoint) == summary['checkpoint_sha256']
        used_files.add(checkpoint)

    adapt = load('ner_domain/adaptation/summary.json')
    adapt_metrics = {}
    for model in ['baseline', 'selected']:
        adapt_metrics[model] = span_metrics(predictions(f'ner_domain/adaptation/domain_test_{model}_predictions.jsonl'))
        assert math.isclose(adapt_metrics[model]['f1'], adapt[f'domain_test_{model}']['f1'])
    corpus_ner = load('ner_domain/adaptation/corpus_entities.summary.json')

    re_summary = load('re_clean/grouped_training/summary.json')
    ner_config = load('ner/bert_identity_repair_rebuilt/config.json')
    re_config = load('re_clean/grouped_training/config.json')
    re_test = re_summary['test']['group_heldout']
    re_pred = predictions('re_clean/grouped_training/group_heldout_predictions.jsonl')
    classes = re_test['confusion_matrix_order']
    cm = np.zeros((9, 9), dtype=int)
    for row in re_pred:
        cm[classes.index(row['relation']), classes.index(row['predicted_relation'])] += 1
    assert cm.tolist() == re_test['confusion_matrix']
    assert int(np.trace(cm)) == re_test['correct']
    assert math.isclose(float(np.mean(2 * np.diag(cm) / (cm.sum(0) + cm.sum(1)))), re_test['macro_f1_all_nine_classes'])
    assert sha(EXP / 're_clean/grouped_training/group_heldout_predictions.jsonl') == re_test['predictions_sha256']
    checkpoint = resolve_recorded_path(EXP / 're_clean/grouped_training/best_model.pt')
    assert sha(checkpoint) == re_summary['checkpoint_sha256']
    used_files.add(checkpoint)
    re_split = load('re_clean/split_manifest.json')
    overall = {'threshold_sensitivity': load('adapted_model_graph/relation_threshold/analysis_summary.json')['thresholds']}
    topics = load('topics/analysis_summary.json')
    curve = rows('topics/topic_coherence_curve.csv')
    best = max(curve, key=lambda row: float(row['coherence_c_v']))
    assert int(best['k']) == topics['best_k'] == 10
    graph_summary = load('graph_k7/analysis_summary.json')
    base_graph = load('graph_k7/base_graph.json')
    graph = load('graph_k7/enhanced_graph.json')
    assert len(base_graph['nodes']) == graph_summary['base_full_KG']['nodes']
    assert len(base_graph['edges']) == graph_summary['base_full_KG']['typed_edges']
    assert len(graph['nodes']) == graph_summary['enhanced_full_KG']['nodes']
    assert len(graph['basic_edges']) + len(graph['mechanism_edges']) == graph_summary['enhanced_full_KG']['typed_edges']
    community = load('graph_k7/automatic_communities.json')
    coupling = load('graph_k7/system_coupling.json')
    patterns = load('patterns/analysis_summary.json')
    hierarchy = load('hub_hierarchy/analysis_summary.json')
    hub_rows = rows('hub_hierarchy/enhanced_system_hub_metrics.csv')
    ablation = load('adapted_model_graph/graph_ablation/analysis_summary.json')
    structures = rows('adapted_model_graph/graph_ablation/structure_comparison.csv')
    common_silver = load('cross_model_silver_evaluation/summary.json')
    common_rows = rows('cross_model_silver_evaluation/comparison.csv')
    chain_rows = rows('graph_k7/propagation_candidate_chains.csv')

    # Figure 3: the executed topic sweep, with its exact measured selection.
    ks = [int(row['k']) for row in curve]
    cs = [float(row['coherence_c_v']) for row in curve]
    plt.figure(figsize=(6.6, 3.35), layout='constrained')
    plt.plot(ks, cs, 'o-', color=BASE_COLOR, markerfacecolor='white',
             markeredgecolor=BASE_COLOR, markeredgewidth=1.3, lw=1.7, ms=5)
    plt.scatter([topics['best_k']], [topics['best_coherence_c_v']], color=ENHANCED_COLOR,
                edgecolor='white', linewidth=.8, s=65, zorder=3)
    plt.annotate(f"$K={topics['best_k']}$, $c_v={topics['best_coherence_c_v']:.4f}$", (topics['best_k'], topics['best_coherence_c_v']),
                 xytext=(6.5, 0.377), fontsize=10,
                 arrowprops={'arrowstyle': '-', 'color': '#687783', 'lw': .8})
    plt.xticks(ks); plt.ylim(0.26, 0.39)
    plt.xlabel('Number of topics, $K$'); plt.ylabel('Topic coherence, $c_v$')
    plt.grid(axis='y', alpha=.8)
    save_plot('topic_coherence.pdf')

    # Figure 4: directional soft system coupling. No artificial floor is added.
    short_systems = ['Fuel', 'Hydraulic', 'Cooling', 'Electrical', 'Mechanical', 'Intake/exhaust', 'Transmission']
    matrix = np.asarray(coupling['max_normalized'])
    fig, ax = plt.subplots(figsize=(6.8, 5.1), layout='constrained')
    heatmap = ax.imshow(matrix, vmin=0, vmax=1, cmap='RdYlBu_r', aspect='equal')
    ax.set_xticks(range(7), short_systems, rotation=35, ha='right', rotation_mode='anchor')
    ax.set_yticks(range(7), short_systems)
    ax.set_xticks(np.arange(-.5, 7, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 7, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=.8)
    ax.tick_params(which='both', length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center',
                    color='white' if matrix[i,j] < .15 or matrix[i,j] > .85 else '#20303e', fontsize=10)
    ax.set_xlabel('Target system label', labelpad=10)
    ax.set_ylabel('Source system label', labelpad=10)
    bar = fig.colorbar(heatmap, ax=ax, shrink=.86, pad=.035,
                       label='Max-normalized association mass', ticks=np.linspace(0, 1, 6))
    bar.outline.set_linewidth(.5)
    save_plot('system_coupling.pdf')

    # Figure 5: recount directed pair weights, averaging parallel typed edges.
    pair_weights = collections.defaultdict(list)
    for edge in graph['analysis_edges']:
        if edge['source'] != edge['target']:
            pair_weights[edge['source'], edge['target']].append(edge['weight'])
    within, between = [], []
    assignment = community['node_community']
    for (u, v), values in pair_weights.items():
        if u in assignment and v in assignment:
            (within if assignment[u] == assignment[v] else between).append(float(np.mean(values)))
    for name, values in [('within', within), ('between', between)]:
        stored = graph_summary['auto_within_between'][name]
        assert len(values) == stored['n_directed_pairs'] and math.isclose(float(np.mean(values)), stored['mean'])
    fig, ax = plt.subplots(figsize=(6.6, 3.45), layout='constrained')
    for name, values, color in [('Between communities', between, BETWEEN_COLOR),
                                 ('Within communities', within, WITHIN_COLOR)]:
        ax.hist(values, bins=np.linspace(0, 1, 16), weights=np.ones(len(values)) / len(values),
                color=color, alpha=.55, edgecolor='white', linewidth=.7, label=name)
        ax.axvline(np.mean(values), color=color, ls='--', lw=1.4)
    ax.set_xlabel('Directed association weight')
    ax.set_ylabel('Fraction of directed pairs per bin')
    ax.legend(frameon=False, loc='upper left')
    ax.text(0, 1.06, f'Within: n={len(within):,}, mean={np.mean(within):.4f}   |   '
                       f'Between: n={len(between):,}, mean={np.mean(between):.4f}',
            transform=ax.transAxes, va='bottom', ha='left', fontsize=9.5, color='#42515e')
    ax.set_xlim(0, 1); ax.grid(axis='y', alpha=.8)
    save_plot('community_weights.pdf')

    # Figure 6 redraws the exported graph topology as a legible vector figure.
    import networkx as nx
    example = load('graph_k7/example_case.json')
    lookup = rows('graph_k7/example_node_lookup.csv')
    node_labels = {row['node_id']: row['plot_label'] for row in lookup}
    node_types = {row['node_id']: row['type'] for row in lookup}
    node_labels['mechanism_aging'] = 'M1'
    node_labels['mechanism_leakage'] = 'M2'
    base_positions = {'N11': (0, .8), 'N9': (1.8, .8), 'N7': (3.6, .8),
                      'N10': (0, 2.4), 'N4': (3.6, 2.4)}
    enhanced_positions = {'N11': (0, .8), 'N9': (1.5, .8), 'N7': (3.6, .8),
                          'N10': (0, -.6), 'N4': (3.6, -.6),
                          'N3': (0, 3.2), 'N6': (.9, 3.2), 'N8': (1.8, 3.2),
                          'N12': (2.7, 3.2), 'N13': (3.6, 3.2),
                          'N5': (0, 2.0), 'M1': (2.5, 2.0), 'M2': (2.5, .1)}
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.05))
    for ax, layer, title in [(axes[0], 'base', f"(a) Base graph\n{len(example['base_nodes'])} nodes, {len(example['base_edges'])} directed pairs"),
                             (axes[1], 'enhanced', f"(b) Enhanced graph\n{len(example['enhanced_nodes'])} nodes, {len(example['enhanced_edges'])} directed pairs")]:
        graph_case = nx.DiGraph()
        graph_case.add_nodes_from(example[layer + '_nodes'])
        graph_case.add_edges_from(example[layer + '_edges'])
        positions = base_positions if layer == 'base' else enhanced_positions
        pos = {node: positions[node_labels[node]] for node in graph_case}
        normal = [node for node in graph_case if not node.startswith('mechanism_')]
        mechanism = [node for node in graph_case if node.startswith('mechanism_')]
        normal_colors = ['#f5ce57' if node_types[node] in ['故障现象', '影响后果']
                         else '#ef5063' for node in normal]
        nx.draw_networkx_nodes(graph_case, pos, nodelist=normal, node_size=440,
                               node_color=normal_colors, edgecolors='white', linewidths=1, ax=ax)
        nx.draw_networkx_nodes(graph_case, pos, nodelist=mechanism, node_size=500,
                               node_color='#547f95', edgecolors='white', linewidths=1, ax=ax)
        edge_colors = [ENHANCED_COLOR if u.startswith('mechanism_') or v.startswith('mechanism_')
                       else '#79838c' for u, v in graph_case.edges]
        nx.draw_networkx_edges(graph_case, pos, arrowsize=13, width=.95,
                               edge_color=edge_colors, node_size=500,
                               min_source_margin=2, min_target_margin=2, ax=ax)
        nx.draw_networkx_labels(graph_case, pos, labels={node: node_labels[node] for node in normal}, font_size=10, ax=ax)
        nx.draw_networkx_labels(graph_case, pos, labels={node: node_labels[node] for node in mechanism},
                                font_color='white', font_size=10, ax=ax)
        ax.set_title(title, fontsize=10.5, linespacing=1.5)
        ax.margins(.18); ax.axis('off')
    from matplotlib.lines import Line2D
    legend_handles = [Line2D([], [], marker='o', ls='', color=c, markersize=7, label=name)
                      for c, name in [('#ef5063', 'Entity mentions'),
                                      ('#f5ce57', 'Phenomenon / outcome'),
                                      ('#547f95', 'Mechanism class')]]
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(.5, .005),
               ncol=3, frameon=False, fontsize=9.5, columnspacing=1.5)
    fig.subplots_adjust(left=.015, right=.985, top=.80, bottom=.16, wspace=.22)
    save_plot('case_graph.pdf')

    # Figure 7: static topology and scope, never time, speed, severity or probability.
    fig = plt.figure(figsize=(7.5, 5.55))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], height_ratios=[1.25, 1], hspace=.80, wspace=.75)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    pattern_names = ['linear', 'branching', 'convergent']
    x = np.arange(3)
    for off, layer, color in [(-.18, 'base', BASE_COLOR), (.18, 'enhanced', ENHANCED_COLOR)]:
        vals = [patterns['layers'][layer]['pattern_unique_sequences'][key] for key in pattern_names]
        bars = axes[0].bar(x + off, vals, .36, color=color, edgecolor='white', lw=.6, label=layer.capitalize())
        axes[0].bar_label(bars, padding=3, fontsize=9)
    axes[0].set_xticks(x, [s.capitalize() for s in pattern_names], rotation=20, ha='right')
    axes[0].set_ylabel('Unique topology candidates')
    axes[0].set_title('(a) Case-supported patterns', loc='left')
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.25)
    axes[0].legend(frameon=False, fontsize=9, loc='upper left')
    axes[0].grid(axis='y', alpha=.8)
    labels = ['Mechanical', 'Transmission', 'Hydraulic', 'Intake/exhaust', 'Electrical', 'Cooling', 'Fuel']
    shares = [100 * float(row['cross_system_incident_weight_share']) for row in hub_rows]
    hub_colors = ['#a98fc1', '#a6a6a6', '#8bb5cf', '#e4b657', '#84b68a', '#5b97c6', '#d96368']
    bars = axes[1].barh(labels[::-1], shares[::-1], color=hub_colors[::-1], edgecolor='white', lw=.6, height=.68)
    axes[1].bar_label(bars, labels=[f'{v:.2f}' for v in shares[::-1]], padding=4, fontsize=9)
    axes[1].set_xlim(0, max(shares) * 1.28)
    axes[1].set_xlabel('Cross-system incident weight (%)')
    axes[1].set_title('(b) Weak system mapping', loc='left')
    axes[1].grid(axis='x', alpha=.8)
    scope = hierarchy['hierarchy_metrics']['enhanced']['unique_path_scope_counts']
    scope_keys = ['component', 'subsystem', 'cross_system_candidate', 'explicit_engine_consequence', 'unknown']
    bars = axes[2].bar(np.arange(5), [scope[key] for key in scope_keys],
                color=['#d95d65', '#3b83b7', '#5c9a59', '#936ca7', '#999999'],
                edgecolor='white', lw=.6, width=.62)
    axes[2].bar_label(bars, labels=[f'{scope[key]:,}' for key in scope_keys], padding=4, fontsize=9.5)
    axes[2].set_xticks(np.arange(5), ['Component', 'Subsystem', 'Cross-system', 'Engine', 'Unknown'])
    axes[2].set_ylabel('Unique candidate paths')
    axes[2].set_title('(c) Static path scope', loc='left')
    axes[2].set_ylim(0, max(scope.values()) * 1.22)
    axes[2].grid(axis='y', alpha=.8)
    fig.subplots_adjust(left=.09, right=.98, top=.92, bottom=.09)
    save_plot('static_structure.pdf')

    # Figure 8: computed structural counts; no interpretability score is plotted.
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 3.0), layout='constrained')
    from matplotlib.ticker import StrMethodFormatter
    metrics = [('Nodes', graph_summary['base_full_KG']['nodes'], graph_summary['enhanced_full_KG']['nodes']),
               ('Typed edges', graph_summary['base_full_KG']['typed_edges'], graph_summary['enhanced_full_KG']['typed_edges']),
               ('Qualified candidate paths', graph_summary['propagation_candidate_paths']['before']['n_qualified_propagation_candidate_paths'],
                graph_summary['propagation_candidate_paths']['after']['n_qualified_propagation_candidate_paths'])]
    for ax, (title, a, b) in zip(axes, metrics):
        bars = ax.bar([0, 1], [a, b], color=[BASE_COLOR, ENHANCED_COLOR], edgecolor='white', lw=.6, width=.60)
        ax.set_xticks([0, 1], ['Base', 'Enhanced'])
        ax.set_title(title.replace('candidate paths', 'candidate\npaths'), fontsize=10.5)
        ax.set_ylim(0, max(a, b) * 1.22)
        ax.bar_label(bars, labels=[f'{a:,}', f'{b:,}'], padding=4, fontsize=10)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.grid(axis='y', alpha=.8)
    axes[0].set_ylabel('Count')
    save_plot('structural_comparison.pdf')

    ner_table_rows = []
    for name, variant in [('Source loading', 'source_roberta'), ('Aligned encoder', 'bert_identity_repair_rebuilt')]:
        for label, part in [('Mixed test (890)', 'all'), ('Template subset (800)', 'template'), ('Case-text subset (90)', 'domain')]:
            m = ner[variant][part]
            ner_table_rows.append([name, label, percentage(m['precision']), percentage(m['recall']), percentage(m['f1'])])
    for name, model in [('Before adaptation', 'baseline'), ('After adaptation', 'selected')]:
        m = adapt_metrics[model]
        ner_table_rows.append([name, 'Internal case test (15)', percentage(m['precision']), percentage(m['recall']), percentage(m['f1'])])
    old_re = re_summary['test']['original225_previously_evaluated']
    silver_rows = []
    for model, col in [('Frozen models', 'baseline_mean'), ('Updated models', 'new_mean')]:
        for label, variant in [('Unweighted', 'base_unweighted__no_mechanisms'), ('Fusion + mechanisms', 'full_fusion__mechanisms')]:
            entries = {r['metric']: r for r in common_rows if r['variant'] == variant}
            silver_rows.append([model, label, percentage(float(entries['recall_at_3'][col])), f"{float(entries['mrr'][col]):.4f}"])

    # The fixed example labels describe these exact stored chain identities.
    expected_examples = json.loads((TEMPLATE / 'chain_examples.json').read_text())
    used_files.add(TEMPLATE / 'chain_examples.json')
    by_identity = {row['node_ids']: row for row in chain_rows}
    selected_chains = []
    chain_cells = []
    for example in expected_examples:
        row = by_identity[example['path_id']]
        selected_chains.append(row)
        left, right = example['label'].rsplit(r' $\to$ ', 1)
        chain_cells.append([left + r'\newline $\to$ ' + right,
                            row['literal_case_frequency'], row['unique_body_support_frequency'],
                            f"{float(row['average_edge_weight']):.4f}"])
    payloads = {'tab:ClassicFaultChain': chain_cells,
                'tab:ner_metrics': ner_table_rows,
                'tab:re_metrics': [
                    ['Inferred-template heldout', re_test['n'], percentage(re_test['accuracy']), percentage(re_test['macro_f1_all_nine_classes'])],
                    ['Previously evaluated original test', old_re['n'], percentage(old_re['accuracy']), percentage(old_re['macro_f1_all_nine_classes'])]],
                'tab:thresholds': [[f'{float(k):.1f}', *(f'{int(v[name]):,}' for name in
                    ['n_retained_occurrence_pairs','n_retained_cases','n_typed_canonical_edges'])]
                    for k,v in sorted(overall['threshold_sensitivity'].items(), key=lambda item:float(item[0]))],
                'tab:silver': silver_rows}
    labels_seen = []

    def update_table(match):
        block = match.group()
        label = re.search(r'\\label\{([^{}]+)\}', block).group(1)
        if label not in payloads:
            return block
        labels_seen.append(label)
        before, body = block.split(r'\midrule', 1)
        _, after = body.split(r'\bottomrule', 1)
        lines, previous = [], None
        for index, cells in enumerate(payloads[label]):
            cells = list(map(str, cells))
            if label == 'tab:ner_metrics' and index in (3,6):
                lines.append(r'\midrule')
            if label == 'tab:silver' and index == 2:
                lines.append(r'\midrule')
            if label in ('tab:ner_metrics','tab:silver'):
                model = cells[0]
                if model == previous:
                    cells[0] = ''
                previous = model
            lines.append(' & '.join(cells) + r' \\')
            if label == 'tab:ClassicFaultChain' and index < len(payloads[label])-1:
                lines.append(r'\addlinespace[3pt]')
        return before + r'\midrule' + '\n' + '\n'.join(lines) + '\n' + r'\bottomrule' + after

    pattern = r'\\begin\{table\}.*?\\end\{table\}'
    manuscript = re.sub(pattern, update_table, original, flags=re.S)
    if set(labels_seen) != set(payloads):
        raise ValueError('Measured table labels in the template are incomplete')
    if re.sub(pattern, '', manuscript, flags=re.S) != re.sub(pattern, '', original, flags=re.S):
        raise ValueError('Non-table manuscript content changed')
    (TEX / 'main.tex').write_text(manuscript, encoding='utf-8')
    # Enforce the quantities already stated in the preserved abstract.
    invariant = json.loads((TEMPLATE / 'measured_text_invariants.json').read_text())
    used_files.add(TEMPLATE / 'measured_text_invariants.json')
    actual = {'cases':len(cases),'unique_bodies':len(set(body_keys)),
              'topic_k':topics['best_k'],'topic_coherence':topics['best_coherence_c_v'],
              'base_nodes':graph_summary['base_full_KG']['nodes'],
              'base_edges':graph_summary['base_full_KG']['typed_edges'],
              'enhanced_nodes':graph_summary['enhanced_full_KG']['nodes'],
              'enhanced_edges':graph_summary['enhanced_full_KG']['typed_edges'],
              'paths_before':graph_summary['propagation_candidate_paths']['before']['n_qualified_propagation_candidate_paths'],
              'paths_after':graph_summary['propagation_candidate_paths']['after']['n_qualified_propagation_candidate_paths'],
              'mixed_ner_f1':ner['bert_identity_repair_rebuilt']['all']['f1'],
              'case_ner_f1':ner['bert_identity_repair_rebuilt']['domain']['f1'],
              're_accuracy':re_test['accuracy'],'re_macro_f1':re_test['macro_f1_all_nine_classes']}
    differing = {name:{'template':value,'result':actual[name]} for name,value in invariant.items()
                 if not math.isclose(float(value),float(actual[name]),rel_tol=0,abs_tol=5e-5)}
    if differing:
        raise ValueError('Measured results differ from preserved manuscript prose; review required: ' + json.dumps(differing))
    metadata = {'status':'completed','created_utc':datetime.now(timezone.utc).isoformat(),
                'experiment':str(EXP),'data_status':'MEASURED_COMPUTATION',
                'non_table_manuscript_content_unchanged':True,
                'tables':{label:len(values) for label,values in payloads.items()},
                'expert_rubric_preserved_without_simulated_scores':True,
                'corpus':{'numbered_records':len(cases),'unique_normalized_bodies':len(set(body_keys))},
                'consumed_files_sha256':{str(path.relative_to(PROJECT_ROOT)):sha(path) for path in sorted(used_files)},
                'generator_code_sha256':source_dependencies('diesel_kg.visualization.manuscript'),
                'figure_count':len(list(FIG.iterdir())),
                'measured_text_invariants_checked':actual,
                'human_causal_or_industrial_validation':False}
    if compile_pdf:
        metadata['compilation'] = compile_document(TEX, OUT)
    archive = OUT / 'overleaf.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(TEX.rglob('*')):
            if path.is_file():
                bundle.write(path, path.relative_to(TEX))
    metadata['source_archive_sha256'] = sha(archive)
    (OUT / 'data_sources.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':metadata['status'],'figures':metadata['figure_count'],
                      'measured_tables':len(payloads),'output':str(OUT),
                      'compiled':compile_pdf},ensure_ascii=False),flush=True)
    return metadata


def compile_document(source: Path, output: Path):
    executable = shutil.which('latexmk')
    if not executable:
        raise FileNotFoundError('latexmk is required for --compile; install a TeX distribution')
    with tempfile.TemporaryDirectory(prefix='latex-',dir=output) as build_dir:
        command = [executable,'-pdf','-interaction=nonstopmode','-halt-on-error',
                   '-file-line-error','-outdir='+build_dir,'main.tex']
        with (output/'compile.log').open('w') as log:
            result = subprocess.run(command,cwd=source,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('LaTeX compilation failed; see '+str(output/'compile.log'))
        pdf = Path(build_dir)/'main.pdf'
        shutil.copy2(pdf,output/'paper.pdf')
        diagnostics = (Path(build_dir)/'main.log').read_text(errors='replace')
        problems = [line.strip() for line in diagnostics.splitlines() if
                    'Overfull \\hbox' in line or 'undefined references' in line or
                    'undefined citations' in line or 'LaTeX Error:' in line]
        if problems:
            raise RuntimeError('Unresolved manuscript layout or reference errors: '+str(problems))
        pages = re.search(r'Output written on .*?\((\d+) pages?',diagnostics,re.S)
        return {'status':'passed','pages':int(pages.group(1)) if pages else None,
                'pdf_sha256':hashlib.sha256((output/'paper.pdf').read_bytes()).hexdigest(),
                'command':command,'layout_and_references_checked':True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compile',action='store_true',dest='compile_pdf')
    args = parser.parse_args()
    build(args.results_dir,args.output,args.compile_pdf)


if __name__ == '__main__':
    main()
