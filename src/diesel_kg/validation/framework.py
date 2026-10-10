"""Verify saved full-run accounting and provenance, independently of summaries.

Passing this verifier confirms computational consistency, never independent
causal or industrial validity. Missing external evidence must stay explicit.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import tempfile

PRIMARY_NER_VARIANTS = ('source_roberta','bert_identity_repair_rebuilt')


def load(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''): h.update(b)
    return h.hexdigest()


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def occurrence_ner_accounting(cases_path, mentions_path, metadata, variant, mentions=None):
    """Bind corpus NER metadata to actual bytes, rows and training identity."""
    records = rows(mentions_path) if mentions is None else mentions
    cases = rows(cases_path)
    mention_count = sum(len(r['entities']) for r in records)
    expected_counts = {'cases':len(cases), 'mentions':mention_count, 'offset_checks':mention_count,
                       'one_character_mentions':sum(len(e['text'])==1 for r in records for e in r['entities'])}
    counts = metadata.get('counts',{})
    counts_valid = all(isinstance(counts.get(k),int) and not isinstance(counts[k],bool) and counts[k]==v for k,v in expected_counts.items())
    return {'count_and_offset_check_accounting': counts_valid,
            'input_output_file_hash_binding': metadata.get('cases_sha256')==sha(cases_path) and metadata.get('output_sha256')==sha(mentions_path),
            'training_variant_binding': metadata.get('variant')==variant and all(r.get('ner_variant')==variant for r in records),
            'recounted_counts':expected_counts}


def source_forward_cache_assessment(summary):
    """No classified pairs means no source logits to compare, never a fake pass."""
    count=summary.get('counts',{}).get('classified_ordered_pairs',0)
    checks=summary.get('encoder_cache_checks',[])
    if not isinstance(count,int) or isinstance(count,bool) or count<0 or not isinstance(checks,list):
        return {'passed':False,'status':'invalid_pair_or_cache_accounting'}
    if count==0:
        return {'passed':not checks and summary.get('counts',{}).get('encoder_windows',0)==0,
                'status':'not_evaluated_no_pairs','classified_ordered_pairs':0,'n_source_forward_comparisons':len(checks),
                'source_logits_compared':False}
    equivalent=bool(checks) and len(checks)<=count
    for record in checks:
        error,tolerance=record.get('maximum_logit_difference'),record.get('atol')
        valid=(isinstance(error,(float,int)) and not isinstance(error,bool) and isinstance(tolerance,(float,int)) and
               not isinstance(tolerance,bool) and math.isfinite(error) and math.isfinite(tolerance) and
               0<=error<=tolerance<=3e-5)
        equivalent &= valid
    return {'passed':equivalent,'status':'evaluated' if equivalent else 'missing_or_failed_source_forward_comparison',
            'classified_ordered_pairs':count,'n_source_forward_comparisons':len(checks),'source_logits_compared':bool(checks)}


def _simple_paths(edge_pairs):
    """Independent uncapped DFS of all simple path prefixes, >=2 nodes."""
    adjacency = defaultdict(set)
    for u, v in edge_pairs:
        if u != v: adjacency[u].add(v)
    found = set()
    for start in sorted(adjacency):
        stack = [(start, (start,))]
        while stack:
            current, path = stack.pop()
            for nxt in adjacency[current] - set(path):
                extended = path + (nxt,); found.add(extended)
                stack.append((nxt, extended))
    return found


def _candidate_in_case(ids, nodes, pair_edges, cid):
    influence = {'causal_candidate', 'functional_influence', 'mechanism_chain_inherited_candidate'}
    first_context = {'structural_context', 'spatial_context', 'physical_mechanism_context'}
    physical = {'故障部位', '部件', '组件', '系统'}
    if nodes[ids[-1]]['entity_type'] not in {'故障现象', '影响后果'}:
        return False
    if any(nodes[n].get('layer') == 'mechanism' and set(ids) & set(nodes[n].get('member_entity_ids', [])) for n in ids):
        return False
    has_influence = False
    for i, (u, v) in enumerate(zip(ids, ids[1:])):
        kinds = {e['kind'] for e in pair_edges[u, v] if cid in e['case_support']}
        has_influence |= bool(kinds & influence)
        allowed = kinds & influence
        if i == 0 and nodes[u]['entity_type'] in physical: allowed |= kinds & first_context
        if i == len(ids)-2: allowed |= kinds & {'manifestation_context'}
        if not allowed: return False
    return has_influence


def verify_fresh_branch(root, variant, cases):
    """Check fresh artifacts against upstream *actual* qualified observations.

    Callable on small temporary fixtures. The full-run caller separately requires
    the original 1522 case IDs; fixtures never become experimental evidence.
    """
    from diesel_kg.graph.analysis import MECHANISMS, FAULT_TYPES, RELATIONS
    root = Path(root); checks = []
    def check(name, passed, detail=None):
        checks.append({'name': variant+'_fresh_'+name, 'passed': bool(passed), 'detail': detail})
    upstream = root/f'occurrence_{variant}/re_locator_fixed'
    folder = root/f'fresh_graph_{variant}'
    upstream_summary = load(upstream/'run_summary.json')
    source = load(upstream/'qualified_predicted_graph.json')
    observations = rows(upstream/'qualified_triples.jsonl')
    summary = load(folder/'analysis_summary.json')
    source_nodes = {n['id']: n for n in source['nodes']}
    source_edges = {e['id']: e for e in source['edges']}
    case_text = {c['case_id']: c['text'] for c in cases}; case_ids = set(case_text)
    check('upstream_accounting', len(source_nodes)==len(source['nodes'])==upstream_summary['qualified_graph_nodes'] and
          len(source_edges)==len(source['edges'])==upstream_summary['qualified_graph_edges'] and
          len(observations)==upstream_summary['counts'].get('qualified_observations', 0))
    if not upstream_summary['qualified_graph_edges']:
        check('empty_upstream_explicit_not_evaluable', summary.get('status')=='not_evaluable_empty_qualified_model_graph' and
              summary.get('qualified_graph_edges')==0 and summary.get('historical_export_inserted') is False and
              Path(summary.get('upstream_summary','')).resolve()==(upstream/'run_summary.json').resolve())
        nonempty = []
        for label in ('base','enhanced'):
            path = folder/f'{label}_graph.json'
            if path.exists():
                data = load(path)
                if data.get('nodes') or data.get('edges') or data.get('basic_edges') or data.get('analysis_edges'):
                    nonempty.append(str(path))
        for path in folder.glob('*/*summary.json'):
            data = load(path)
            if data.get('status','').startswith('completed'): nonempty.append(str(path))
        check('empty_upstream_no_historical_rescue_or_stale_nonempty_artifacts', not source_nodes and not source_edges and
              not observations and not nonempty, nonempty)
        return checks

    required = [folder/'run_config.json', folder/'base_graph.json', folder/'enhanced_graph.json',
                folder/'patterns/analysis_summary.json', folder/'hub_hierarchy/analysis_summary.json',
                folder/'engineering/analysis_summary.json', folder/'engineering/run_config.json',
                folder/'engineering/case_queries_and_inference.jsonl', folder/'graph_modes_verification.json']
    for label in ('base','enhanced'):
        required.extend(folder/f'{label}_{suffix}' for suffix in ['case_path_completion.csv', 'case_grounded_paths.jsonl', 'propagation_candidate_paths.jsonl'])
    missing = [str(p) for p in required if not p.is_file()]
    check('downstream_artifacts_present', not missing, missing)
    if missing: return checks
    basic, enhanced = load(folder/'base_graph.json'), load(folder/'enhanced_graph.json')
    configuration = load(folder/'run_config.json')
    check('model_occurrence_support_mode', all(d.get('support_mode')=='model_occurrences' for d in [basic, enhanced, summary]) and
          configuration.get('arguments',{}).get('support_mode')=='model_occurrences')
    check('actual_model_input_and_cases_hash', configuration['input_sha256'].get('export')==sha(upstream/'qualified_predicted_graph.json') and
          configuration['input_sha256'].get('cases')==sha(root/'inputs/cases_all.jsonl') and
          Path(configuration['arguments'].get('export_json','')).resolve()==(upstream/'qualified_predicted_graph.json').resolve())
    node_by_semantic = {(n['entity_type'],n['name']): n['id'] for n in source_nodes.values()}
    observed_node_cases, observed_edge_cases = defaultdict(set), defaultdict(set)
    observations_valid = True
    for triple in observations:
        cid = triple['case_id']; semantic = []
        for endpoint in ('head','tail'):
            mention = triple[endpoint]; key = (mention['type'], mention['text']); semantic.append(key)
            nid = node_by_semantic.get(key)
            valid_span = (isinstance(mention.get('start'), int) and not isinstance(mention.get('start'), bool) and
                          isinstance(mention.get('end'), int) and not isinstance(mention.get('end'), bool) and cid in case_text and
                          0<=mention['start']<mention['end']<=len(case_text[cid]) and
                          case_text[cid][mention['start']:mention['end']]==mention['text'])
            observations_valid &= nid is not None and valid_span and triple.get('causal_gold') is False
            if nid is not None: observed_node_cases[nid].add(cid)
        observed_edge_cases[(semantic[0], triple['relation'], semantic[1])].add(cid)
    source_valid = len(node_by_semantic)==len(source_nodes) and observations_valid
    for nid, node in source_nodes.items():
        support = set(node.get('case_ids',[]))
        source_valid &= bool(support) and support==observed_node_cases[nid] and support<=case_ids
    edge_semantics = set()
    for edge in source_edges.values():
        u,v = source_nodes[edge['source']],source_nodes[edge['target']]
        key = ((u['entity_type'],u['name']),edge['relation_type'],(v['entity_type'],v['name']))
        edge_semantics.add(key); support = set(edge.get('case_ids',[]))
        source_valid &= bool(support) and support==observed_edge_cases[key] and support<=set(u['case_ids']) & set(v['case_ids'])
    check('qualified_observation_to_graph_exact_case_support', source_valid and edge_semantics==set(observed_edge_cases))
    base_nodes = {n['id']:n for n in basic['nodes']}; base_edges = {e['id']:e for e in basic['edges']}
    preserved = set(base_nodes)==set(source_nodes) and set(base_edges)==set(source_edges)
    for nid,node in base_nodes.items():
        raw = source_nodes.get(nid,{})
        preserved &= (node.get('name'),node.get('entity_type'))==(raw.get('name'),raw.get('entity_type'))
        preserved &= set(node.get('case_support',[]))==set(raw.get('case_ids',[])) and 'input_model_qualified' in node.get('case_support_kind','')
    for eid,edge in base_edges.items():
        raw = source_edges.get(eid,{})
        preserved &= (edge.get('source'),edge.get('target'),edge.get('relation_type'))==(raw.get('source'),raw.get('target'),raw.get('relation_type'))
        preserved &= set(edge.get('case_support',[]))==set(raw.get('case_ids',[])) and 'input_model_qualified' in edge.get('case_support_kind','')
    check('base_nodes_edges_and_cases_preserved_without_literal_expansion', preserved)
    nodes = {n['id']:n for n in enhanced['nodes']}; bridges = enhanced.get('mechanism_edges',[])
    augmentation_valid = set(base_nodes)<=set(nodes) and enhanced.get('basic_edges')==basic['edges']
    for nid,node in nodes.items():
        if nid in base_nodes:
            augmentation_valid &= node==base_nodes[nid]; continue
        mechanism = nid.removeprefix('mechanism_'); members = node.get('member_entity_ids',[])
        expected = {i for i,n in base_nodes.items() if n['entity_type'] in FAULT_TYPES and
                    any(term in n['name'] for term in MECHANISMS.get(mechanism,[]))}
        support = set().union(*(set(base_nodes[i]['case_support']) for i in expected))
        augmentation_valid &= nid.startswith('mechanism_') and mechanism in MECHANISMS and node.get('layer')=='mechanism'
        augmentation_valid &= set(members)==expected and bool(expected) and set(node['case_support'])==support
    base_analysis_ids = {e['id'] for e in basic['analysis_edges']}
    eligible_ids = {eid for eid,e in base_edges.items() if e['relation_type'] in RELATIONS}
    relation_mapping_valid = True
    physical_types = {'故障部位','部件','组件','系统'}
    for analysis_edge in basic['analysis_edges']:
        basis = analysis_edge.get('basis_base_edge_ids',[])
        if len(basis)!=1 or basis[0] not in eligible_ids:
            relation_mapping_valid=False; continue
        original = base_edges[basis[0]]; kind,reverse = RELATIONS[original['relation_type']]
        u,v = (original['target'],original['source']) if reverse else (original['source'],original['target'])
        if kind=='spatial_context' and base_nodes[v]['entity_type'] in physical_types and base_nodes[u]['entity_type'] not in physical_types:
            u,v=v,u
        relation_mapping_valid &= (analysis_edge.get('source'),analysis_edge.get('target'),analysis_edge.get('kind'))==(u,v,kind)
        relation_mapping_valid &= set(analysis_edge.get('case_support',[]))==set(original['case_support']) and analysis_edge.get('causal_gold') is False
    check('base_analysis_relation_mapping_and_cases_exact', relation_mapping_valid)
    check('metadata_not_promoted_to_physical_or_influence_edges', base_analysis_ids=={'analysis_'+eid for eid in eligible_ids} and
          all(not (set(e['basis_base_edge_ids'])-eligible_ids) for e in bridges if e['kind']!='mechanism_membership_context'))
    for e in bridges:
        basis = set(e.get('basis_base_edge_ids',[])); support = set(e.get('case_support',[]))
        allowed_cases = set().union(*(set(base_edges[b]['case_support']) for b in basis if b in base_edges))
        augmentation_valid &= bool(basis) and basis<=set(base_edges) and e.get('causal_gold') is False
        augmentation_valid &= support<=allowed_cases & set(nodes[e['source']]['case_support']) & set(nodes[e['target']]['case_support'])
        if e['kind']=='mechanism_chain_inherited_candidate':
            augmentation_valid &= all(RELATIONS[base_edges[b]['relation_type']][0] in {'causal_candidate','functional_influence'} for b in basis if b in eligible_ids)
    check('only_traceable_mechanism_augmentation_no_new_physical_nodes', augmentation_valid)
    topic_folder = root/f'topics_{variant}_dictionary'; topics = load(topic_folder/'analysis_summary.json')
    topic_sources = summary.get('topic_sources',{})
    topic_valid = topics.get('input',{}).get('entities_sha256')==sha(root/f'occurrence_{variant}/corpus_entities.jsonl')
    topic_valid &= topics.get('input',{}).get('cases_sha256')==sha(root/'inputs/cases_all.jsonl')
    for filename,path_key,hash_key in [('document_topic.csv','document_topic','document_topic_sha256'),
                                      ('case_system_probabilities.csv','case_system_probabilities','case_system_sha256')]:
        expected_path = topic_folder/'paper_fixed_K07'/filename
        topic_valid &= Path(topic_sources.get(path_key,'')).resolve()==expected_path.resolve() and topic_sources.get(hash_key)==sha(expected_path)
    check('semantic_weights_use_this_new_NER_dictionary_topics', topic_valid)
    all_paths_valid = True; strict_count = {}
    for label,data in [('base',basic),('enhanced',enhanced)]:
        layer_nodes = {n['id']:n for n in data['nodes']}; pair_edges = defaultdict(list); pairs_by_case = defaultdict(set)
        for edge in data['analysis_edges']:
            pair_edges[edge['source'],edge['target']].append(edge)
            for cid in edge['case_support']: pairs_by_case[cid].add((edge['source'],edge['target']))
        saved_paths = rows(folder/f'{label}_case_grounded_paths.jsonl')
        actual = defaultdict(set); expected_strict = defaultdict(set)
        for path in saved_paths:
            ids = tuple(path['node_ids']); support = set(path['case_ids'])
            valid = len(ids)>=2 and len(ids)==len(set(ids)) and len(support)==path['literal_case_frequency']==path.get('model_qualified_case_frequency')
            valid &= support<=case_ids and all((u,v) in pair_edges for u,v in zip(ids,ids[1:]))
            for cid in support:
                actual[cid].add(ids)
                valid &= all((u,v) in pairs_by_case[cid] for u,v in zip(ids,ids[1:]))
                if _candidate_in_case(ids,layer_nodes,pair_edges,cid): expected_strict[ids].add(cid)
            all_paths_valid &= valid
        cache = {}; independent_total = 0
        completion = csv_rows(folder/f'{label}_case_path_completion.csv')
        by_id = {r['case_id']:r for r in completion}
        completion_valid = len(completion)==len(cases) and set(by_id)==case_ids
        for cid in case_ids:
            signature = tuple(sorted(pairs_by_case[cid]))
            if signature not in cache: cache[signature] = _simple_paths(signature)
            expected = cache[signature]; independent_total += len(expected)
            record = by_id.get(cid,{})
            completion_valid &= actual[cid]==expected and record.get('complete')=='True' and record.get('n_simple_paths')==str(len(expected))
            completion_valid &= record.get('n_grounded_edges')==str(len(signature))
        path_stats = summary['case_path_statistics']['before' if label=='base' else 'after']
        completion_valid &= path_stats['n_cases_complete']==len(cases) and path_stats['n_cases_incomplete']==0 and path_stats['hop_cap'] is None
        completion_valid &= len(saved_paths)==len({tuple(p['node_ids']) for p in saved_paths})==path_stats['n_unique_case_grounded_simple_paths']
        completion_valid &= independent_total==path_stats['path_case_incidences']
        check(label+'_all_cases_uncapped_paths_independently_enumerated', completion_valid,
              {'cases':len(cases),'independent_path_case_incidences':independent_total,'unique_case_graphs':len(cache)})
        strict = rows(folder/f'{label}_propagation_candidate_paths.jsonl'); strict_count[label]=len(strict)
        saved_strict = {tuple(p['node_ids']):set(p['case_ids']) for p in strict}
        check(label+'_strict_candidates_have_case_specific_influence_support', len(saved_strict)==len(strict) and saved_strict==dict(expected_strict))
        if label=='enhanced' and not data['analysis_edges']:
            communities = summary['automatic_communities']
            check(label+'_zero_projection_statistics_null_without_rescue', not strict and not saved_paths and
                  communities['n_auto_communities']==0 and communities['modularity'] is None and
                  all(v['n_directed_pairs']==0 and v['mean'] is None for v in summary['auto_within_between'].values()))
    check('all_saved_context_paths_restricted_to_actual_model_cases', all_paths_valid and summary.get('path_count_complete') is True)
    if not strict_count['enhanced']:
        check('no_strict_paths_explicit_not_evaluable', summary.get('propagation_evaluation_status')=='not_evaluable_no_qualified_propagation_candidates')
    patterns = load(folder/'patterns/analysis_summary.json'); hubs = load(folder/'hub_hierarchy/analysis_summary.json')
    downstream_bound = patterns.get('support_mode')==hubs.get('support_mode')=='model_occurrences'
    for data in [patterns,hubs]:
        for label in ('base','enhanced'): downstream_bound &= data['input_sha256'][label+'_graph']==sha(folder/f'{label}_graph.json')
        downstream_bound &= data['input_sha256']['cases']==sha(root/'inputs/cases_all.jsonl')
    check('patterns_hubs_bound_to_this_fresh_graph', downstream_bound)
    try:
        from diesel_kg.validation.graph_modes import run as verify_modes
        with tempfile.TemporaryDirectory(prefix='independent_fresh_modes_') as directory:
            mode_result=verify_modes(folder,root/'inputs/cases_all.jsonl',folder/'patterns',folder/'hub_hierarchy',Path(directory)/'checks.json')
        check('pattern_spans_topology_hub_mass_and_hierarchy_independently_verified', mode_result['status']=='passed')
    except (AssertionError,KeyError,ValueError) as exc:
        check('pattern_spans_topology_hub_mass_and_hierarchy_independently_verified',False,str(exc))
    engineering = load(folder/'engineering/analysis_summary.json'); eng_config = load(folder/'engineering/run_config.json')
    scope = engineering.get('source_scope',{}); expected_scope='model_qualified_occurrence_graph_retrospective_symptom_query_replay'
    check('engineering_actual_model_source_scope', scope.get('support_mode')=='model_occurrences' and
          scope.get('graph_source_kind')=='model_qualified_occurrence_relation_candidates' and
          scope.get('evaluation_scope')==expected_scope==engineering['retrospective_query_comparison']['evaluation_scope'] and
          scope.get('graph_source_is_historical_export') is False and scope.get('case_support_is_observed_causal_gold') is False and eng_config.get('source_scope')==scope)
    inputs_bound = all(Path(eng_config['graph_inputs'][label]['path']).resolve()==(folder/f'{label}_graph.json').resolve() and
                       eng_config['graph_inputs'][label]['sha256']==sha(folder/f'{label}_graph.json') for label in ('base','enhanced'))
    queries = rows(folder/'engineering/case_queries_and_inference.jsonl')
    queries_valid = len(queries)==len(cases) and {r['case_id'] for r in queries}==case_ids and inputs_bound
    for query in queries:
        queries_valid &= query['case_body_sha256']==hashlib.sha256(case_text[query['case_id']].encode()).hexdigest()
        for label,data in [('base',basic),('enhanced',enhanced)]:
            graph_nodes={n['id'] for n in data['nodes']}; analysis_ids={e['id'] for e in data['analysis_edges']}
            result=query[label]; queries_valid &= result['n_root_candidates']==len(result['candidate_roots']) and result['query_has_candidate_root']==bool(result['candidate_roots'])
            for candidate in result['candidate_roots']:
                explanation=candidate['best_explanation']; queries_valid &= candidate['node_id'] in base_nodes and set(explanation['node_ids'])<=graph_nodes
                queries_valid &= explanation.get('case_support_mode')=='model_occurrences' and explanation.get('has_verified_causal_gold') is False
                queries_valid &= all(set(e['edge_ids'])<=analysis_ids and set(e['basis_base_edge_ids'])<=set(source_edges) for e in explanation['edges'])
            if not data['analysis_edges']: queries_valid &= not result['candidate_roots'] and not result['core_nodes']
    check('engineering_all_cases_and_explanations_use_actual_fresh_inputs',queries_valid)
    return checks


def verify_argmax_branch(root, variant, cases_file=None, source_re_dir=None, topics_dir=None):
    """Bind saved threshold sensitivity and its independent graph execution.

    Optional paths support temporary fixtures; production uses this variant's
    actual primary RE outputs, corpus and fresh dictionary topics.
    """
    from diesel_kg.pipelines.argmax import check_threshold_ready
    root=Path(root); checks=[]
    cases_file=Path(cases_file) if cases_file else root/'inputs/cases_all.jsonl'
    source_re_dir=Path(source_re_dir) if source_re_dir else root/f'occurrence_{variant}/re_locator_fixed'
    topics_dir=Path(topics_dir) if topics_dir else root/f'topics_{variant}_dictionary'
    threshold_dir=root/f'relation_threshold_{variant}'; folder=threshold_dir/'argmax_graph_analysis'
    def check(name,passed,detail=None): checks.append({'name':variant+'_argmax_'+name,'passed':bool(passed),'detail':detail})
    required=[threshold_dir/'analysis_summary.json',threshold_dir/'run_config.json',threshold_dir/'threshold0_qualified_predicted_graph.json',
              threshold_dir/'threshold0_qualified_triples.jsonl',folder/'execution.json',folder/'analysis_summary.json']
    missing=[str(p) for p in required if not p.is_file()]
    check('required_completed_audit_and_execution_records',not missing,missing)
    if missing: return checks
    try:
        audit,audit_config,source,cases=check_threshold_ready(threshold_dir,topics_dir,cases_file)
        ready=True
    except (OSError,KeyError,ValueError) as exc:
        check('completed_threshold_input_hash_and_dictionary_binding',False,str(exc)); return checks
    check('completed_threshold_input_hash_and_dictionary_binding',ready)
    primary=load(source_re_dir/'run_summary.json'); refs=audit_config['inputs']; binding=audit['source_RE_binding']
    entity_file=source_re_dir.parent/'corpus_entities.jsonl'
    # Snapshot inputs may have a different path, but must match formal bytes.
    formal_entity=entity_file if entity_file.is_file() else Path(refs['entities']['path'])
    formal_raw=source_re_dir/'all_occurrence_pair_predictions.jsonl.gz'
    formal_raw=formal_raw if formal_raw.is_file() else Path(refs['raw_predictions']['path'])
    bound=(refs['source_RE_run_summary']['sha256']==sha(source_re_dir/'run_summary.json')==binding.get('source_RE_summary_sha256') and
           refs['entities']['sha256']==sha(formal_entity) and refs['raw_predictions']['sha256']==sha(formal_raw) and
           binding.get('declared_source_inputs')==primary.get('input_sha256'))
    declared=primary.get('input_sha256',{})
    if str(formal_entity) in declared: bound &= declared[str(formal_entity)]==refs['entities']['sha256']
    check('matching_primary_RE_summary_raw_entities_and_declared_inputs',bound)
    counts_valid=True
    for measured,recorded in [('processed_cases','processed_cases'),('ordered_pairs','ordered_pairs'),('classified_pairs','classified_ordered_pairs'),('mentions','mentions')]:
        counts_valid &= audit.get('counts',{}).get(measured,0)==primary.get('counts',{}).get(recorded,0)
    counts_valid &= audit.get('counts',{}).get('processed_cases',0)==len(cases)
    counts_valid &= audit.get('raw_relation_counts')==primary.get('relation_counts')
    main=audit.get('thresholds',{}).get('0.8',{})
    counts_valid &= (main.get('n_retained_occurrence_pairs')==primary.get('counts',{}).get('qualified_observations',0) and
                     main.get('n_retained_cases')==primary.get('counts',{}).get('cases_with_qualified_edges',0) and
                     main.get('n_typed_canonical_edges')==primary.get('qualified_graph_edges',0) and
                     binding.get('main_0_8_rejection_and_retention_counts_reproduced') is True)
    check('all_pair_counts_saved_labels_and_main_0_8_retention_reproduced',counts_valid)
    execution=load(folder/'execution.json'); graph_summary=load(folder/'analysis_summary.json')
    unchanged=(execution.get('main_0_8_artifacts_unchanged') is True and execution.get('historical_export_inserted') is False and
               execution.get('relation_labels_changed') is False and execution.get('new_gold_annotation_performed') is False and
               execution.get('support_mode')=='model_occurrences' and
               execution.get('main_0_8_artifacts_before_sha256')==execution.get('main_0_8_artifacts_after_sha256'))
    maps=[execution.get('input_sha256',{}),execution.get('main_0_8_artifacts_after_sha256',{}),execution.get('implementation_sha256',{}),execution.get('output_artifacts_sha256',{})]
    hashes=all(bool(m) and all(Path(p).is_file() and sha(p)==h for p,h in m.items()) for m in maps)
    for item in execution.get('commands',[]):
        hashes &= item.get('status')=='completed' and item.get('exit_code')==0
        hashes &= all(Path(p).is_file() and sha(p)==h for p,h in item.get('artifacts_sha256',{}).items())
        hashes &= bool(item.get('local_code_sha256')) and all(Path(p).is_file() and sha(p)==h for p,h in item.get('local_code_sha256',{}).items())
    check('main_0_8_unchanged_and_actual_execution_input_code_artifact_hashes',unchanged and hashes)
    if not source['edges']:
        nonempty=[]
        for label in ['base','enhanced']:
            path=folder/f'{label}_graph.json'
            if path.is_file() and any(load(path).get(k) for k in ['nodes','edges','basic_edges','analysis_edges']): nonempty.append(str(path))
        check('zero_qualified_argmax_graph_explicit_without_historical_rescue',execution.get('status')=='not_evaluable_empty_qualified_model_graph' and
              graph_summary.get('status')=='not_evaluable_empty_qualified_model_graph' and graph_summary.get('qualified_graph_edges')==0 and
              graph_summary.get('historical_export_inserted') is False and not execution.get('commands') and not source['nodes'] and not nonempty,nonempty)
    else:
        verification_path=folder/'argmax_evidence_verification.json'
        verification=load(verification_path) if verification_path.is_file() else {}
        expected_stages=['graph','patterns','hub_hierarchy','engineering','graph_modes_verification']
        check('nonempty_candidate_pipeline_completed_and_independently_verified',
              execution.get('status')=='completed_candidate_computation_not_causal_validation' and
              [c.get('stage') for c in execution.get('commands',[])]==expected_stages and
              graph_summary.get('support_mode')=='model_occurrences' and graph_summary.get('path_count_complete') is True and
              verification.get('status')=='passed' and verification.get('cases')==len(cases) and verification.get('causal_gold_validated') is False and
              bool(verification.get('checks')) and all(c.get('passed') for c in verification['checks']) and
              execution.get('verification_sha256')==(sha(verification_path) if verification_path.is_file() else None))
    return checks


def run(root):
    root=Path(root); checks=[]
    config=load(root/'execution_configuration.json')
    def check(name,passed,detail=None):
        checks.append({'name':name,'passed':bool(passed),'detail':detail})
    cases=rows(root/'inputs/cases_all.jsonl'); case_ids={r['case_id'] for r in cases}
    check('all_original_cases_retained',len(cases)==1522 and len(case_ids)==1522)
    frozen=load(root/'inputs/frozen_manifest.json')
    check('frozen_inputs_match_sources',all(sha(p['source_path'])==p['sha256']==sha(p['frozen_path']) for p in frozen))
    references=load(root/'model_references.json')
    for variant in PRIMARY_NER_VARIANTS:
        summary_path=Path(references[variant]['summary_path'])
        folder=summary_path.parent; summary=load(summary_path); history=load(folder/'epoch_history.json')
        values=[r['dev']['token_entity_micro_f1'] for r in history]
        first_max=values.index(max(values))+1
        check(variant+'_dev_only_selection',summary['best_epoch']==first_max and math.isclose(summary['best_dev_token_micro_f1'],max(values)))
        check(variant+'_model_selection_valid_after_checkpoint_reload',summary.get('status') in {'complete','completed'} and
              summary.get('valid_for_model_selection') is not False and
              all(math.isfinite(score) and 0<=score<=1 for score in [summary['best_dev_token_micro_f1'],summary['dev']['token_entity_micro_f1']]) and
              math.isclose(summary['best_dev_token_micro_f1'],summary['dev']['token_entity_micro_f1'],rel_tol=1e-9,abs_tol=1e-12))
        check(variant+'_checkpoint_hash',sha(references[variant]['checkpoint_path'])==summary['checkpoint_sha256']==references[variant]['checkpoint_sha256'])
        predictions=rows(folder/'test_predictions.jsonl'); gold=set(); predicted=set()
        for r in predictions:
            for e in r['gold_entities']: gold.add((r['sample_id'],e['start'],e['end'],e['type']))
            for e in r['predicted_entities']: predicted.add((r['sample_id'],e['start'],e['end'],e['type']))
        result=summary['test']['exact_span_type']; matched=len(gold&predicted)
        check(variant+'_test_exact_span_recount',len(predictions)==890 and (len(gold),len(predicted),matched)==(result['gold'],result['predicted'],result['matched']))
        occurrence=root/f'occurrence_{variant}'
        mentions=rows(occurrence/'corpus_entities.jsonl'); meta=load(occurrence/'corpus_entities.summary.json')
        case_text={r['case_id']:r['text'] for r in cases}
        check(variant+'_full_corpus_NER_case_and_checkpoint_identity',
              {r['case_id'] for r in mentions}==case_ids and len(mentions)==len(cases) and
              meta['checkpoint_sha256']==summary['checkpoint_sha256'] and
              all(r['text']==case_text[r['case_id']] and r['ner_checkpoint_sha256']==summary['checkpoint_sha256'] for r in mentions))
        accounting=occurrence_ner_accounting(root/'inputs/cases_all.jsonl',occurrence/'corpus_entities.jsonl',meta,variant,mentions)
        for key in ('count_and_offset_check_accounting','input_output_file_hash_binding','training_variant_binding'):
            check(variant+'_full_corpus_NER_'+key,accounting[key],accounting['recounted_counts'] if key=='count_and_offset_check_accounting' else None)
        good_spans=True; occurrence_ids=[]
        for r in mentions:
            for e in r['entities']:
                good_spans &= 0<=e['start']<e['end']<=len(r['text']) and r['text'][e['start']:e['end']]==e['text']
                occurrence_ids.append(e['occurrence_id'])
        check(variant+'_all_occurrence_spans_and_IDs',good_spans and len(set(occurrence_ids))==len(occurrence_ids))
        re_dir=occurrence/'re_locator_fixed'; re_summary=load(re_dir/'run_summary.json'); counts=load(re_dir/'case_counts.json')
        re_ref=references['re_locator_fixed']
        check(variant+'_RE_checkpoint_binding',sha(re_ref['checkpoint_path'])==re_ref['checkpoint_sha256']==
              re_summary['input_sha256'].get(re_ref['checkpoint_path']))
        expected_pairs=sum(len(r['entities'])*(len(r['entities'])-1) for r in mentions)
        check(variant+'_uncapped_ordered_pair_accounting',
              len(counts)==1522 and {r['case_id'] for r in counts}==case_ids and
              sum(r['ordered_pairs'] for r in counts)==expected_pairs==re_summary['counts']['ordered_pairs'] and
              all(r['ordered_pairs']==r['classified_pairs']+r['unclassified_pairs'] for r in counts))
        n_records=n_predictions=n_unclassified=0; raw_ids=set(); pairs_unique=True
        with gzip.open(re_dir/'all_occurrence_pair_predictions.jsonl.gz','rt',encoding='utf8') as f:
            for line in f:
                r=json.loads(line); n_records+=1; raw_ids.add(r['case_id'])
                predictions=r['predictions_ij_relation_probability_token_window']
                n_predictions+=len(predictions); n_unclassified+=sum(r['unclassified_reasons'].values())
                pairs_unique &= len({tuple(p[:2]) for p in predictions})==len(predictions)
        check(variant+'_compressed_raw_pairs_recount',n_records==1522 and raw_ids==case_ids and pairs_unique and
              n_predictions==re_summary['counts'].get('classified_ordered_pairs',0) and n_predictions+n_unclassified==expected_pairs)
        cache_assessment=source_forward_cache_assessment(re_summary)
        cache_name=('_source_forward_equivalence_not_evaluated_no_pairs' if cache_assessment['status']=='not_evaluated_no_pairs'
                    else '_unchanged_source_forward_equivalence')
        check(variant+cache_name,cache_assessment['passed'],cache_assessment)
        graph=load(re_dir/'qualified_predicted_graph.json'); triples=rows(re_dir/'qualified_triples.jsonl')
        graph_nodes={n['id'] for n in graph['nodes']}
        check(variant+'_qualified_graph_and_observation_accounting',
              len(triples)==re_summary['counts']['qualified_observations'] and
              len(graph_nodes)==re_summary['qualified_graph_nodes'] and len(graph['edges'])==re_summary['qualified_graph_edges'] and
              all(e['source'] in graph_nodes and e['target'] in graph_nodes for e in graph['edges']))
        check(variant+'_no_false_existence_or_causal_gold',
              all(t['relationship_existence_modeled']==graph['schema']['relationship_existence_head'] and not t['causal_gold'] for t in triples) and
              graph['schema']['role']=='model_qualified_candidates_not_causal_gold')
        try:
            checks.extend(verify_fresh_branch(root,variant,cases))
        except (KeyError,ValueError,OSError,AssertionError) as exc:
            check(variant+'_fresh_verification_executable',False,str(exc))
        try:
            checks.extend(verify_argmax_branch(root,variant))
        except (KeyError,ValueError,OSError,AssertionError) as exc:
            check(variant+'_argmax_verification_executable',False,str(exc))
    for k in (7,10):
        g=load(root/f'graph_k{k}/analysis_summary.json')
        check(f'K{k}_all_case_path_completion',g['path_count_complete'] and
              all(g['case_path_statistics'][n]['n_cases_complete']==1522 and g['case_path_statistics'][n]['hop_cap'] is None for n in ('before','after')))
    sensitivity=load(root/'graph_topic_sensitivity.json')
    check('K7_regression_and_K10_fixed_topology',sensitivity['status']=='passed' and all(sensitivity['checks'].values()))
    physical=load(root/'system_validation/analysis_summary.json')
    check('physical_system_consistency_requires_external_gold',bool(config.get('physical_system_labels')) or physical['status']=='not_evaluable')
    engineering=load(root/'engineering/analysis_summary.json'); query=rows(root/'engineering/case_queries_and_inference.jsonl')
    check('paired_engineering_all_cases_computed',len(query)==1522 and {r['case_id'] for r in query}==case_ids and
          engineering['retrospective_query_comparison']['n_cases']==1522)
    check('external_expert_and_causal_evidence_status_explicit',
          (bool(config.get('expert_scores')) or engineering['expert_SIS']['status']=='not_evaluable') and
          (bool(config.get('gold')) or engineering['human_gold_diagnosis_evaluation']['status']=='not_evaluable'))
    temporal=load(root/'temporal/analysis_summary.json')
    check('external_timestamp_evidence_status_explicit',bool(config.get('events')) or
          (temporal['status']=='not_evaluable' and not temporal.get('static_graph_used_as_time_data',True)))
    existence=load(root/'relation_existence/run_summary.json')
    check('missing_null_relation_gold_explicit',existence['status'].startswith('not_evaluable'))
    manifest=load(root/'stage_manifest.json')
    check('all_executed_stages_completed',all(r['status']=='completed' for r in manifest.values()))
    check('executed_stage_artifacts_hashes',all(sha(p)==h for r in manifest.values() for p,h in r.get('artifacts_sha256',{}).items()))
    result={'status':'passed' if all(r['passed'] for r in checks) else 'failed','checks':checks,
            'passed_count':sum(r['passed'] for r in checks),'total_count':len(checks),
            'scope':'saved computational outputs and input/model provenance; not independent validation of paper claims',
            'exact_original_paper_numeric_reproduction_confirmed':False}
    (root/'full_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    if result['status']!='passed': raise AssertionError('Full verification failed; inspect full_verification.json')
    print(json.dumps(result,ensure_ascii=False),flush=True); return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output-root',type=Path,required=True)
    run(p.parse_args().output_root)
