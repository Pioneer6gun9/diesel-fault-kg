"""Trace candidate nodes/pairs to a bounded manufacturer architecture catalog.

Produces evidence features for review; never changes prediction or supplies gold.
"""
from __future__ import annotations
from diesel_kg.paths import CONFIG_DIR
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from diesel_kg.evaluation.phase import sha, dump, write_jsonl

def match_roles(name, catalog):
    matches=[]
    for role,item in catalog['roles'].items():
        aliases=[a for a in item['aliases'] if a.casefold() in name.casefold()]
        if aliases:
            matches.append({'role':role,'system_hint':item['system'],'matched_aliases':aliases,
                            'source':item['source'],'source_url':catalog['sources'][item['source']]['url'],
                            'ambiguous_scope':item.get('ambiguous_scope',False)})
    return matches

def pair_references(head_matches, tail_matches, catalog):
    hs={m['role'] for m in head_matches};ts={m['role'] for m in tail_matches}
    refs=[]
    for r in catalog['relations']:
        if r['source_role'] in hs and r['target_role'] in ts:direction='same_as_reference'
        elif r['target_role'] in hs and r['source_role'] in ts:direction='reverse_of_reference'
        else:continue
        refs.append({**r,'lexical_pair_direction':direction,'source_url':catalog['sources'][r['source']]['url'],
                     'candidate_relation_semantics_validated':False})
    return refs

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--catalog',type=Path,default=CONFIG_DIR / 'physical_reference_catalog.json')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);beg=time.perf_counter()
    before={str(x.resolve()):sha(x) for x in (a.graph,a.catalog)}
    g=json.loads(a.graph.read_text());catalog=json.loads(a.catalog.read_text())
    node_rows=[];matches={};systems=Counter()
    for n in g['nodes']:
        m=match_roles(n['name'],catalog);matches[n['id']]=m
        systems.update(set(r['system_hint'] for r in m))
        node_rows.append({'node_id':n['id'],'name':n['name'],'predicted_entity_type':n['entity_type'],
                          'reference_matches':m,'expert_system_label':None,'reviewed':False})
    edge_rows=[];counts=Counter()
    for e in g['edges']:
        refs=pair_references(matches[e['source']],matches[e['target']],catalog)
        counts.update(set(r['lexical_pair_direction'] for r in refs))
        edge_rows.append({'edge_id':e['id'],'source':e['source'],'target':e['target'],
            'predicted_relation_type':e['relation_type'],'actual_candidate_case_ids':e['case_ids'],
            'reference_pair_matches':refs,'reference_status':'lexically_anchored_architecture_pair' if refs else 'unknown_not_negative',
            'causal_gold':False,'physical_strength':None})
    write_jsonl(a.output/'node_reference_matches.jsonl',node_rows)
    write_jsonl(a.output/'edge_reference_matches.jsonl',edge_rows)
    dump(a.output/'reference_catalog_snapshot.json',catalog)
    if any(sha(x)!=s for x,s in before.items()):raise ValueError('Reference input changed during audit')
    summary={'status':'completed_source_grounded_lexical_reference_audit','nodes':len(node_rows),'typed_edges':len(edge_rows),
        'nodes_with_any_reference_alias':sum(bool(r['reference_matches']) for r in node_rows),
        'typed_edges_with_any_reference_pair':sum(bool(r['reference_pair_matches']) for r in edge_rows),
        'reference_direction_counts_nonexclusive':dict(counts),'system_hint_node_counts_nonexclusive':dict(systems),
        'source_input_sha256':before,'script_sha256':sha(__file__),'command':[sys.executable,*sys.argv],
        'source_graph_modified':False,'expert_system_accuracy':None,'physical_causality_accuracy':None,
        'limitations':catalog['limitations'],'elapsed_seconds':time.perf_counter()-beg}
    dump(a.output/'analysis_summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
