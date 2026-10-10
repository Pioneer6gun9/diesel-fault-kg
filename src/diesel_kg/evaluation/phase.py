"""Audit actual occurrence candidates against lexical case phases and direction.

These frozen rules produce auditable candidate subsets, never reviewed gold.
Offsets are half-open Unicode character positions. Original predictions stay intact.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

DIAGNOSIS = ('检查', '拆检', '检测', '排查', '诊断', '发现', '确认', '测量')
REPAIR = ('更换', '修复', '维修', '检修', '清洗', '清理', '调整', '紧固', '校准', '校正', '疏通', '重新安装')
RECOVERY = ('恢复', '消除', '正常', '解决', '顺畅', '缓解')
NOMINAL_REPAIR = ('清洗机', '清洗剂', '清洗设备', '调整器', '修复剂')
NEGATIONS = ('并不是', '并非', '不是', '未能', '无法', '不能', '没有', '尚未', '不再', '未', '不', '无', '非')
CAUSAL = {'导致', '由…引起', '影响'}
META = {'需…维修', '通过…检测', '适用于'}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4194304), b''): h.update(b)
    return h.hexdigest()

def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(path, rows):
    with Path(path).open('w') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')

def hits(text, words, nominal=False):
    found = []
    for word in words:
        for m in re.finditer(re.escape(word), text):
            left = text[max(0,m.start()-8):m.start()]
            neg = [n for n in NEGATIONS if left.endswith(n)]
            if word in RECOVERY:
                neg += [n for n in NEGATIONS if any(left.endswith(n+w) for w in ('恢复','消除'))]
            is_nominal = nominal and any(text.startswith(n,m.start()) for n in NOMINAL_REPAIR)
            found.append({'word':word,'start':m.start(),'end':m.end(),
                          'negation_hints':neg,'nominal_hint':is_nominal})
    return sorted(found,key=lambda x:(x['start'],x['end'],x['word']))

def case_phases(case):
    text = case['text']
    repairs = hits(text,REPAIR,True)
    valid = [m for m in repairs if not m['nominal_hint'] and not m['negation_hints']]
    diagnostic = hits(text,DIAGNOSIS)
    return {'case_id':case['case_id'],'text':text,'repair_markers':repairs,
            'diagnostic_markers':diagnostic,'recovery_markers':hits(text,RECOVERY),
            'first_repair_start':min((m['start'] for m in valid),default=None),
            'first_diagnostic_start':min((m['start'] for m in diagnostic if not m['negation_hints']),default=None),
            'phase_evidence_kind':'fixed_lexical_hints_not_observed_event_times'}

def mention_phase(mention, phases):
    repair = phases['first_repair_start']; diagnostic = phases['first_diagnostic_start']
    rec = [m for m in phases['recovery_markers'] if not m['negation_hints']
           and m['start'] < mention['end'] and mention['start'] < m['end']]
    if rec: phase = 'recovery_hint'
    elif repair is not None and mention['start'] >= repair: phase = 'repair_or_post_repair_hint'
    elif diagnostic is not None and mention['start'] >= diagnostic: phase = 'diagnostic_context_hint'
    else: phase = 'presentation_or_unclassified_hint'
    return {'phase':phase,'recovery_markers':rec,
            'at_or_after_repair':repair is not None and mention['start'] >= repair}

def validate_mention(m,text,inventory):
    if not (isinstance(m['start'],int) and isinstance(m['end'],int)
            and 0 <= m['start'] < m['end'] <= len(text)):
        raise ValueError('Invalid occurrence offset')
    if text[m['start']:m['end']] != m['text']: raise ValueError('Mention text/offset mismatch')
    saved = inventory.get(m['occurrence_id'])
    if saved is None or any(saved[k] != m[k] for k in ('text','type','start','end')):
        raise ValueError('Occurrence is not bound to actual saved NER')

def explicit_direction(triple,text):
    """Restrict affirmative connectors to the same punctuation clause.

    Still weak lexical support: attachment, negation, causality and correctness
    require independent adjudication; absence is unknown, never a negative label.
    """
    if triple['relation'] not in CAUSAL: return []
    h,t=triple['head'],triple['tail']
    cause,effect=(t,h) if triple['relation']=='由…引起' else (h,t)
    stop=re.compile(r'[，,。；;！？!?\n]')
    clues=[]
    for m in hits(text,('导致','引起','造成','致使','影响')):
        if m['negation_hints']: continue
        if cause['end'] <= m['start'] and m['end'] <= effect['start']:
            middle=text[cause['end']:effect['start']]
            if not stop.search(middle):
                clues.append({**m,'direction':'cause_to_effect','attachment':'same_clause_between_mentions'})
    # "effect由cause引起": bind both the opening and closing markers.
    if effect['end'] <= cause['start']:
        for m in hits(text,('由',)):
            if m['negation_hints']: continue
            if not (effect['end'] <= m['start'] < m['end'] <= cause['start']): continue
            for end in hits(text,('引起','造成','导致')):
                if cause['end'] <= end['start'] <= cause['end']+6 and not end['negation_hints']:
                    if not stop.search(text[effect['end']:end['end']]):
                        clues.append({**m,'closing_marker':end,'direction':'effect_due_to_cause',
                                      'attachment':'same_clause_bracketed_cause'})
    return clues

def classify(triple, phases):
    h,t=triple['head'],triple['tail']; hp=mention_phase(h,phases);tp=mention_phase(t,phases)
    flags=[]
    if triple['relation'] in META: flags.append('metadata_relation_not_propagation')
    if h['type']=='维修建议' or t['type']=='维修建议': flags.append('maintenance_entity_endpoint')
    if hp['at_or_after_repair'] or tp['at_or_after_repair']: flags.append('repair_or_post_repair_endpoint')
    if hp['recovery_markers'] or tp['recovery_markers']: flags.append('nonnegated_recovery_cue_endpoint')
    cues=explicit_direction(triple,phases['text'])
    if triple['relation'] in CAUSAL and not cues: flags.append('causal_candidate_without_local_direction_connector')
    # This sensitivity rule deliberately trades coverage for conservatism.
    conservative=not any(x in flags for x in ('metadata_relation_not_propagation',
        'maintenance_entity_endpoint','repair_or_post_repair_endpoint','nonnegated_recovery_cue_endpoint'))
    return {'head_phase':hp,'tail_phase':tp,'review_flags':flags,'local_direction_cues':cues,
            'retained_phase_conservative_candidate':conservative,
            'retained_local_connector_candidate':conservative and bool(cues),
            'human_reviewed':False,'relation_correctness':None,'causal_validity':None}

def subset_graph(source, triples, role):
    """Preserve source IDs and include only actual selected occurrence support."""
    lookup={(n['entity_type'],n['name']):n for n in source['nodes']}
    edge_lookup={(e['source'],e['relation_type'],e['target']):e for e in source['edges']}
    nodes={};edges={}
    for row in triples:
        pair=[];cid=row['case_id']
        for m in (row['head'],row['tail']):
            original=lookup[(m['type'],m['text'])];nid=original['id'];pair.append(nid)
            n=nodes.setdefault(nid,{**original,'case_ids':set(),'mention_ids':set()})
            n['case_ids'].add(cid);n['mention_ids'].add(m['occurrence_id'])
        orig=edge_lookup[(pair[0],row['relation'],pair[1])]
        e=edges.setdefault(orig['id'],{**orig,'case_ids':set(),'n_occurrence_observations':0,'probability_sum':0.})
        e['case_ids'].add(cid);e['n_occurrence_observations']+=1;e['probability_sum']+=row['probability']
    for n in nodes.values():
        n['case_ids']=sorted(n['case_ids']);n['mention_ids']=sorted(n['mention_ids'])
    for e in edges.values():
        e['case_ids']=sorted(e['case_ids']);e['mean_probability']=e.pop('probability_sum')/e['n_occurrence_observations']
    return {'nodes':list(nodes.values()),'edges':list(edges.values()),'schema':{
        **source['schema'],'role':role,'support_mode':'model_occurrences','human_reviewed':False,
        'rule_selected_candidate_not_gold':True,'relation_precision':None,'causal_validity':None}}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cases','entities','triples','graph','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    inputs={name:{'path':str(getattr(a,name).resolve()),'sha256':sha(getattr(a,name))} for name in ('cases','entities','triples','graph')}
    cases=read_jsonl(a.cases);entities=read_jsonl(a.entities);triples=read_jsonl(a.triples)
    cmap={r['case_id']:r for r in cases};emap={r['case_id']:r for r in entities}
    if len(cmap)!=len(cases) or len(emap)!=len(entities) or set(cmap)!=set(emap):raise ValueError('Case IDs inconsistent')
    inventories={};phases={}
    for cid,c in cmap.items():
        if emap[cid]['text']!=c['text']:raise ValueError('NER case text mismatch')
        inv={m['occurrence_id']:m for m in emap[cid]['entities']}
        if len(inv)!=len(emap[cid]['entities']):raise ValueError('Duplicate mention ID')
        for m in inv.values():validate_mention(m,c['text'],inv)
        inventories[cid]=inv;phases[cid]=case_phases(c)
    enriched=[];selected=[];connected=[];counts=Counter();seen=set();casecounts=Counter()
    for row in triples:
        cid=row['case_id'];text=cmap[cid]['text']
        for m in (row['head'],row['tail']):validate_mention(m,text,inventories[cid])
        key=(cid,row['head']['occurrence_id'],row['tail']['occurrence_id'])
        if key in seen:raise ValueError('Duplicate candidate observation')
        seen.add(key)
        if not math.isfinite(row['probability']) or not 0<=row['probability']<=1:raise ValueError('Invalid probability')
        audit=classify(row,phases[cid]);enriched.append({**row,'phase_direction_audit':audit})
        counts.update(audit['review_flags']);casecounts[cid]+=1
        if audit['retained_phase_conservative_candidate']:selected.append(row)
        if audit['retained_local_connector_candidate']:connected.append(row)
    source=json.loads(a.graph.read_text());reconstructed=subset_graph(source,triples,'unfiltered_recount')
    # Independently prove the exact full candidate baseline is recovered.
    def signature(g):
        return ({n['id']:(n['case_ids'],n['mention_ids']) for n in g['nodes']},
                {e['id']:(e['case_ids'],e['n_occurrence_observations'],round(e['mean_probability'],12)) for e in g['edges']})
    if signature(source)!=signature(reconstructed):raise ValueError('Complete candidate reconstruction mismatch')
    graphs={name:subset_graph(source,rows,name) for name,rows in [('phase_conservative_candidates',selected),('local_connector_candidates',connected)]}
    write_jsonl(a.output/'case_phase_evidence.jsonl',phases.values())
    write_jsonl(a.output/'all_candidate_phase_direction_audit.jsonl',enriched)
    write_jsonl(a.output/'phase_conservative_triples.jsonl',selected)
    write_jsonl(a.output/'local_connector_triples.jsonl',connected)
    for name,g in graphs.items():dump(a.output/(name+'_graph.json'),g)
    settings={'diagnosis':DIAGNOSIS,'repair':REPAIR,'recovery':RECOVERY,'nominal_exclusions':NOMINAL_REPAIR,
              'negation_hints':NEGATIONS,'causal_relations':sorted(CAUSAL),'metadata_relations':sorted(META),
              'selection':'predeclared lexical sensitivity; not tuned to labels, scores or paper targets'}
    for name,data in inputs.items():
        if sha(data['path'])!=data['sha256']:raise ValueError('Input changed during audit')
    output_counts={name:{'observations':len(rows),'cases':len({r['case_id'] for r in rows}),
        'nodes':len(graphs[name]['nodes']),'typed_edges':len(graphs[name]['edges'])}
        for name,rows in [('phase_conservative_candidates',selected),('local_connector_candidates',connected)]}
    summary={'status':'completed_lexical_evidence_candidate_audit','n_cases':len(cases),'n_mentions':sum(len(i) for i in inventories.values()),
        'n_original_candidate_observations':len(triples),'n_cases_with_candidates':len(casecounts),
        'review_flag_counts_nonexclusive':dict(counts),'candidate_subsets':output_counts,
        'independent_full_baseline_graph_reconstruction_passed':True,'source_predictions_modified':False,
        'relation_precision':None,'causal_accuracy':None,'human_reviewed':False,
        'limitations':['Lexical phases are text order hints, not observed time or verified recovery/intervention.',
            'Local connectors do not prove relation attachment, existence or physical causality.',
            'Conservative exclusion may lose valid cross-phase descriptions; it is a disclosed sensitivity comparison.',
            'Absence of a connector remains unknown; it is never turned into no_relation training gold.'],
        'settings':settings,'input_files':inputs,'script_sha256':sha(__file__),
        'elapsed_seconds':time.perf_counter()-start,'command':[sys.executable,*sys.argv]}
    dump(a.output/'analysis_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,allow_nan=False),flush=True)

if __name__=='__main__':main()
