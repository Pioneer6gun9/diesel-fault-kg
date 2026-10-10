"""Occurrence-level original RE with complete pair windows and honest filtering.

Keeps repeated surface mentions, uses the supplied interaction-MLP checkpoint,
and checks cached classification against its unchanged forward(). A nine-class
checkpoint never establishes relationship existence. No probabilities or graph
counts are adjusted to paper readouts.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import json
import os
from pathlib import Path
import time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import numpy as np
import torch
from diesel_kg.inference.paired import source_classes, locate_offsets
from diesel_kg.models.loading import load_tokenizer, file_sha256, DEFAULT_ONTOLOGY, relation_allowed
from diesel_kg.data.neo4j import stable_id


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def pair_window(offsets, head, tail, budget=126):
    """Full inclusive source-token spans plus a shared half-open token window."""
    hp=locate_offsets(offsets, int(head['start']), int(head['end']))
    tp=locate_offsets(offsets, int(tail['start']), int(tail['end']))
    if hp is None or tp is None:
        return None, 'mention_unaligned'
    lo=min(hp[0],tp[0]); hi=max(hp[1],tp[1])+1
    if hi-lo>budget:
        return None, 'pair_span_exceeds_model_context'
    left=max(0,lo-(budget-(hi-lo))//2)
    right=min(len(offsets),left+budget)
    left=max(0,right-budget)
    return (left,right,(hp[0]-left+1,hp[1]-left+1),
            (tp[0]-left+1,tp[1]-left+1)), None


def validate_mentions(row):
    ids=set()
    for n,e in enumerate(row['entities']):
        if not isinstance(e.get('start'), int) or not isinstance(e.get('end'), int) or not (0 <= e['start'] < e['end'] <= len(row['text'])):
            raise ValueError(f"Invalid half-open span {row['case_id']} {n}")
        if not e.get('type') or not e.get('text'):
            raise ValueError(f"Empty entity {row['case_id']} {n}")
        if row['text'][int(e['start']):int(e['end'])]!=e['text']:
            raise ValueError(f"Span mismatch {row['case_id']} {n}")
        e.setdefault('occurrence_id',stable_id('mention_', [row['case_id'],e['start'],e['end'],e['type']]))
        if e['occurrence_id'] in ids:
            raise ValueError('Duplicate occurrence ID')
        ids.add(e['occurrence_id'])


def infer(a):
    begun=time.perf_counter(); a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.threads); torch.manual_seed(42)
    rows=[json.loads(l) for l in a.entities.read_text().splitlines() if l.strip()]
    if a.case_limit: rows=rows[:a.case_limit]
    if len({r['case_id'] for r in rows})!=len(rows): raise ValueError('Duplicate case ID')
    for r in rows: validate_mentions(r)
    tok=load_tokenizer(a.tokenizer,default_do_lower_case=False)
    if tok.num_special_tokens_to_add(pair=False)!=2: raise ValueError('Expected BERT CLS/SEP')
    _,cls,class_hash=source_classes(a.source)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    relmap=saved['rel2id']; id2rel={int(v):k for k,v in relmap.items()}
    model=cls(str(a.pretrained),len(relmap),dropout_rate=.2)
    model.load_state_dict(saved['model_state_dict'],strict=True); del saved
    model.eval().to(a.device)
    existence_available='no_relation' in relmap
    counts=Counter(); rejects=Counter(); relations=Counter(); nodes={}; edges={}; checks=[]; case_counts=[]
    with gzip.open(a.output/'all_occurrence_pair_predictions.jsonl.gz','wt',encoding='utf8') as raw, (a.output/'qualified_triples.jsonl').open('w') as triples:
        for row in rows:
            entities=row['entities']; enc=tok(row['text'],add_special_tokens=False,return_offsets_mapping=True,truncation=False)
            offsets=enc['offset_mapping']; groups=defaultdict(list); skipped=Counter(); pair_count=0
            for i,h in enumerate(entities):
                for j,t in enumerate(entities):
                    if i==j: continue
                    pair_count+=1
                    win,reason=pair_window(offsets,h,t,a.max_length-2)
                    if reason: skipped[reason]+=1; continue
                    left,right,hp,tp=win; groups[left,right].append((i,j,hp,tp))
            allpred=[]; accepted=0; window_keys=list(groups)
            for first in range(0,len(window_keys),a.batch_size):
                keys=window_keys[first:first+a.batch_size]
                batch=tok.pad({'input_ids':[tok.build_inputs_with_special_tokens(enc['input_ids'][l:r]) for l,r in keys]},
                              padding='max_length',max_length=a.max_length,return_tensors='pt')
                ids=batch['input_ids'].to(a.device); masks=batch['attention_mask'].to(a.device)
                with torch.inference_mode(): hidden=model.bert(input_ids=ids,attention_mask=masks).last_hidden_state
                counts['encoder_windows']+=len(keys)
                for local,key in enumerate(keys):
                    observations=groups[key]; cache={}; cpu_hidden=hidden[local].cpu()
                    for start in range(0,len(observations),512):
                        subset=observations[start:start+512]; hh=[]; tt=[]
                        for i,j,hp,tp in subset:
                            for p in [hp,tp]:
                                if p not in cache: cache[p]=cpu_hidden[p[0]:p[1]+1].mean(0)
                            hh.append(cache[hp]); tt.append(cache[tp])
                        h=torch.stack(hh).to(a.device); t=torch.stack(tt).to(a.device)
                        with torch.inference_mode():
                            interacted=t+model.entity_interact(torch.cat([h,t],-1))
                            logits=model.classifier(model.dropout(torch.cat([hidden[local,0].expand(len(subset),-1),h,interacted],-1)))
                            if len(checks)<12:
                                i,j,hp,tp=subset[0]
                                actual=model(ids[local:local+1],masks[local:local+1],torch.tensor([hp],device=a.device),torch.tensor([tp],device=a.device))
                                err=float((actual[0]-logits[0]).abs().max().cpu())
                                if err>3e-5: raise AssertionError('Cache differs from unchanged source forward')
                                checks.append({'case_id':row['case_id'],'maximum_logit_difference':err,'atol':3e-5})
                            probs,labels=logits.softmax(-1).max(-1)
                        for (i,j,hp,tp),score,label in zip(subset,probs.cpu().tolist(),labels.cpu().tolist()):
                            head,tail=entities[i],entities[j]; rel=id2rel[label]
                            allpred.append([i,j,label,score,key[0],key[1]])
                            counts['classified_ordered_pairs']+=1; relations[rel]+=1
                            if rel=='no_relation': rejects['predicted_no_relation']+=1; continue
                            if min(len(head['text']),len(tail['text']))<a.min_entity_length: rejects['short_entity']+=1; continue
                            if score<a.confidence: rejects['uncalibrated_confidence_filter']+=1; continue
                            if not relation_allowed(DEFAULT_ONTOLOGY,rel,head['type'],tail['type']): rejects['type_qualification']+=1; continue
                            if (head['type'],head['text'])==(tail['type'],tail['text']): rejects['canonical_self_loop']+=1; continue
                            triple={'case_id':row['case_id'],'head':head,'tail':tail,'relation':rel,'probability':score,
                                    'token_window':[key[0],key[1]],'text_evidence':row['text'][offsets[key[0]][0]:offsets[key[1]-1][1]],
                                    'relationship_existence_modeled':existence_available,'causal_gold':False}
                            triples.write(json.dumps(triple,ensure_ascii=False)+'\n'); accepted+=1
                            pair=[]
                            for e in [head,tail]:
                                nid=stable_id('n_',[e['type'],e['text']]); pair.append(nid)
                                node=nodes.setdefault(nid,{'id':nid,'name':e['text'],'entity_type':e['type'],'analysis_eligible':True,'case_ids':set(),'mention_ids':set()})
                                node['case_ids'].add(row['case_id']); node['mention_ids'].add(e['occurrence_id'])
                            eid=stable_id('e_',[pair[0],rel,pair[1]])
                            edge=edges.setdefault(eid,{'id':eid,'source':pair[0],'target':pair[1],'relation_type':rel,'analysis_eligible':True,'case_ids':set(),'probabilities':[],'causal_gold':False})
                            edge['case_ids'].add(row['case_id']); edge['probabilities'].append(score)
            counts['processed_cases']+=1; counts['mentions']+=len(entities); counts['ordered_pairs']+=pair_count
            counts['qualified_observations']+=accepted; counts['cases_with_qualified_edges']+=bool(accepted)
            rejects.update(skipped)
            case_counts.append({'case_id':row['case_id'],'mentions':len(entities),'ordered_pairs':pair_count,'classified_pairs':len(allpred),'unclassified_pairs':sum(skipped.values()),'qualified_observations':accepted})
            raw.write(json.dumps({'case_id':row['case_id'],'entities':entities,'predictions_ij_relation_probability_token_window':allpred,'unclassified_reasons':dict(skipped)},ensure_ascii=False)+'\n')
            if counts['processed_cases']%50==0: print(json.dumps(dict(counts),ensure_ascii=False),flush=True)
    for n in nodes.values(): n['case_ids']=sorted(n['case_ids']); n['mention_ids']=sorted(n['mention_ids'])
    for e in edges.values(): e['case_ids']=sorted(e['case_ids']); e['mean_probability']=float(np.mean(e.pop('probabilities')))
    dump(a.output/'qualified_predicted_graph.json',{'nodes':list(nodes.values()),'edges':list(edges.values()),'schema':{'role':'model_qualified_candidates_not_causal_gold','relationship_existence_head':existence_available}})
    dump(a.output/'case_counts.json',case_counts)
    if counts['ordered_pairs']!=counts['classified_ordered_pairs']+sum(r['unclassified_pairs'] for r in case_counts): raise AssertionError('Pair accounting incomplete')
    summary={'status':'completed_with_quality_limits','counts':dict(counts),'rejection_counts':dict(rejects),'relation_counts':dict(relations),
             'qualified_graph_nodes':len(nodes),'qualified_graph_edges':len(edges),'elapsed_seconds':time.perf_counter()-begun,
             'encoder_cache_checks':checks,'source_class_hashes':class_hash,'all_occurrences_preserved':True,'pair_cap':None,
             'protocol':{'max_length':a.max_length,'confidence_threshold':a.confidence,'min_entity_length':a.min_entity_length,'device':a.device},
             'input_sha256':{str(p):file_sha256(p) for p in [a.entities,a.checkpoint,a.source]},
             'limits':['A nine-positive-class checkpoint does not establish relation existence.','Confidence thresholds are uncalibrated; this is candidate generation.','Pairs outside the 128-token joint window are counted as unclassified, never silently omitted.','Graph candidates are not independent causal/diagnostic gold.']}
    dump(a.output/'run_summary.json',summary); print(json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['entities','checkpoint','source','tokenizer','pretrained','output']: p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu'); p.add_argument('--threads',type=int,default=2)
    p.add_argument('--max-length',type=int,default=128); p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--confidence',type=float,default=.8); p.add_argument('--min-entity-length',type=int,default=2)
    p.add_argument('--case-limit',type=int)
    infer(p.parse_args())
if __name__=='__main__': main()
