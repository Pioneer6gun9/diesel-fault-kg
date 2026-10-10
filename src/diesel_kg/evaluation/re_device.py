"""Compare the unchanged supplied RE forward on real occurrence spans across devices.

This bounded numerical check is not an accuracy evaluation. Run it only after
other work has released the requested accelerator.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from diesel_kg.inference.paired import source_classes
from diesel_kg.inference.relations import pair_window, validate_mentions
from diesel_kg.models.loading import file_sha256, load_tokenizer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','checkpoint','pretrained','tokenizer','entities','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',choices=['mps','cuda'],required=True)
    p.add_argument('--pairs',type=int,default=32)
    p.add_argument('--atol',type=float,default=3e-5)
    a=p.parse_args()
    torch.set_num_threads(4); torch.manual_seed(42)
    tokenizer=load_tokenizer(a.tokenizer,default_do_lower_case=False)
    examples=[]; identities=[]
    for line in a.entities.read_text().splitlines():
        row=json.loads(line); validate_mentions(row)
        encoded=tokenizer(row['text'],add_special_tokens=False,return_offsets_mapping=True,truncation=False)
        entities=row['entities']
        # One forward example per case prevents a long first case from filling
        # the entire check with nearly identical contexts.
        found=False
        for i,head in enumerate(entities):
            for j,tail in reversed(list(enumerate(entities))):
                if i==j: continue
                window,_=pair_window(encoded['offset_mapping'],head,tail,126)
                if window is None: continue
                left,right,hp,tp=window
                examples.append((tokenizer.build_inputs_with_special_tokens(encoded['input_ids'][left:right]),hp,tp))
                identities.append({'case_id':row['case_id'],'head':head['occurrence_id'],'tail':tail['occurrence_id']})
                found=True; break
            if found: break
        if len(examples)>=a.pairs: break
    if not examples: raise ValueError('No actual aligned occurrence pair is available for the device check')
    batch=tokenizer.pad({'input_ids':[x[0] for x in examples]},padding='max_length',max_length=128,return_tensors='pt')
    heads=torch.tensor([x[1] for x in examples]); tails=torch.tensor([x[2] for x in examples])
    _,cls,class_hash=source_classes(a.source)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    model=cls(str(a.pretrained),len(saved['rel2id']),dropout_rate=.2)
    model.load_state_dict(saved['model_state_dict'],strict=True); del saved
    model.eval()
    noncontiguous=[name for name,value in model.named_parameters() if not value.is_contiguous()]
    with torch.inference_mode():
        cpu=model(batch['input_ids'],batch['attention_mask'],heads,tails).cpu()
    model.to(a.device)
    with torch.inference_mode():
        accelerator=model(batch['input_ids'].to(a.device),batch['attention_mask'].to(a.device),
                          heads.to(a.device),tails.to(a.device)).cpu()
    finite=bool(torch.isfinite(cpu).all() and torch.isfinite(accelerator).all())
    difference=float((cpu-accelerator).abs().max()) if finite else None
    checks={'finite_logits':finite,
            'logits_within_atol':finite and difference<=a.atol,
            'all_argmax_equal':finite and bool(torch.equal(cpu.argmax(-1),accelerator.argmax(-1)))}
    result={'status':'passed' if all(checks.values()) else 'failed','checks':checks,
            'maximum_logit_difference':difference,'atol':a.atol,
            'accelerator_device':a.device,'actual_pairs':len(examples),'pair_identities':identities,
            'source_class_hashes':class_hash,'noncontiguous_parameter_names':noncontiguous,
            'input_sha256':{str(path.resolve()):file_sha256(path) for path in (a.source,a.checkpoint,a.entities)},
            'script_sha256':file_sha256(Path(__file__)),
            'scope':'Bounded device consistency check on actual inputs; not held-out model accuracy.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if not all(checks.values()): raise RuntimeError('CPU/accelerator RE forward disagree; use CPU inference')


if __name__=='__main__': main()
