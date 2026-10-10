"""CPU numerical replay of the already frozen grouped RE checkpoint.

Verifies complete dev selection and all saved final predictions. No fitting,
selection, new split, gold correction, or confidence threshold is performed.
This is deployment/checkpoint correctness verification, not another benchmark
used to select a model. Inputs and predictions must remain byte-identical.
"""
from __future__ import annotations
from diesel_kg.paths import module_path, resolve_recorded_path
import argparse
import json
from pathlib import Path
import sys
import time
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from diesel_kg.training.grouped_re import sha, dump, read_rows, source_classes, offset_locator, external_first_occurrence_locator, evaluate_model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--pretrained',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4);began=time.perf_counter()
    run=args.output/'grouped_training'
    execution=json.loads((run/'execution.json').read_text())
    assert execution['status']=='completed' and execution['exit_code']==0
    config=json.loads((run/'config.json').read_text());summary=json.loads((run/'summary.json').read_text())
    checkpoint=resolve_recorded_path(summary['checkpoint_path']);assert sha(checkpoint)==summary['checkpoint_sha256']
    assert sha(args.source)==config['execution']['source_sha256']
    asset_checks = {}
    removed_download_records = []
    for name,expected in config['pretrained_sha256'].items():
        path = args.pretrained / name
        if name == 'download_headers.txt' and not path.exists():
            removed_download_records.append({'name':name,'recorded_sha256':expected,
                'reason':'Removed HTTP download headers; no model weights, configuration or tokenizer content.'})
            continue
        assert sha(path) == expected, name
        asset_checks[name] = expected
    for name in ('train','dev','test'):
        manifest=json.loads((args.output/'split_manifest.json').read_text())
        assert sha(args.output/'grouped_data'/f'{name}.json')==manifest['splits'][name]['sha256']
    namespace,ast_sha=source_classes(args.source);assert ast_sha==config['class_ast_sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    rel2id=state['rel2id'];labels=[name for name,_ in sorted(rel2id.items(),key=lambda x:x[1])]
    assert labels==config['labels']
    model=namespace['RelationModel'](str(args.pretrained),len(labels),0.2)
    model.load_state_dict(state['model_state_dict'],strict=True);del state;model.eval()
    assert all(p.is_contiguous() for p in model.bert.parameters())
    tokenizer=BertTokenizerFast.from_pretrained(str(args.pretrained),local_files_only=True)
    strict_class=namespace['REDataset'];strict_class._locate_entity=offset_locator
    external_class=type('ExternalFirstOccurrenceDataset',(strict_class,),{'_locate_entity':external_first_occurrence_locator})
    result={'status':'running','sys_argv':sys.argv,'device':'cpu','script_sha256':sha(Path(__file__)),
        'evaluation_module_sha256':sha(module_path('diesel_kg.training.grouped_re')),
        'checkpoint_sha256':sha(checkpoint),'source_sha256':sha(args.source),'configuration_sha256':sha(run/'config.json'),
        'verified_pretrained_assets_sha256':asset_checks,
        'removed_download_records':removed_download_records,
        'scope':'Post-freeze CPU numerical replay only; no fitting or model selection. Test performance conclusions remain the one frozen final model.', 'checks':{}}
    for role,data_path,cls in [('dev_selection',args.output/'grouped_data/dev.json',strict_class),
                               ('group_heldout',args.output/'grouped_data/test.json',strict_class),
                               ('original225_previously_evaluated',args.output/'sealed_original225.json',external_class)]:
        dataset=cls(str(data_path),tokenizer,rel2id,128)
        predictions=evaluate_model(model,DataLoader(dataset,batch_size=64,shuffle=False,num_workers=0),labels,torch.device('cpu'))
        if role=='dev_selection':
            accuracy=sum(x['relation']==x['predicted_relation'] for x in predictions)/len(predictions)
            check={'samples':len(predictions),'cpu_accuracy_micro_f1':accuracy,'frozen_selected_dev_value':summary['best_dev_accuracy_micro_f1'],
                   'dev_selection_value_identical':accuracy==summary['best_dev_accuracy_micro_f1'],'data_sha256':sha(data_path)}
            result['checks'][role]=check
        else:
            prediction_path=run/(role+'_predictions.jsonl');saved=read_rows(prediction_path)
            assert sha(prediction_path)==summary['test'][role]['predictions_sha256'] and len(saved)==len(predictions)
            different=[i for i,(a,b) in enumerate(zip(predictions,saved)) if any(a[k]!=b[k] for k in ('relation','predicted_relation'))]
            result['checks'][role]={'samples':len(saved),'mismatched_indices':different,'all_predictions_identical':not different,
                'frozen_predictions_sha256':sha(prediction_path),'data_sha256':sha(data_path)}
        dump(run/'full_cpu_checkpoint_replay.json',result)
        print(json.dumps({'role':role,'check':result['checks'][role]}),flush=True)
    valid=result['checks']['dev_selection']['dev_selection_value_identical'] and all(result['checks'][r]['all_predictions_identical'] for r in ('group_heldout','original225_previously_evaluated'))
    result.update({'status':'passed' if valid else 'failed','elapsed_seconds':time.perf_counter()-began,'exit_code':0 if valid else 1,
                   'checkpoint_unchanged':sha(checkpoint)==summary['checkpoint_sha256']})
    dump(run/'full_cpu_checkpoint_replay.json',result)
    assert valid and result['checkpoint_unchanged'],'Frozen RE deployment replay mismatch; refuse this as an independently verified model.'


if __name__=='__main__':main()
