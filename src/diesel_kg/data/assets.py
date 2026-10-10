"""Prepare pinned official BERT/HFL assets with exact SHA256 verification.

Only public model files are downloaded. Existing mismatched files are rejected;
corpus or experiment data is never transmitted. --verify-only is read-only
except for the explicitly requested verification manifest.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ASSETS={
 'bert-base-chinese':{'repository':'google-bert/bert-base-chinese','revision':'8f23c25b06e129b6c986331a13d8d025a92cf0ea','files':{
  'config.json':'88af762cfb15fa9ce183c84dcbfdc9d635196ba635dfe81f265fdd21d67f4c4c',
  'vocab.txt':'45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c',
  'tokenizer_config.json':'0f6d13e6f4da6f9e24f22ada6bc3be571123d858d7c0c05a8a7cd55a9c23c2e8',
  'model.safetensors':'3404a1ffd8da507042e8161013ba2a4fc49858b4e3f8fbf5ce5724f94883aec3'}},
 'chinese-roberta-wwm-ext':{'repository':'hfl/chinese-roberta-wwm-ext','revision':'5c58d0b8ec1d9014354d691c538661bf00bfdb44','files':{
  'config.json':'61609babfbada201546297d62f7f3a642d8fd2f1cefec5571c6cbd1e2e2132d9',
  'vocab.txt':'45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c',
  'tokenizer_config.json':'61785aeaba176fba6d6489f27dccfd2ddee6aee2af0e590451cab7d8b57e0874',
  'special_tokens_map.json':'303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3',
  'pytorch_model.bin':'1ded5a5a1c7841dee6e47942f7b5bf2bcf6f73ff19197580f852f7f638f86b35'}}}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''): digest.update(b)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--assets-dir',type=Path,required=True)
    p.add_argument('--verify-only',action='store_true'); p.add_argument('--manifest',type=Path)
    p.add_argument('--proxy'); a=p.parse_args(); records=[]
    proxies={'http':a.proxy,'https':a.proxy} if a.proxy else urllib.request.getproxies()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler(proxies))
    for name,asset in ASSETS.items():
        directory=a.assets_dir/'tokenizers'/name; files=[]
        for filename,expected in asset['files'].items():
            dest=directory/filename; url=f'https://huggingface.co/{asset["repository"]}/resolve/{asset["revision"]}/{filename}?download=true'
            if dest.exists():
                if sha(dest)!=expected: raise ValueError(f'Existing asset has a different SHA256: {dest}')
            elif a.verify_only: raise FileNotFoundError(dest)
            else:
                directory.mkdir(parents=True,exist_ok=True); temporary=directory/(filename+'.partial')
                with opener.open(url,timeout=120) as response, temporary.open('wb') as f:
                    for block in iter(lambda:response.read(4194304),b''): f.write(block)
                if sha(temporary)!=expected:
                    temporary.unlink(); raise ValueError(f'Download SHA256 mismatch: {filename}')
                temporary.replace(dest)
            item={'name':filename,'sha256':expected,'bytes':dest.stat().st_size,'url':url}; files.append(item)
            records.append({'path':str(dest.resolve()),**item})
        if not a.verify_only:
            manifest=directory/'asset_manifest.json'
            value={'repository':asset['repository'],'revision':asset['revision'],'files':files}
            if not manifest.exists(): manifest.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    result={'status':'verified','assets':records,'corpus_transmitted':False}
    if a.manifest:
        a.manifest.parent.mkdir(parents=True,exist_ok=True); a.manifest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'verified','files':len(records)},ensure_ascii=False))


if __name__=='__main__': main()
