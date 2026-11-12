"""Download the Transformers-format model with repository-provided SHA-256 checks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
import requests

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'repository.marker').is_file())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frame_anchors.common import atomic, locked, sha

REPO = 'IDEA-Research/grounding-dino-tiny'
REVISION = '2e311b914779978895827276d44f59285fc49f95'
BASE = 'https://modelscope.cn'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'models/frame_anchors/grounding-dino-tiny')
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    with locked(out/'.download.lock'):
        response = requests.get(f'{BASE}/api/v1/models/{REPO}/repo/files',
                                params={'Revision':REVISION,'Recursive':'true'}, timeout=(15,60))
        response.raise_for_status()
        payload = response.json()
        if payload.get('Code') != 200: raise RuntimeError(payload)
        files = [f for f in payload['Data']['Files'] if f['Type']=='blob' and
                 (f['Path'].endswith(('.json','.txt','.safetensors')) or f['Path']=='README.md')]
        names = {f['Path'] for f in files}
        if not {'config.json','model.safetensors','preprocessor_config.json','tokenizer_config.json','tokenizer.json'} <= names:
            raise RuntimeError('Incomplete repository file list')
        def download(f):
            name = f['Path']
            if Path(name).name != name: raise ValueError('Unexpected nested model file')
            path = out/name
            if path.exists():
                if path.stat().st_size==f['Size'] and sha(path)==f['Sha256']:
                    print('Verified existing: '+name, flush=True); return
                raise ValueError('Existing model file differs; choose another output directory: '+str(path))
            temporary = path.with_name(name+'.part')
            url=f'{BASE}/models/{REPO}/resolve/{REVISION}/{name}'
            print('Downloading: '+name, flush=True)
            for attempt in range(6):
                offset=temporary.stat().st_size if temporary.exists() else 0
                if offset==f['Size']:break
                if offset>f['Size']:raise ValueError('Oversized partial file: '+name)
                try:
                    headers={'Range':f'bytes={offset}-','Accept-Encoding':'identity'} if offset else {'Accept-Encoding':'identity'}
                    with requests.get(url,headers=headers,stream=True,timeout=(15,45)) as r:
                        r.raise_for_status()
                        if r.status_code==206:
                            if not r.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                                raise ValueError('Incorrect resume range: '+name)
                            mode='ab'
                        elif r.status_code==200:mode='wb'
                        else:raise ValueError('Unexpected download status: '+str(r.status_code))
                        with temporary.open(mode) as stream:
                            for block in r.iter_content(1024*1024):stream.write(block)
                    if temporary.stat().st_size==f['Size']:break
                except requests.RequestException as exc:
                    print(f'Retry {attempt+1}/6: {name}: {type(exc).__name__}',flush=True)
                    if attempt==5:raise
            if temporary.stat().st_size!=f['Size'] or sha(temporary)!=f['Sha256']:
                raise ValueError('Download size/hash mismatch: '+name)
            temporary.replace(path)
            print('Verified: '+name, flush=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(download,files))
        atomic(out/'download_receipt.json',{'source':BASE,'repository':REPO,'revision':REVISION,
              'verification':'size and SHA-256 against source repository metadata',
              'files':{f['Path']:{'bytes':f['Size'],'sha256':f['Sha256']} for f in files}})
        print('Model ready: '+str(out),flush=True)

if __name__=='__main__': main()
