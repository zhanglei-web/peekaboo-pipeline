#!/usr/bin/env python3
"""Check source integrity, portable imports, links, and publishable file boundaries."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
LOCK=ROOT/'components/annotation/dependencies.lock.json'

def source_files():
    return sorted(p for directory in ['components','scripts','configs'] for p in (ROOT/directory).rglob('*')
                  if p.is_file() and p.suffix in {'.py','.json','.yaml','.txt'} and '__pycache__' not in p.parts and not p.name.endswith('.local.json') and p!=LOCK)

def build_lock():
    return {'version':'portable-source-v1','code':[{'path':str(p.relative_to(ROOT)),
            'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in source_files()]}

def check():
    errors=[];count=0
    expected=json.loads(LOCK.read_text())
    if expected!=build_lock():errors.append('Source dependency lock differs; review edits before refreshing the lock')
    for p in ROOT.rglob('*'):
        rel=p.relative_to(ROOT)
        if any(x in {'.git','__pycache__','.pytest_cache','runs','.venv','data','models','external','legacy','logs'} for x in rel.parts) or not p.is_file() or p.name.endswith('.local.json'):continue
        if p.suffix.lower() in {'.mkv','.mp4','.pt','.pth','.gguf','.safetensors','.pem','.key'}:errors.append('Unexpected binary/credential artifact: '+str(rel))
        if p.name=='.env' or '.secrets' in rel.parts:errors.append('Credential file: '+str(rel))
        if p.stat().st_size>5*1024*1024:errors.append('Unexpected large file: '+str(rel))
        try:s=p.read_text()
        except UnicodeDecodeError:continue
        if p.suffix=='.py':
            count+=1
            try:ast.parse(s,filename=str(rel))
            except SyntaxError as exc:errors.append(str(exc))
        if re.search(r'/home/(?!user/)[A-Za-z0-9_-]+/',s):errors.append('Author-specific path: '+str(rel))
        if re.search(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|sk-[A-Za-z0-9]{24,}',s):errors.append('Possible credential: '+str(rel))
        if p.suffix=='.md':
            for target in re.findall(r'\]\(([^)]+)\)',s):
                if target.startswith(('https:','http:','#','<')):continue
                if not (p.parent/target.split('#')[0]).exists():errors.append(f'Broken link {rel}: {target}')
    return {'status':'passed' if not errors else 'failed','python_files':count,'errors':errors,
            'scope':'source hashes, Python syntax, relative documentation links, publication artifact checks; no GPU/model inference'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write-lock',action='store_true');args=p.parse_args()
    if args.write_lock:LOCK.write_text(json.dumps(build_lock(),ensure_ascii=False,indent=2)+'\n')
    result=check();print(json.dumps(result,ensure_ascii=False,indent=2));sys.exit(result['status']!='passed')
