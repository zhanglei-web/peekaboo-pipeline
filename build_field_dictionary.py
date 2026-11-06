"""Generate a complete path/type/constraint index from the pinned output contract."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent

def types(n):
 if '$ref' in n:return n['$ref']
 if 'const' in n:return 'const '+json.dumps(n['const'],ensure_ascii=False)
 if 'enum' in n:return 'enum '+json.dumps(n['enum'],ensure_ascii=False)
 if 'anyOf' in n:return ' 或 '.join(types(x) for x in n['anyOf'])
 t=n.get('type','未限定');return '/'.join(t) if isinstance(t,list) else t

def constraints(n):
 parts=[]
 for k in ['minimum','maximum','minLength','maxLength','minItems','maxItems','uniqueItems','additionalProperties']:
  if k in n:
   v=n[k]
   if isinstance(v,dict):v='动态键值类型 '+types(v)
   parts.append(k+'='+str(v))
 if 'allOf' in n:parts.append('含条件约束，见机器schema（非region节点须有object_ref）')
 return '; '.join(parts) or '—'

def walk(n,path,required,rows):
 rows.append((path,'必有' if required else '可选',types(n),constraints(n)))
 for k,v in n.get('properties',{}).items():walk(v,path+'.'+k,k in n.get('required',[]),rows)
 if isinstance(n.get('items'),dict):walk(n['items'],path+'[]',True,rows)
 if isinstance(n.get('additionalProperties'),dict):walk(n['additionalProperties'],path+'.<动态键>',False,rows)
 for i,branch in enumerate(n.get('anyOf',[])):
  if branch.get('type')!='null':
   for k,v in branch.get('properties',{}).items():walk(v,path+'.'+k,k in branch.get('required',[]),rows)
   if isinstance(branch.get('items'),dict):walk(branch['items'],path+'[]',True,rows)

def main():
 s=json.loads((ROOT/'contracts/final_annotation.schema.json').read_text());lines=['# 11-4 完整字段字典','', '由版本内的JSON Schema生成；语义、来源与结果确认约定见 [OUTPUT_SPEC.md](OUTPUT_SPEC.md)。`必有`指父对象存在时该键必须存在；null是允许的值类型，不等于可以省略。数组行表示成员类型。`#/$defs/node`在末尾单列并递归用于children，避免无限展开。动态键主要用于实体/先验目录。','']
 for name,node in list(s['properties'].items())+[("$defs."+k,v) for k,v in s.get('$defs',{}).items()]:
  rows=[];walk(node,name,name in s.get('required',[]),rows);lines += ['## '+name,'','| 路径 | 必选性 | 类型/枚举 | 约束 |','|---|---|---|---|']
  for row in rows:lines.append('| '+' | '.join('`'+str(v).replace('|','\\|').replace('\n',' ')+'`' if i in [0,2] else str(v) for i,v in enumerate(row))+' |')
  lines.append('')
 (ROOT/'docs/FIELD_DICTIONARY.md').write_text('\n'.join(lines)+'\n')
 print('Wrote complete field dictionary',len(lines),'lines')
if __name__=='__main__':main()
