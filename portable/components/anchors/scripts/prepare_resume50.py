"""Prepare exact event-description mentions offline; never call a text model.

Uses an explicit bilingual noun vocabulary as an extraction aid. This is
source-bound lexical preparation, not a claim of per-event semantic review.
"""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from frame_anchors.common import atomic,digest,read,sha
from frame_anchors.contract import plan
from frame_anchors.description_scope import SCHEMA

spec=importlib.util.spec_from_file_location('offline_vocab',Path(__file__).with_name('prepare_noqwen_selections.py'))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
ZH={**base.NOUNS,
 '笔记本电脑屏幕':'laptop screen','笔记本屏幕':'laptop screen','笔记本电脑键盘':'laptop keyboard',
 '笔记本键盘':'laptop keyboard','笔记本顶盖':'laptop lid','笔记本上盖':'laptop lid',
 '台灯底座':'lamp base','灯罩':'lampshade','灯泡':'light bulb','机箱':'computer tower',
 '画框':'picture frame','木框':'wooden frame','桌板':'tabletop','台面':'desk',
 '柜体':'cabinet','柜门':'cabinet door','抽屉把手':'drawer handle','抽屉拉手':'drawer handle',
 '织物':'cloth','布垫':'cloth pad','毛巾':'towel','床品':'bedding','床铺':'bed',
 '地面':'floor','地板':'floor','墙面':'wall','墙壁':'wall','木门':'wooden door',
 '小黑板':'small chalkboard','口罩':'face mask','方片':'square card','软垫':'cushion',
 '长条状物体':'long object','细长物体':'slender object','长方形物体':'rectangular object',
 '圆形物体':'round object','圆柱状物体':'cylindrical object','方形物体':'square object',
 '物体':'object','物件':'object','小件':'small object','设备':'device','吊牌':'tag','耳机':'headphones'}
EN_TEXT='''bedside table|nightstand|table surface|desk surface|tabletop|countertop|desk lamp|lamp base|lamp shade|lampshade|light bulb|bulb|lamp|laptop keyboard|laptop screen|laptop lid|laptop cover|laptop|notebook|computer monitor|monitor stand|monitor base|monitor|computer mouse|mouse pad|mousepad|mouse|keyboard|computer tower|computer case|tablet|smartphone|mobile phone|phone|screen|trackpad|picture frame|photo frame|framed picture|framed card|framed sign|framed object|wooden frame|desk calendar|standing calendar|calendar|coffee cup|paper cup|cup|mug|tissue box|tissue packet|tissue pack|tissue package|tissues|pen holder|pencil holder|pen basket|pen-like object|marker pen|marker|pens|pencils|pen|wicker basket|woven basket|wicker tray|woven tray|wire basket|basket|plastic storage bin|plastic bin|storage bin|storage box|storage container|bin|cardboard box|tool case|toolbox|tool box|cabinet drawer|drawer handle|drawer unit|shelving unit|shelf unit|storage cabinet|cabinet|drawer|shelves|shelf|table|desk|fire extinguisher box|fire extinguisher cabinet|fire extinguisher|power strip|power adapter|power outlet|electric drill|power drill|drill|plastic bag|paper bag|package|packet|packaging|box|case|bowl|tray|plate|tape roll|roll of tape|tape|cable|cord|remote control|remote|calculator|stapler|scissors|brush|broom|dustpan|bottle|canister|container|cylinder|potted plant|plant|vase|headphones|usb drive|usb stick|access card|key card|tag|card|bed linens|linens|bedding|bed sheet|duvet|pillow|cushion|sofa|armchair|chair|bed|headboard|pouf|floor|wall|door|window|curtain|carpet|cloth|towel|slippers|shopping cart|trolley|wire rack|rack|book|folder|notepad|paper|document|sponge|plush toy|robot|exercise equipment|refrigerator|chalkboard|blackboard|wrench|screwdriver|tool|tube|stick|rod|card reader|keypad|disc|lid|ornament|toy|rectangular object|cylindrical object|round object|square object|flat object|object|device|item'''
EN={x:x for x in EN_TEXT.split('|')}
EN.update({'nightstand':'bedside table','table surface':'table','desk surface':'desk','woven basket':'wicker basket',
 'framed picture':'picture frame','framed card':'picture frame','framed sign':'picture frame','framed object':'picture frame',
 'photo frame':'picture frame','computer monitor':'monitor','computer mouse':'mouse','monitor base':'monitor stand',
 'computer case':'computer tower','tissue pack':'tissue packet','tissue package':'tissue packet','marker pen':'marker',
 'plastic storage bin':'storage bin','plastic bin':'storage bin','toolbox':'tool case','tool box':'tool case',
 'remote':'remote control','usb drive':'usb flash drive','usb stick':'usb flash drive','linens':'bedding','bed linens':'bedding',
 'shelves':'shelf','lamp shade':'lampshade','bulb':'light bulb','drill':'power drill'})
COLORS={**base.COLORS,'浅色':'light-colored','深色':'dark','金色':'gold','半透明':'translucent'}
QUAL={**base.SHAPES,'小型':'small','大型':'large','小':'small','大':'large','编织':'woven','塑料':'plastic',
 '金属':'metal','左侧':'left','右侧':'right','上层':'upper','下层':'lower','底层':'bottom','中层':'middle'}
EQ=set('white black silver grey gray blue red yellow orange brown purple green gold golden beige pink dark light clear transparent translucent wooden metal metallic plastic wicker woven wire paper cardboard rectangular square round circular cylindrical oval flat long slender thin small large smaller larger miniature mini tiny folded open closed lidded spiral-bound spiral standing left right leftmost rightmost upper lower middle bottom top central adjacent'.split())
COLOR_WORDS=set('white black silver gray blue red yellow orange brown purple green gold beige pink dark'.split())
POS=set('left right leftmost rightmost upper lower middle bottom top'.split())
GENERIC={'object','device','item','small object'}
FAMILIES={'desk':'table','desk surface':'table','table surface':'table','tabletop':'table','bedside table':'table','wicker basket':'basket','wire basket':'basket',
 'coffee cup':'cup','paper cup':'cup','desk lamp':'lamp','storage bin':'bin','storage box':'box','cardboard box':'box',
 'tissue box':'tissue package','tissue packet':'tissue package','tissue':'tissue package','wicker tray':'tray'}
ZHRE=re.compile('|'.join(re.escape(x) for x in sorted(ZH,key=len,reverse=True)))
ZHQR=re.compile('(?:'+'|'.join(re.escape(x) for x in sorted(set(COLORS)|set(QUAL),key=len,reverse=True))+')+$')
ZHTOK=re.compile('|'.join(re.escape(x) for x in sorted(set(COLORS)|set(QUAL),key=len,reverse=True)))
ENRE=re.compile(r'(?<![a-z])(?:'+'|'.join(re.escape(x)+('(?:es|s)?' if not x.endswith('s') else '') for x in sorted(EN,key=len,reverse=True))+r')(?![a-z])',re.I)

def positive(text,a,b):
    if any(m.start()<=a<m.end() for m in re.finditer(r'【[^】]*】|\[[^\]]*\]',text)):return False
    before=re.split(r'[。；;.!?，,]',text[:a])[-1][-90:]
    if re.search(r'(?:未见|未发现|没有看到|没看到|未观察到|看不到|未找到|无法确认|不能确认|是否有|有没有|寻找)[^。；，,]{0,22}$',before):return False
    if re.search(r'\b(?:no|without|absence of|searching for|looking for|to find|search for)\b[^,.;]{0,55}$',before,re.I):return False
    if re.search(r'\b(?:unclear (?:if|whether)|cannot confirm|uncertain (?:if|whether)|not sure (?:if|whether))\b[^,.;]{0,55}$',before,re.I):return False
    if re.search(r'(?:image|picture|graphic|illustration|drawing|depiction)\s+of\s+(?:an?\s+)?$',before,re.I):return False
    if re.match(r'\s+(?:image|picture|graphic|illustration|drawing|depiction)\b',text[b:],re.I):return False
    if re.match(r"\s+(?:is|are|was|were)\s+(?:not|never|nowhere)\s+(?:clearly\s+)?(?:visible|seen|identified|found|present)",text[b:],re.I):return False
    return True

def mentions(text):
    out=[]
    for m in ZHRE.finditer(text):
        q=ZHQR.search(text[max(0,m.start()-18):m.start()]);a=m.start()-(len(q.group()) if q else 0)
        quals=[]
        for token in ZHTOK.finditer(text[a:m.start()]):quals.extend((COLORS|QUAL)[token.group()].split())
        noun=ZH[m.group()]
        if noun in GENERIC and not (set(quals)&COLOR_WORDS or set(quals)&{'rectangular','square','round','circular','cylindrical','oval','flat','long','slender'}):continue
        if positive(text,a,m.end()):out.append({'label':text[a:m.end()],'noun':noun,'q':list(dict.fromkeys(quals)),'start':a,'end':m.end()})
    for m in ENRE.finditer(text):
        raw=m.group().lower();term=next((x for x in (raw,raw[:-1],raw[:-2]) if x in EN),None)
        if term is None:continue
        a=m.start();quals=[]
        for w in reversed(list(re.finditer(r'[a-zA-Z]+(?:-[a-zA-Z]+)*',text[:a]))[-7:]):
            if text[w.end():a].strip() not in {'',','} or w.group().lower() not in EQ:break
            a=w.start();quals.insert(0,w.group().lower().replace('grey','gray'))
        noun=EN[term]
        if noun in GENERIC and not (set(quals)&COLOR_WORDS or set(quals)&{'rectangular','square','round','circular','cylindrical','oval','flat','long','slender'}):continue
        if positive(text,a,m.end()):out.append({'label':text[a:m.end()],'noun':noun,'q':quals,'start':a,'end':m.end()})
    return out

def sig(m):
    return (FAMILIES.get(m['noun'],m['noun']),frozenset(set(m['q'])&COLOR_WORDS),frozenset(set(m['q'])&POS))
def fits(a,b):
    fa,ca,pa=sig(a);fb,cb,pb=sig(b)
    return fa==fb and (not ca or not cb or ca==cb) and (not pa or not pb or pa==pb)
def prompt(m):
    return ' '.join(list(dict.fromkeys(x for x in m['q'] if x not in POS and x not in m['noun'].split()))+[m['noun']])

def build(doc,event):
    fs=base.fields_for(event);all_m=[];refs=[];groups=[];warnings=[]
    catalog=doc['search_event_states']['entity_reference_catalog']
    local={f['entity_ref'] for s in event['states'] for f in s.get('focus_object_states',[])}
    for kind,ref,label in [('event_entity',r,catalog[r]['label']) for r in sorted(local)]+[('scene_object',o['id'],o['identity']['name']) for o in doc['scene_memory']['objects']]:
        ms=mentions(label)
        if ms:refs.append({'kind':kind,'ref':ref,'label':label,'m':max(ms,key=lambda x:x['end']-x['start'])})
    for path,text in fs:
        for m in mentions(text):all_m.append({**m,'ev':{'path':path,'quote':text[m['start']:m['end']]}})
    for m in sorted(all_m,key=lambda x:(-len(x['q']),-len(x['label']))):
        matches=[g for g in groups if fits(g['m'],m)]
        exact=[g for g in matches if sig(g['m'])==sig(m)]
        if len(exact)==1:matches=exact
        if len(matches)==1:
            if m['ev'] not in matches[0]['ev']:matches[0]['ev'].append(m['ev'])
        elif len(matches)>1:warnings.append({'type':'ambiguous_generic_antecedent','evidence':m['ev']})
        else:groups.append({'m':m,'ev':[m['ev']]})
    candidates=[];used=set()
    for g in groups:
        m=g['m'];matched=[r for r in refs if fits(r['m'],m)]
        exact=[r for r in matched if r['label'].lower()==m['label'].lower()]
        if exact:matched=exact
        local_matches=[r for r in matched if r['kind']=='event_entity']
        if local_matches:matched=local_matches
        if len(matched)==1 and (matched[0]['kind'],matched[0]['ref']) not in used:
            r=matched[0];kind=r['kind'];ref=r['ref'];label=r['label'];cid=('entity:' if kind=='event_entity' else 'scene:')+ref;used.add((kind,ref))
        else:
            kind='description_mention';ref=None;label=m['label']
            cid='mention:'+event['event_id']+':'+digest({'noun':m['noun'],'q':m['q']})[:12]
            if matched:warnings.append({'type':'ambiguous_reference','label':label,'possible_refs':[r['ref'] for r in matched]})
        existing=next((c for c in candidates if c['candidate_id']==cid),None)
        if existing:
            for ev in g['ev']:
                if ev not in existing['description_evidence']:existing['description_evidence'].append(ev)
        else:candidates.append({'candidate_id':cid,'reference_kind':kind,'source_ref':ref,'label':label,'prompt':prompt(m),
            'role':'description_mentioned_object','description_evidence':g['ev']})
    return {'candidates':candidates,'excluded_mentions':[],'selection_warnings':warnings}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--ready',action='store_true');args=ap.parse_args()
    root=args.root.resolve();inventory=read(root/'input_inventory.json');state=read(root/'batch_state.json') if (root/'batch_state.json').exists() else {'completed':{}}
    report={'records':0,'events':0,'candidates':0,'errors':{},'empty_events':[],'text_model':None,'semantic_human_reviewed':False,'items':[]}
    for row in inventory['records']:
        key=row['group']+'/'+row['record_id']
        if key in state['completed']:continue
        try:
            doc=read(row['source_json'])
            if sha(row['source_json'])!=row['source_sha256']:raise ValueError('Source JSON changed')
            mapping={'schema':SCHEMA,'source_document_digest':digest(doc),'policy':{
                'selection_review_status':'source_bound_lexicon_not_human_semantic_reviewed',
                'scope':'event_description_union_in_each_frame','text_model':None,
                'selection_method':'assistant-authored bilingual noun vocabulary and source reference matching; no per-event semantic review',
                'ready_for_grounding':args.ready},'events':{e['event_id']:build(doc,e) for e in doc['search_event_states']['events']}}
            plan(doc,mapping,'event_description',verify_images=False)
            for eid,content in mapping['events'].items():
                if not content['candidates']:report['empty_events'].append(eid);mapping['policy']['ready_for_grounding']=False
                report['events']+=1;report['candidates']+=len(content['candidates'])
                report['items'].append({'record':key,'event_id':eid,'prompts':[c['prompt'] for c in content['candidates']]})
            atomic(root/'selections'/row['group']/(row['record_id']+'.json'),mapping);report['records']+=1
        except Exception as exc:report['errors'][key]=str(exc)
    atomic(root/'selection_preparation_report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='items'},ensure_ascii=False))
if __name__=='__main__':main()
