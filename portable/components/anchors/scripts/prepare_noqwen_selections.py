"""Prepare event-description candidate maps without invoking any language model.

Selection is restricted to exact positive mentions found in the original event
description fields. Existing event/scene references are preferred when their
original labels are explicitly mentioned. A small fixed noun lexicon covers
description mentions absent from the input catalogs. This is a lexical aid,
not semantic or human visual review; all evidence remains source-bound.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frame_anchors.common import atomic, digest, read, sha
from frame_anchors.contract import plan
from frame_anchors.description_scope import SCHEMA


NOUNS = {
    "笔记本电脑": "laptop", "笔记本": "laptop", "鼠标垫": "mouse pad",
    "鼠标": "mouse", "键盘": "keyboard", "显示器底座": "monitor stand",
    "显示器支架": "monitor stand", "显示器": "monitor", "屏幕": "screen",
    "触控板": "trackpad", "主机": "computer tower", "电脑": "computer",
    "平板电脑": "tablet", "平板": "tablet", "手机": "phone",
    "U盘": "usb flash drive", "吊牌": "tag", "相框": "picture frame",
    "咖啡杯": "coffee cup", "纸杯": "paper cup", "杯子": "cup", "杯": "cup",
    "纸巾包": "tissue packet", "纸巾盒": "tissue box", "纸巾": "tissue",
    "笔筒": "pen holder", "台历": "desk calendar", "日历": "calendar",
    "台灯": "desk lamp", "灯": "lamp", "抽屉": "drawer", "床头柜": "bedside table",
    "柜子": "cabinet", "柜": "cabinet", "置物架": "shelving unit",
    "搁板": "shelf", "层板": "shelf", "桌面": "desk surface", "桌面板": "desk surface",
    "桌子": "table", "办公桌": "desk", "桌": "desk", "门": "door",
    "床": "bed", "枕头": "pillow", "靠垫": "cushion", "沙发": "sofa",
    "篮子": "basket", "篮筐": "basket", "编织篮": "wicker basket", "篮": "basket",
    "纸箱": "cardboard box", "纸盒": "paper box", "盒子": "box", "盒": "box",
    "包装袋": "package", "包装盒": "box", "包装": "package", "袋子": "bag",
    "胶带卷": "tape roll", "胶带": "tape", "线缆": "cable", "电线": "cable",
    "电源线": "power cable", "插座": "power strip", "遥控器": "remote control",
    "计算器": "calculator", "订书机": "stapler", "剪刀": "scissors",
    "刷子": "brush", "记号笔": "marker", "马克笔": "marker", "笔": "pen",
    "书本": "book", "书籍": "book", "书": "book", "纸张": "sheet of paper",
    "纸": "paper", "托盘": "tray", "盘子": "plate", "瓶子": "bottle",
    "瓶": "bottle", "容器": "container", "收纳箱": "storage bin",
    "收纳盒": "storage box", "绿植": "potted plant", "植物": "plant",
    "灭火器箱": "fire extinguisher box", "工具箱": "tool case", "工具盒": "tool box",
    "螺丝刀": "screwdriver", "电钻": "power drill", "扳手": "wrench",
    "购物车": "shopping cart", "插头": "plug", "相机": "camera",
}
COLORS = {"白色":"white", "黑色":"black", "银色":"silver", "灰色":"gray",
          "蓝色":"blue", "浅蓝色":"light blue", "浅蓝":"light blue", "深蓝色":"dark blue",
          "红色":"red", "黄色":"yellow", "橙色":"orange", "棕色":"brown",
          "紫色":"purple", "绿色":"green", "透明":"clear", "木色":"wooden", "木质":"wooden"}
SHAPES = {"长方形":"rectangular", "长方体":"rectangular", "长条":"long", "细长":"slender",
          "圆形":"round", "圆柱形":"cylindrical", "方形":"square", "扁平":"flat",
          "透明":"clear", "带盖":"lidded", "折叠":"folded", "打开的":"open"}
GENERIC = ("物品", "物体", "小物", "小件", "杂物", "东西", "区域", "空间", "范围", "内容物")
NEGATIVE_LEFT = re.compile(r"(?:未见|没有看到|没看到|未发现|没发现|未观察到|看不到|没有找到|未找到|无法确认(?:是否有|有没有)?|不能确认(?:是否有|有没有)?)\s*$")
ANATOMY = re.compile(r"(?:手|手指|手掌|手腕|手臂|胳膊|腿|脚|操作者|人|人物)$")
PREFIXES = sorted(set(COLORS) | set(SHAPES) | {"大号", "小号", "大型", "小型", "较小", "较大", "左侧", "右侧", "上层", "下层", "前方", "后方"}, key=len, reverse=True)


def fields_for(event):
    out=[]
    for key in ("event_summary", "search_region"):
        text=event.get(key)
        if isinstance(text,str) and text.strip(): out.append((key,text))
    for i,text in enumerate(event.get("observed_changes",[]) or []):
        if isinstance(text,str) and text.strip(): out.append((f"observed_changes/{i}",text))
    for i,state in enumerate(event["states"]):
        for key in ("state_description", "hand_action", "visibility_and_occlusion"):
            text=state.get(key)
            if isinstance(text,str) and text.strip(): out.append((f"states/{i}/{key}",text))
    return out


def without_goal(text):
    return re.sub(r"【[^】]*】", " ", text)


def is_positive(text, start, end):
    left=text[max(0,start-22):start]
    # A preceding negation directly governing this noun indicates it was not seen.
    if NEGATIVE_LEFT.search(left): return False
    return True


def occurrences(label, fields):
    found=[]
    if not label or len(label.strip()) < 2: return found
    for path,text in fields:
        clean=without_goal(text)
        flags=re.IGNORECASE if re.search(r"[A-Za-z]",label) else 0
        for match in re.finditer(re.escape(label),clean,flags):
            if is_positive(clean,match.start(),match.end()):
                found.append({"path":path,"quote":match.group()})
    return found


def phrase_prompt(label, fallback_term=None):
    raw=label.strip().lower()
    if re.fullmatch(r"[a-z0-9][a-z0-9 '\-/]*",raw) and len(raw)<=120:
        return raw.rstrip(".").strip()
    term=fallback_term
    if term is None:
        term=next((n for n in sorted(NOUNS,key=len,reverse=True) if n in label),None)
    if term is None: return None
    prefix=label[:label.rfind(term)]
    adjectives=[]
    for token,en in sorted(COLORS.items(),key=lambda x:len(x[0]),reverse=True):
        if token in prefix and en not in adjectives: adjectives.append(en)
    for token,en in sorted(SHAPES.items(),key=lambda x:len(x[0]),reverse=True):
        if token in prefix and en not in adjectives: adjectives.append(en)
    noun=NOUNS[term]
    prompt=" ".join(adjectives+[noun])
    prompt=re.sub(r"\s+"," ",prompt).strip()
    return prompt if re.fullmatch(r"[a-z0-9][a-z0-9 '\-/]*",prompt) and len(prompt)<=120 else None


def mention_label(text, start, end, term):
    start0=max(0,start-6); prefix=text[start0:start]
    chosen=""
    for token in PREFIXES:
        pos=prefix.rfind(token)
        if pos>=0 and pos+len(token)==len(prefix):
            chosen=prefix[pos:]+text[start:end]
            break
    return chosen or term


def build_event(event, doc):
    fields=fields_for(event); by_ref={}; local_refs=set()
    for state in event["states"]:
        for focus in state.get("focus_object_states",[]) or []:
            local_refs.add(focus["entity_ref"])
    catalog=doc["search_event_states"]["entity_reference_catalog"]
    refs=[]
    for ref,item in catalog.items():
        if ref.startswith(event["event_id"]+":"):
            refs.append(("event_entity",ref,item.get("label",""),item))
    scene={o["id"]:o for o in doc["scene_memory"]["objects"]}
    for oid,obj in scene.items():
        refs.append(("scene_object",oid,obj.get("identity",{}).get("name",""),obj))
    candidates=[]; ref_names=set(); matched_prompts=set()
    for kind,ref,label,_ in refs:
        ev=occurrences(label,fields)
        if not ev or label.endswith(GENERIC) or ANATOMY.search(label): continue
        prompt=phrase_prompt(label)
        if not prompt: continue
        # Prefer event-local references, and suppress catalog aliases with an identical label.
        identity=(kind,label.casefold())
        if identity in ref_names: continue
        ref_names.add(identity)
        cid=("entity:" if kind=="event_entity" else "scene:")+ref
        role="interaction_object" if kind=="event_entity" else "description_mentioned_object"
        candidates.append({"candidate_id":cid,"reference_kind":kind,"source_ref":ref,"label":label,
            "prompt":prompt,"role":role,"description_evidence":ev})
        matched_prompts.add(prompt)
    # Fixed noun vocabulary covers explicit mentions that have no existing JSON reference.
    for path,text in fields:
        clean=without_goal(text)
        for term in sorted(NOUNS,key=len,reverse=True):
            for m in re.finditer(re.escape(term),clean):
                if not is_positive(clean,m.start(),m.end()): continue
                label=mention_label(clean,m.start(),m.end(),term)
                prompt=phrase_prompt(label,term)
                if not prompt or prompt in matched_prompts: continue
                if any(x in label for x in GENERIC) or ANATOMY.search(label): continue
                # One object per detection phrase; source text is the evidence, not an identity claim.
                cid="mention:"+event["event_id"]+":"+digest({"label":label,"prompt":prompt})[:12]
                if cid not in by_ref:
                    by_ref[cid]={"candidate_id":cid,"reference_kind":"description_mention","source_ref":None,
                        "label":label,"prompt":prompt,"role":"description_mentioned_object",
                        "description_evidence":[]}
                evidence={"path":path,"quote":m.group()}
                if evidence not in by_ref[cid]["description_evidence"]: by_ref[cid]["description_evidence"].append(evidence)
                matched_prompts.add(prompt)
    candidates.extend(by_ref.values())
    # Merge duplicate mention prompts while retaining every distinct source location.
    merged=[]; mention_by_prompt={}
    for candidate in candidates:
        if candidate["reference_kind"]=="description_mention" and candidate["prompt"] in mention_by_prompt:
            target=mention_by_prompt[candidate["prompt"]]
            for ev in candidate["description_evidence"]:
                if ev not in target["description_evidence"]: target["description_evidence"].append(ev)
            continue
        merged.append(candidate)
        if candidate["reference_kind"]=="description_mention": mention_by_prompt[candidate["prompt"]]=candidate
    return {"candidates":merged,"excluded_mentions":[]}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-root",type=Path,default=Path("outputs/frame_anchor_batch_20261109"))
    ap.add_argument("--root",type=Path,default=Path("outputs/frame_anchor_batch_20261110_noqwen"))
    ap.add_argument("--limit",type=int)
    args=ap.parse_args(); source=args.source_root.resolve(); root=args.root.resolve()
    inventory=read(source/"input_inventory.json")
    if inventory.get("errors"): raise ValueError("Input inventory is not clean")
    root.mkdir(parents=True,exist_ok=True)
    target_inventory=root/"input_inventory.json"
    if not target_inventory.exists(): shutil.copyfile(source/"input_inventory.json",target_inventory)
    selected=inventory["records"][:args.limit] if args.limit else inventory["records"]
    report={"records":0,"events":0,"frames":0,"candidate_events_empty":0,"candidates":0,"untranslatable":0,"items":[]}
    for row in selected:
        key=row["group"]+"/"+row["record_id"]; doc=read(row["source_json"])
        if sha(row["source_json"])!=row["source_sha256"]: raise ValueError("Source JSON changed: "+key)
        mapping={"schema":SCHEMA,"source_document_digest":digest(doc),"policy":{
            "selection_review_status":"offline_lexical_selection_not_human_semantic_reviewed",
            "scope":"exact_positive_event_description_mentions_each_frame",
            "selection_method":"fixed local term/reference matching; no language model invoked"},"events":{}}
        for event in doc["search_event_states"]["events"]:
            content=build_event(event,doc);mapping["events"][event["event_id"]]=content
            report["events"]+=1;report["frames"]+=3;report["candidates"]+=len(content["candidates"])
            if not content["candidates"]:report["candidate_events_empty"]+=1
            report["items"].append({"record":key,"event_id":event["event_id"],"candidate_count":len(content["candidates"]),
                "prompts":[c["prompt"] for c in content["candidates"]]})
        plan(doc,mapping,"event_description",verify_images=False)
        out=root/"selections"/row["group"]/(row["record_id"]+".json")
        out.parent.mkdir(parents=True,exist_ok=True)
        atomic(out,mapping)
        report["records"]+=1
    report_path=root/"selection_preparation_report.json"
    atomic(report_path,report)
    print(json.dumps({k:v for k,v in report.items() if k!="items"},ensure_ascii=False,indent=2))
    print("REPORT",report_path)


if __name__=="__main__":main()
