from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import html
import json
import numpy as np
from .common import atomic, sha, inside, read, digest
from .contract import assert_preserved


def save_masks(path, masks):
    packed = np.packbits(masks.reshape(len(masks), -1), axis=1) if len(masks) else np.zeros((0,0),dtype=np.uint8)
    np.savez_compressed(path, packed=packed, shape=np.array(masks.shape, dtype=np.int64))


def load_masks(path):
    with np.load(path, allow_pickle=False) as data:
        shape = tuple(int(x) for x in data["shape"])
        if len(shape)!=3 or min(shape)<0 or shape[1]*shape[2]>100_000_000:
            raise ValueError("Invalid mask archive shape")
        if not shape[0]:
            return np.zeros(shape, dtype=bool)
        return np.unpackbits(data["packed"], axis=1, count=shape[1]*shape[2]).reshape(shape).astype(bool)


def overlay(image, result, masks, path, focus_only=False):
    canvas = image.convert("RGB").copy()
    arr = np.array(canvas)
    for i,obj in enumerate(result["objects"]):
        if focus_only and not obj["is_frame_focus"]:
            continue
        if obj["mask_index"] is not None:
            mask = masks[obj["mask_index"]]
            color = np.array([(53*i+70)%220+25,(97*i+30)%220+25,(137*i+80)%220+25])
            arr[mask] = (arr[mask]*.8 + color*.2).astype(np.uint8)
    canvas = Image.fromarray(arr)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(12,image.width//90))
    for i,obj in enumerate(result["objects"]):
        if focus_only and not obj["is_frame_focus"]:
            continue
        color = "#ffb000" if obj["review_flags"] else "#00d9a0"
        draw.rectangle(obj["bbox_xyxy_pixels"], outline=color, width=3)
        x,y = obj["anchor"]["pixel_xy"]
        draw.ellipse((x-5,y-5,x+5,y+5),fill=color,outline="black",width=1)
        text = f'{i+1}: {obj["source_candidate_id"].split(":")[-1]} {obj["detection_prompt"]}'
        tx,ty = obj["bbox_xyxy_pixels"][:2]
        tx=max(0,min(tx,image.width-draw.textlength(text,font=font)-4));ty=max(0,ty-font.size-4)
        bounds=draw.textbbox((tx,ty),text,font=font)
        draw.rectangle(bounds,fill="black");draw.text((tx,ty),text,fill=color,font=font)
    canvas.save(path,quality=92)


def make_review(out, frames, results):
    sections=[]
    audit_rows=[]
    for i,frame in enumerate(frames):
        result=results[frame["state_id"]]
        items=[]
        for j,obj in enumerate(result["objects"]):
            items.append(f'<tr><td>{j+1}</td><td>{html.escape(obj["source_candidate_id"])}</td>'
                         f'<td>{html.escape(obj["detection_prompt"])}</td><td>{obj["detection_score"]:.3f}</td>'
                         f'<td>{html.escape(obj["anchor"]["method"])}</td><td>{html.escape(", ".join(obj["review_flags"]))}</td></tr>')
            audit_rows.append({"state_id":frame["state_id"],"anchor_id":obj["anchor_id"],
                "source_candidate_id":obj["source_candidate_id"],"object_correct":None,
                "anchor_on_correct_object":None,"bbox_acceptable":None,"notes":""})
        missing=[x["candidate_id"] for x in result["candidate_outcomes"] if x["status"]=="not_detected"]
        evidence_details=''
        if frame.get('candidate_scope')=='event_description':
            lines=[c['prompt']+' ← '+ '; '.join(e['path']+': '+e['quote'] for e in c['description_evidence']) for c in frame['candidates']]
            evidence_details='<details><summary>本事件描述涉及的候选及原文出处</summary><pre>'+html.escape(chr(10).join(lines))+'</pre></details>'
        sections.append(f'<section><h2>{html.escape(frame["state_id"])}</h2>'
            +evidence_details+
            f'<p>原帧号 {frame["frame_index"]} · 点击切换全部候选与关注物体；黄色表示待复核，绿色仅表示几何匹配。</p>'
            f'<img class="all" src="frames/{i:03d}/overlay.jpg">'
            f'<img class="focus" style="display:none" src="frames/{i:03d}/focus.jpg">'
            f'<details><summary>定位条目与冲突</summary><table><tr><th>序号</th><th>原引用</th><th>提示词</th><th>分数</th><th>锚点方法</th><th>待复核原因</th></tr>{"".join(items)}</table></details>'
            f'<details><summary>未定位候选：{len(missing)}（不代表不存在）</summary><pre>{html.escape(chr(10).join(missing))}</pre></details></section>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>逐帧锚点试点复核</title><style>body{max-width:1300px;margin:32px auto;padding:0 20px;font:16px/1.6 sans-serif;background:#f5f7fa;color:#17212b}section{background:white;padding:20px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:12px}td,th{border:1px solid #ccc;padding:6px;text-align:left;overflow-wrap:anywhere}pre{white-space:pre-wrap}button{padding:10px 18px}header{position:sticky;top:0;background:#f5f7fa;padding:12px;z-index:1}</style><header><h1>现有JSON · 逐帧物体锚点</h1><p>自动标注，未经人工语义验收。大小为图像投影；既有ID只是候选引用，不宣称跨帧实体对应。</p><button onclick="document.querySelectorAll('.all,.focus').forEach(x=>x.style.display=x.style.display==='none'?'block':'none')">切换：全部候选 / 关注物体</button></header>'''+''.join(sections)+"</html>"
    (out/'review.html').write_text(page)
    atomic(out/'human_review_template.json',{"status":"not_reviewed","rows":audit_rows,
        "frame_recall_review":[{"state_id":f["state_id"],"missed_required_objects":None,"notes":""} for f in frames]})


def validate(out):
    out=Path(out).resolve();manifest=read(out/'run_manifest.json');original=read(manifest['config']['input_json'])
    if digest({k:v for k,v in manifest.items() if k!='manifest_key'})!=manifest['manifest_key']:
        raise ValueError('Manifest digest mismatch')
    if sha(manifest['config']['input_json'])!=manifest['input_sha256']:
        raise ValueError('Original JSON changed')
    if sha(manifest['config']['prompt_map'])!=manifest['prompt_map_sha256']:
        raise ValueError('Prompt map changed')
    from .contract import plan
    frames=plan(original,read(manifest['config']['prompt_map']),manifest['config']['scope'])
    enriched=read(out/'enriched.json');assert_preserved(original,enriched)
    sidecar=read(out/'anchors.sidecar.json')
    meta=enriched['frame_anchor_enrichment']
    if sidecar.get('metadata')!=meta or meta['manifest_key']!=manifest['manifest_key'] or meta['source_sha256']!=manifest['input_sha256']:
        raise ValueError('Sidecar/metadata provenance mismatch')
    if meta['semantic_human_reviewed'] or meta['physical_dimensions_available'] or meta['cross_frame_identity_verified']:
        raise ValueError('Unsupported semantic claim')
    states={s['state_id']:s for e in enriched['search_event_states']['events'] for s in e['states']}
    if set(sidecar['frames'])!=set(states):raise ValueError('Sidecar frame coverage mismatch')
    counts=Counter();flags=Counter();focus=Counter()
    for i,frame in enumerate(frames):
        folder=out/'frames'/f'{i:03d}';receipt=read(folder/'receipt.json')
        if receipt['manifest_key']!=manifest['manifest_key'] or receipt['state_id']!=frame['state_id']:
            raise ValueError('Frame receipt belongs to another run/state')
        required={str((folder/name).relative_to(out)) for name in ['anchors.json','masks.npz','detections.json','timing.json','overlay.jpg','focus.jpg']}
        if not required<=set(receipt['files']):raise ValueError('Incomplete frame receipt')
        for relative,h in receipt['files'].items():
            if sha(inside(out,relative))!=h:raise ValueError('Frame artifact changed: '+relative)
        result=read(folder/'anchors.json');masks=load_masks(folder/'masks.npz')
        if result!=states[frame['state_id']]['object_anchors']:
            raise ValueError('Enriched frame differs from audited result')
        if sidecar['frames'][frame['state_id']]!=result:raise ValueError('Sidecar frame differs')
        if result['mask_artifact']!=str((folder/'masks.npz').relative_to(out)) or result['mask_artifact_sha256']!=sha(folder/'masks.npz'):
            raise ValueError('Mask reference mismatch')
        if result['proposal_limit_reached']!=read(folder/'timing.json')['mask_limit_reached']:
            raise ValueError('Mask proposal limit mismatch')
        # Recompute all geometry and conflict flags from immutable raw model evidence.
        from .geometry import ground_frame
        expected=ground_frame(frame,read(folder/'detections.json'),masks,manifest['config'])
        for key,value in expected.items():
            if result.get(key)!=value:raise ValueError('Geometry or source-reference mismatch: '+key)
        if masks.shape[1:]!=(frame['height'],frame['width']):raise ValueError('Mask raster mismatch')
        for obj in result['objects']:
            x,y=obj['anchor']['pixel_xy'];w,h=frame['width'],frame['height']
            if not 0<=x<w or not 0<=y<h:raise ValueError('Out-of-bounds anchor')
            if obj['mask_index'] is not None:
                if not masks[obj['mask_index'],int(y),int(x)]:raise ValueError('Anchor outside referenced mask')
            if obj['apparent_size']['metric_dimensions'] is not None or obj['human_reviewed']:
                raise ValueError('Unsupported metric or human-review claim')
            counts[obj['segmentation_status']]+=1;flags.update(obj['review_flags'])
        counts['candidate_not_detected']+=sum(x['status']=='not_detected' for x in result['candidate_outcomes'])
        for c in result['candidate_outcomes']:
            if c['is_frame_focus']:focus[c['status']]+=1
        counts['frames']+=1
    summary={'status':'passed','frames':len(frames),'events':len(original['search_event_states']['events']),
        'original_values_preserved':True,'source_images_verified':True,'anchor_geometry_verified':True,
        'semantic_human_reviewed':False,'accuracy_measured':False,'counts':dict(counts),
        'focus_candidate_counts':dict(focus),'review_flags':dict(flags),'enriched_sha256':sha(out/'enriched.json'),
        'scope':'source hashes, additive preservation, frame completeness, recomputed geometry, mask membership and artifact integrity; not semantic accuracy'}
    return summary
