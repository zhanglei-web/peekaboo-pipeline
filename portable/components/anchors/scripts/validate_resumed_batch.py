"""Independently audit a resumed delivery, without models or output mutation."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from frame_anchors.common import atomic,read,sha
from frame_anchors.description_scope import check_mapping,candidates_for_event

FIELDS={'name','ref','anchor_xy_normalized','size_wh_normalized','status'}
STATUSES={'mask_unreviewed','mask_needs_review','bbox_center_needs_review','not_detected'}


def audit(root,verify_images=True):
    root=Path(root).resolve();inventory=read(root/'input_inventory.json');state=read(root/'batch_state.json')
    issues={};groups=Counter();statuses=Counter();provenance=Counter();seen_images={};missing=[]
    expected_files=set();completed=state['completed'];checked=0;frames=0;candidate_observations=0
    for row in inventory['records']:
        key=row['group']+'/'+row['record_id'];target=root/'json'/(key+'.json');expected_files.add(target)
        if key not in completed:
            missing.append(key);continue
        try:
            entry=completed[key]
            if str(target)!=entry['json']:raise ValueError('Delivery path mismatch')
            if sha(row['source_json'])!=row['source_sha256']:raise ValueError('Source JSON hash mismatch')
            if sha(target)!=entry['sha256']:raise ValueError('Delivery hash mismatch')
            source=read(row['source_json']);compact=read(target);restored=deepcopy(compact)
            origin=Path(entry.get('retained_from',root));mapping_path=origin/'selections'/(key+'.json')
            mapping=read(mapping_path);check_mapping(source,mapping)
            if entry.get('mapping_sha256') and sha(mapping_path)!=entry['mapping_sha256']:
                raise ValueError('Candidate map hash mismatch')
            scene={o['id']:o for o in source['scene_memory']['objects']}
            catalog=source['search_event_states']['entity_reference_catalog']
            original_events={e['event_id']:e for e in source['search_event_states']['events']}
            record_frames=0
            for event in restored['search_event_states']['events']:
                eid=event['event_id'];original=original_events[eid]
                for state_value in event['states']:
                    anchors=state_value.pop('object_anchors');record_frames+=1
                    if not isinstance(anchors,list):raise ValueError('Compact anchor field is not an array')
                    selected=candidates_for_event(original,state_value,mapping,scene,catalog)
                    expected={c['candidate_id'] for c in selected}
                    if {a['ref'] for a in anchors}!=expected:raise ValueError('Per-frame candidate coverage mismatch')
                    candidate_observations+=len(expected)
                    by_ref={c['candidate_id']:c for c in selected}
                    for a in anchors:
                        if set(a)!=FIELDS or a['status'] not in STATUSES:raise ValueError('Compact fields/status mismatch')
                        if a['name']!=by_ref[a['ref']]['source_label']:raise ValueError('Candidate source label mismatch')
                        if a['status']=='not_detected':
                            if a['anchor_xy_normalized'] is not None or a['size_wh_normalized'] is not None:
                                raise ValueError('Undetected candidate has coordinates')
                        else:
                            for k in ['anchor_xy_normalized','size_wh_normalized']:
                                if not isinstance(a[k],list) or len(a[k])!=2:raise ValueError('Invalid coordinate shape')
                                if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and 0<=v<=1 for v in a[k]):
                                    raise ValueError('Invalid normalized values')
                        statuses[a['status']]+=1
                    path=state_value['image_path'];image_hash=state_value['image_sha256']
                    if path in seen_images and seen_images[path]!=image_hash:raise ValueError('Inconsistent source image hashes')
                    seen_images[path]=image_hash
            if restored!=source:raise ValueError('An original JSON value changed')
            if record_frames!=row['frame_count']:raise ValueError('Record frame count mismatch')
            checked+=1;frames+=record_frames;groups[row['group']]+=1
            provenance[mapping['policy']['selection_review_status']]+=1
        except Exception as exc:issues[key]={'type':type(exc).__name__,'error':str(exc)}
    image_issues={}
    if verify_images:
        for path,expected_hash in seen_images.items():
            try:
                if sha(path)!=expected_hash:image_issues[path]='Image hash mismatch'
            except Exception as exc:image_issues[path]=str(exc)
    unexpected=sorted(str(p) for p in (root/'json').rglob('*.json') if p not in expected_files)
    valid=(not inventory.get('errors') and checked==len(inventory['records']) and not missing and
           not issues and not image_issues and not unexpected and not state.get('failures'))
    return {'status':'passed' if valid else 'incomplete','expected_records':len(inventory['records']),
        'validated_records':checked,'unprocessed':missing,'record_issues':issues,'record_failures':state.get('failures',{}),
        'unexpected_json_files':unexpected,'frames':frames,'unique_images':len(seen_images),
        'source_image_hashes_verified':verify_images,'image_issues':image_issues,
        'groups':dict(groups),'candidate_observations':candidate_observations,'anchor_status_counts':dict(statuses),
        'selection_provenance_counts':dict(provenance),'semantic_human_reviewed':False,'accuracy_measured':False,
        'scope':'source and image hashes, original value preservation, event/frame candidate coverage, compact fields and normalized ranges; not human semantic acceptance',
        'json_directory':str(root/'json'),'validated_at':time.time()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path)
    ap.add_argument('--skip-images',action='store_true');args=ap.parse_args()
    report=audit(args.root,not args.skip_images)
    if args.output:atomic(args.output,report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if report['record_issues'] or report['image_issues']:raise SystemExit(1)
if __name__=='__main__':main()
