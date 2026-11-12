"""Create a compact additive JSON with a separate link to detailed inference evidence."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frame_anchors.common import atomic, read, sha
from frame_anchors.artifacts import validate


def export(run_dir, output_dir):
    run_dir=Path(run_dir).resolve();output_dir=Path(output_dir).resolve()
    validate(run_dir)
    manifest=read(run_dir/'run_manifest.json')
    source=Path(manifest['config']['input_json'])
    original=read(source);detailed=read(run_dir/'enriched.json')
    compact=deepcopy(original)
    output_dir.mkdir(parents=True,exist_ok=False)
    detailed_states={s['state_id']:s for e in detailed['search_event_states']['events'] for s in e['states']}
    audit=[];stats={'frames':0,'detected_entries':0,'not_detected_entries':0}
    for event in compact['search_event_states']['events']:
        for state in event['states']:
            if 'object_anchors' in state:raise ValueError('Original already has object_anchors')
            full=detailed_states[state['state_id']]['object_anchors']
            by_id={o['anchor_id']:o for o in full['objects']}
            rows=[]
            for candidate in full['candidate_outcomes']:
                for anchor_id in candidate['anchor_ids'] or [None]:
                    if anchor_id is None:
                        row={'name':candidate['source_label'],'ref':candidate['candidate_id'],
                             'anchor_xy_normalized':None,'size_wh_normalized':None,'status':'not_detected'}
                        stats['not_detected_entries']+=1
                    else:
                        obj=by_id[anchor_id]
                        if obj['source_candidate_id']!=candidate['candidate_id']:raise ValueError('Candidate mismatch')
                        method=obj['anchor']['method']
                        if method=='bbox_center_fallback':status='bbox_center_needs_review'
                        elif obj['review_flags']:status='mask_needs_review'
                        else:status='mask_unreviewed'
                        row={'name':candidate['source_label'],'ref':candidate['candidate_id'],
                             'anchor_xy_normalized':[round(x,6) for x in obj['anchor']['normalized_xy']],
                             'size_wh_normalized':[round(obj['apparent_size'][k],6) for k in ['bbox_width_fraction','bbox_height_fraction']],
                             'status':status}
                        if not all(0<=x<=1 for x in row['anchor_xy_normalized']+row['size_wh_normalized']):
                            raise ValueError('Coordinates or apparent size out of range')
                        stats['detected_entries']+=1
                    audit.append({'state_id':state['state_id'],'entry_index':len(rows),'detailed_anchor_id':anchor_id,
                                  'source_candidate_id':candidate['candidate_id']})
                    rows.append(row)
            if len(rows)!=len(full['objects'])+sum(not c['anchor_ids'] for c in full['candidate_outcomes']):
                raise ValueError('Lost detection or undetected candidate')
            state['object_anchors']=rows;stats['frames']+=1
    restored=deepcopy(compact)
    for event in restored['search_event_states']['events']:
        for state in event['states']:del state['object_anchors']
    if restored!=original:raise ValueError('Original values changed')
    if sha(source)!=manifest['input_sha256']:raise ValueError('Source changed')
    target=output_dir/source.name
    atomic(target,compact)
    if read(target)!=compact:raise ValueError('Written JSON differs')
    atomic(output_dir/'audit_index.json',{'source_json':str(source),'source_sha256':sha(source),
        'compact_json':str(target),'compact_sha256':sha(target),
        'detailed_json':str(run_dir/'enriched.json'),'detailed_sha256':sha(run_dir/'enriched.json'),
        'detailed_run_directory':str(run_dir),'entries':audit})
    report={'status':'passed','original_values_preserved':True,'all_candidates_accounted_for':True,
            'semantic_human_reviewed':False,**stats,'original_bytes':source.stat().st_size,
            'detailed_bytes':(run_dir/'enriched.json').stat().st_size,'compact_bytes':target.stat().st_size,
            'compact_json':str(target),'source_sha256':sha(source),'compact_sha256':sha(target)}
    atomic(output_dir/'validation.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(export(args.run_dir,args.output_dir),ensure_ascii=False))
