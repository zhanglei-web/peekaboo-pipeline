"""Recheck final contract, original evidence and both target exports."""
from pathlib import Path
import sys
import jsonschema
from .common import ROOT, atomic, read, sha
from .direct import validate

def validate_run(root,only=None):
    from .engine import STEPS
    root=Path(root); manifest=read(root/'run_manifest.json'); cfg=manifest['config']
    out=root/'annotations'; schema=read(out/'target.schema.json')
    sys.path.insert(0,str(ROOT)); sys.path.insert(0,'/home/baai/zl/end_version')
    from annotation_contract import validate_potential
    from validate_final_annotation_sample import audit_temporal,assert_target_fields
    from export_confirmed import verify
    reference=Path('/home/baai/zl/end_version'); passed=[]; errors=[]
    expected=manifest['episode_ids']
    if only is not None:
        if not set(only)<=set(expected):raise ValueError('Validation subset contains unknown episode IDs')
        expected=[eid for eid in expected if eid in set(only)]
    for eid in expected:
        try:
            folder=out/'episodes'/eid; row=read(root/'rows'/(eid+'.json'))
            from .location_facts import verify as verify_locations
            verify_locations(row)
            for file,h in row.get('source_file_hashes',{}).items():
                if sha(file)!=h: raise ValueError('Original capture changed: '+file)
            validate(folder/'input')
            for stage in STEPS:
                receipt=read(root/'receipts'/eid/(stage+'.json'))
                if receipt['status']!='passed' or receipt['stage']!=stage or receipt['episode_id']!=eid:
                    raise ValueError('Invalid required stage receipt: '+stage)
                for file,h in receipt['outputs'].items():
                    if sha(file)!=h: raise ValueError('Stage output hash changed: '+file)
            from .dispatch import episode_mode,verify_episode
            if episode_mode(cfg): verify_episode(root,cfg,eid)
            path=out/'json'/(eid+'.json'); doc=read(path)
            if read(folder/'complete.json')['sha256']!=sha(path): raise ValueError('Final receipt mismatch')
            jsonschema.validate(doc,schema); validate_potential(doc); audit_temporal(doc)
            if row.get('dataset_location_facts'):
                if doc['metadata'].get('dataset_location_facts')!=row['dataset_location_facts']:
                    raise ValueError('Owner-supplied location facts changed or missing')
                for name in (eid+'.json','experiment_records.json'):
                    if read(out/'exports'/eid/name)['hidden'].get('location_descriptions')!=row['dataset_location_facts']:
                        raise ValueError('Exported location facts changed or missing')
            if row.get('input_evidence_exception'):
                if doc['metadata'].get('input_evidence_exception')!=row['input_evidence_exception']:
                    raise ValueError('Missing right frame exception not preserved')
                for name in (eid+'.json','experiment_records.json'):
                    if read(out/'exports'/eid/name)['metadata'].get('input_evidence_exception')!=row['input_evidence_exception']:
                        raise ValueError('Exported input exception changed or missing')
            if doc['metadata']['target']['description']!=row['target']: raise ValueError('Target changed')
            events=doc['search_event_states']['events']
            if len(events)!=row['event_count']: raise ValueError('Event count mismatch')
            if sha(row['target_markdown'])!=row['target_markdown_sha256']: raise ValueError('MD changed')
            for event,original in zip(events,row['event_index']['events'],strict=True):
                if event['event_id']!=original['event_id']: raise ValueError('Event identity changed')
                for a,b in zip(event['states'],original['states'],strict=True):
                    for key in ('phase','image_path','image_sha256','original_video_frame_index'):
                        if a[key]!=b[key]: raise ValueError('Event evidence changed: '+key)
                    if sha(a['image_path'])!=a['image_sha256']: raise ValueError('Event image changed')
            for name,template,kind in [(eid+'.json','cp_0011_session_20261029_115034.json','episode'),
                                        ('experiment_records.json','experiment_records.json','experiment')]:
                assert_target_fields(read(reference/template),read(out/'exports'/eid/name),kind)
            confirmed=out/'confirmed_json'/(eid+'.json')
            if cfg.get('facts') and eid in read(cfg['facts'])['scope']['episode_ids'] and not confirmed.exists():
                raise ValueError('Missing required scoped confirmation output')
            if confirmed.exists():
                projected=read(confirmed); jsonschema.validate(projected,schema); validate_potential(projected); audit_temporal(projected)
                verify(projected,read(cfg['facts']))
                receipt=read(out/'confirmed_receipts'/(eid+'.json'))
                if receipt['source_sha256']!=sha(path) or receipt['confirmation_sha256']!=sha(cfg['facts']):
                    raise ValueError('Confirmation source changed')
                for file,h in receipt['outputs'].items():
                    if sha(file)!=h: raise ValueError('Confirmed output changed')
                for name,template,kind in [(eid+'.json','cp_0011_session_20261029_115034.json','episode'),
                                            ('experiment_records.json','experiment_records.json','experiment')]:
                    assert_target_fields(read(reference/template),read(out/'confirmed_exports'/eid/name),kind)
            passed.append(eid)
        except Exception as exc: errors.append({'episode_id':eid,'error':str(exc)})
    result={'status':'passed' if len(passed)==len(expected) and not errors else 'incomplete',
        'validated':len(passed),'expected':len(expected),'errors':errors,
        'semantic_human_reviewed':False,'scope':'Direct package, schema, temporal tree, evidence, hashes and both exports'}
    if only is not None:
        result['episode_ids']=expected;result['validation_subset']=True
    atomic(root/('subset_validation.json' if only is not None else 'final_validation.json'),result); return result
