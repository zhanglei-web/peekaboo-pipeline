"""Explicitly scoped approval for one unavailable right frame; never substitute it."""
from pathlib import Path
from copy import deepcopy
import cv2
from .common import read,sha

def partial_right(raw,md,left_index,authorization_path):
    authorization=read(authorization_path);eid=md.parent.name
    if authorization['episode_id']!=eid or authorization['side']!='right' or authorization['policy']!='use_complete_left_keep_right_missing':
        raise ValueError('No matching explicit approval for missing right evidence')
    index=deepcopy(left_index);index.update(schema='partial_right_event_image_index.v1',directory=str(md.parent/'right'))
    missing=[];present=[]
    for event in index['events']:
        kept=[]
        for state in event['states']:
            path=md.parent/'right'/state['image_filename']
            authorized_tail = (authorization.get('decoded_right_frames') is not None
                               and str(path)==authorization['missing_file']
                               and state['original_video_frame_index']>=authorization['decoded_right_frames'])
            if not path.is_file() or authorized_tail:
                missing.append({'ordinal':event['ordinal'],'phase':state['phase'],
                                'frame_index':state['original_video_frame_index'],'expected_image_path':str(path)})
                continue
            state.update(image_path=str(path),image_sha256=sha(path));kept.append(state);present.append(path.name)
        event['states']=kept
    if len(missing)!=1 or missing[0]['expected_image_path']!=authorization['missing_file'] or missing[0]['frame_index']!=authorization['frame_index']:
        raise ValueError('Missing right evidence exceeds explicitly authorized scope')
    original_names={p.name for p in (md.parent/'right').glob('*.jpg')}
    excluded=authorization.get('excluded_out_of_bounds_image_sha256')
    if excluded:
        excluded_path=Path(authorization['missing_file'])
        if sha(excluded_path)!=excluded:raise ValueError('Excluded right image changed')
        original_names.remove(excluded_path.name)
    if set(present)!=original_names:
        raise ValueError('Unexpected right frame images')
    video=raw/'videos/right.mkv';cap=cv2.VideoCapture(str(video));count=0
    try:
        while True:
            ok,_=cap.read()
            if not ok:break
            count+=1
    finally:cap.release()
    tail_ok=missing[0]['frame_index']==count
    if authorization.get('decoded_right_frames') is not None:
        tail_ok=(count==authorization['decoded_right_frames']
                 and missing[0]['frame_index']==read(raw/'meta.json')['topics']['left']['written_frames']-1
                 and count<=missing[0]['frame_index']<=count+1)
    if count!=read(raw/'meta.json')['topics']['right']['written_frames'] or not tail_ok:
        raise ValueError('Exception requires verified absence exactly after the right video ends')
    index['image_count']=len(present);index['missing_states']=missing
    report={'status':'left_complete_right_partial_user_approved','episode_id':eid,
            'authorization_file':str(authorization_path),'authorization_sha256':sha(authorization_path),
            'source_video':str(video),'source_video_sha256':sha(video),'decoded_right_frames':count,
            'missing_states':missing,'substitute_frame_used':False}
    return index,report

def verify(row):
    report=row.get('input_evidence_exception')
    if report is None:return
    if report['episode_id']!=row['episode_id'] or report['substitute_frame_used'] is not False:
        raise ValueError('Invalid evidence exception identity')
    if sha(report['authorization_file'])!=report['authorization_sha256'] or sha(report['source_video'])!=report['source_video_sha256']:
        raise ValueError('Evidence exception source changed')
    authorization=read(report['authorization_file']);missing=report['missing_states']
    if len(missing)!=1 or authorization['episode_id']!=row['episode_id'] or authorization['policy']!='use_complete_left_keep_right_missing':
        raise ValueError('Invalid evidence exception approval')
    if missing[0]['expected_image_path']!=authorization['missing_file'] or missing[0]['frame_index']!=authorization['frame_index']:
        raise ValueError('Evidence exception outside authorized frame')
    missing_path=Path(missing[0]['expected_image_path'])
    excluded=authorization.get('excluded_out_of_bounds_image_sha256')
    if excluded:
        if not missing_path.is_file() or sha(missing_path)!=excluded:raise ValueError('Excluded original image changed')
    elif missing_path.exists():raise ValueError('Missing original evidence is now available; re-enroll explicitly')
    if authorization.get('decoded_right_frames') is not None:
        if report['decoded_right_frames']!=authorization['decoded_right_frames'] or not (report['decoded_right_frames']<=missing[0]['frame_index']<=report['decoded_right_frames']+1):
            raise ValueError('Tail exception outside verified video bounds')
    if row['right_event_index']['missing_states']!=missing or row['right_event_index']['image_count']!=row['event_count']*3-1:
        raise ValueError('Partial right evidence count/mapping mismatch')

def schema():
    text={'type':'string','minLength':1}
    def obj(fields):return {'type':'object','properties':fields,'required':list(fields),'additionalProperties':False}
    return obj({'status':{'const':'left_complete_right_partial_user_approved'},'episode_id':text,
        'authorization_file':text,'authorization_sha256':text,'source_video':text,'source_video_sha256':text,
        'decoded_right_frames':{'type':'integer','minimum':1},'substitute_frame_used':{'const':False},
        'missing_states':{'type':'array','minItems':1,'maxItems':1,'items':obj({'ordinal':{'type':'integer','minimum':1},
            'phase':{'enum':['start','middle','end']},'frame_index':{'type':'integer','minimum':0},'expected_image_path':text})}})
