"""Configured selector transport; retain the frozen candidate/evidence contract."""
import json
import os
import time
from pathlib import Path
import jsonschema
from .common import bind,read,sha,atomic
from .selector_transport import request_verified


def call(selector, config_path, out, phase, paths, ids, maximum, prompt, geometry=None):
    folder=Path(out)/phase;folder.mkdir(exist_ok=True)
    model={**read(config_path)['model'],'max_output_tokens':5000,'timeout_s':300}
    bind(Path(out)/(phase+'_profile.json'),read(config_path)['model'])
    schema=selector.selection_schema(ids,maximum)
    images=[{'path':str(p),'sha256':sha(p),'image_ref':p.stem} for p in paths]
    context={'allowed_original_frame_indices':ids,'maximum_frames':maximum,
             'frame_label_namespace':'original_video_frame_index','image_layout':'row-major'}
    if geometry is not None:context['tf_viewpoint_evidence']=geometry
    descriptor={'images':images,'context':context,'schema':schema}
    result_path=folder/'selection.json'
    if result_path.exists():
        if read(folder/'inputs.json')!=descriptor or (folder/'prompt.txt').read_text()!=prompt:
            raise ValueError('Cached selection evidence or prompt changed')
        result=read(result_path);jsonschema.validate(result,schema);return result
    atomic(folder/'inputs.json',descriptor);(folder/'prompt.txt').write_text(prompt)
    body=selector.make_body(model,prompt,schema,context,images)
    print(phase+' request started; model='+model['name'],flush=True)
    tick=time.perf_counter()
    raw,text=request_verified(folder,model,body,os.environ[model['api_key_env']],
                              sha(folder/'inputs.json')+'-'+phase)
    result=json.loads(text);jsonschema.validate(result,schema)
    chosen=[x['frame'] for x in result['selected']]
    if len(chosen)!=len(set(chosen)):raise ValueError('Duplicate selected frame')
    atomic(result_path,result)
    atomic(folder/'timing.json',{'seconds':time.perf_counter()-tick,'model':raw['model'],
                                'usage':raw.get('usage'),'selected_frames':chosen})
    print(json.dumps({'phase':phase,'model':raw['model'],'frames':chosen}),flush=True)
    return result
