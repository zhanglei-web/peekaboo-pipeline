"""Isolated stage adapter: legacy module globals never cross worker processes."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import time
from .common import QWEN, ROOT, atomic, bind, digest, now, read, sha, tree
from .client import Client, YieldGPU, CapacityError, TransportPending

class InferenceBlocked(BaseException): pass

def setup(config):
    sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(QWEN/'scripts'))
    import annotate_batch as annotation
    import batch_temporal_trees as temporal
    annotation.OUT = Path(config['output_root'])/'annotations'
    return annotation,temporal

def run(stage, config, row):
    a,t = setup(config); out = a.OUT; folder = out/'episodes'/row['episode_id']
    folder.mkdir(parents=True,exist_ok=True)
    profile = config['endpoints'][config['routes'].get(stage,'local')] if stage in ('scene','events','temporal') else None
    if profile:
        def before_request():
            if (Path(config['output_root'])/'stop_requested').exists():
                raise YieldGPU()
            if profile['placement'] == 'local_gpu' and (Path(config['output_root'])/'yield_local').exists():
                raise YieldGPU()
        client = Client(profile,before_request)
        override_path=Path(config['output_root'])/'recovery_overrides'/(row['episode_id']+'.json')
        overrides=read(override_path).get(stage,{}) if override_path.exists() else {}
        def call(request_folder,prompt,images,schema,max_tokens=10000):
            relative=str(Path(request_folder).relative_to(folder))
            if relative in overrides:
                prompt+='\nValidation feedback for regeneration (all original evidence and constraints still apply): '+overrides[relative]
            try: return client(request_folder,prompt,images,schema,max_tokens)
            except (CapacityError,TransportPending) as exc:
                # Do not let the legacy semantic-repair loop resubmit uncertain
                # transport completions or a request known not to fit context.
                raise InferenceBlocked(str(exc)) from None
        a.pipe.local_call = call; t.local_call = call
        # Accepted temporal cache keys in the legacy module omit endpoint identity.
        # Bind the whole stage directory to endpoint/model/config before any hit.
        bind(folder/(stage+'_profile.json'),profile)
    tick = time.perf_counter()
    if stage == 'astra':
        from .direct import validate
        validate(folder/'input')
        import select_scene_keyframes as selector
        from .selector import call as configured_call
        def call(output,phase,paths,ids,maximum,prompt,geometry=None):
            return configured_call(selector,config['astra_config'],output,phase,paths,ids,maximum,prompt,geometry)
        selector.call = call
        sys.argv = ['select_scene_keyframes','--input',str(folder/'input'),'--output',str(folder/'selection')]
        selector.main()
        selection_path=folder/'selection/selected_frames.json'
        selection=read(selection_path)
        selection['method']=selection['method'].replace('Astra',read(config['astra_config'])['model']['name'])
        atomic(selection_path,selection)
    elif stage == 'scene':
        scene = a.pipe.scene_job(folder/'selection'); atomic(folder/'scene.json',scene)
    elif stage == 'events':
        payloads = [a.event_job(e,folder/'events'/('event_%02d'%e['ordinal']),row['target'])
                    for e in row['event_index']['events']]
        atomic(folder/'event_payloads.json',payloads)
    elif stage == 'temporal':
        scene = read(folder/'scene.json'); payloads = read(folder/'event_payloads.json')
        source = a.pipe.export_episode(row,scene,payloads)
        source['metadata']['search_scope'] = 'Provided direct frame package and original event evidence'
        source['metadata']['provenance'] = 'direct_package='+str(folder/'input')+'; pipeline=11-5; endpoints='+json.dumps(config['endpoints'],sort_keys=True)+'; TF is unverified VIO camera evidence.'
        source['search_event_states']['provenance']['description_source'] = 'Configured Qwen API; original triplet evidence; endpoint identity in stage receipts.'
        atomic(folder/'visual_source.json',source)
        graph = a.initialize_temporal_graph(scene); temporal = deepcopy(source)
        # One episode lock is held by the parent throughout this subprocess.
        # Every accepted update is saved by the original temporal implementation.
        for event in temporal['search_event_states']['events']:
            graph,changes,uncertainties,cached = t.annotate_event(source,event,graph,folder/'temporal'/('event_%02d'%event['ordinal']))
            event['spatial_changes'] = changes
            event['spatial_change_annotation'] = {'status':'local_qwen_visual_inference_unreviewed',
                'uncertainties':uncertainties,'source':'original_event_triplet_images','description_language':'en'}
        temporal['schema'] = t.VERSION
        temporal['scene_memory']['spatial_structure'] = a.nested_temporal_tree(graph)
        t.validate_output(source,temporal,t.output_schema(source['schema']))
        atomic(folder/'temporal.json',temporal); atomic(folder/'parent_graph.json',graph)
    elif stage == 'assemble':
        source = read(folder/'visual_source.json'); temporal = read(folder/'temporal.json')
        a.finish(row,folder,source,read(folder/'event_payloads.json'),temporal,read(folder/'parent_graph.json'),{},tick)
        # Keep enum vocabulary compatible; actual model identity and transport
        # are explicit in metadata and immutable receipts, not inferred from enums.
        final_path = out/'json'/(row['episode_id']+'.json'); final = read(final_path)
        final['README']['annotation_record']['label_source'] = '11-5_configured_selector_qwen_direct_frames_unreviewed'
        final['README']['annotation_record']['manifest'] = str(Path(config['output_root'])/'run_manifest.json')
        from .location_facts import attach,project_exports
        attach(final,row)
        import jsonschema
        jsonschema.validate(final,read(out/'target.schema.json'))
        atomic(final_path,final)
        if row.get('dataset_location_facts') or row.get('input_evidence_exception'):
            episode,experiment=project_exports(final,*a.export(final))
            episode['scene_ref']='../../json/'+row['episode_id']+'.json#/scene_memory'
            atomic(out/'exports'/row['episode_id']/(row['episode_id']+'.json'),episode)
            atomic(out/'exports'/row['episode_id']/'experiment_records.json',experiment)
        receipt = read(folder/'complete.json'); receipt['sha256'] = sha(final_path)
        receipt['assembly_seconds'] = receipt.pop('local_processing_seconds')
        receipt['completed_stage_seconds_sum'] = sum(read(p)['seconds'] for p in (Path(config['output_root'])/'receipts'/row['episode_id']).glob('*.json') if p.stem!='assemble') + receipt['assembly_seconds']
        receipt['timing_scope'] = 'Sum of stage wall times, including cached work and transfer; parallel episodes may overlap. Not total batch elapsed time.'
        receipt['pipeline_version'] = '11-5'; receipt['endpoints'] = config['endpoints']; atomic(folder/'complete.json',receipt)
        facts_path = config.get('facts')
        if facts_path:
            import export_confirmed as confirmed
            facts = read(facts_path)
            if row['episode_id'] in facts['scope']['episode_ids']:
                projected = confirmed.transform(final,facts)
                import jsonschema
                jsonschema.validate(projected,read(out/'target.schema.json')); a.validate_potential(projected); a.audit_temporal(projected)
                target = out/'confirmed_json'/(row['episode_id']+'.json')
                atomic(target,projected); episode,experiment = project_exports(projected,*a.export(projected))
                episode['scene_ref'] = '../../confirmed_json/'+row['episode_id']+'.json#/scene_memory'
                paths = {target:projected,out/'confirmed_exports'/row['episode_id']/(row['episode_id']+'.json'):episode,
                         out/'confirmed_exports'/row['episode_id']/'experiment_records.json':experiment}
                for path,value in paths.items(): atomic(path,value)
                atomic(out/'confirmed_receipts'/(row['episode_id']+'.json'),{'status':'passed',
                    'source_sha256':sha(final_path),'confirmation_sha256':sha(facts_path),
                    'outputs':{str(p):sha(p) for p in paths},'model_calls_for_outcome_projection':0})
    else: raise ValueError('Unknown adapter stage '+stage)
    return {'seconds':time.perf_counter()-tick,'stage':stage,'completed_at':now(),'profile':profile}

def main():
    stage,config_path,row_path = sys.argv[1:]
    try:
        result = run(stage,read(config_path),read(row_path)); print(json.dumps(result),flush=True)
    except YieldGPU:
        raise SystemExit(75)
    except InferenceBlocked as exc:
        print(str(exc),file=sys.stderr,flush=True); raise SystemExit(76)

if __name__ == '__main__': main()
