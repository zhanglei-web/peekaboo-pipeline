"""Explicit, source-bound mention selection; never expand to all scene objects."""
from .common import digest

SCHEMA = 'ego.event_description_candidates.v1'
TEXT_FIELDS = {'event_summary', 'observed_changes', 'search_region', 'states'}
STATE_FIELDS = {'state_description', 'hand_action', 'visibility_and_occlusion'}


def evidence_text(event, evidence):
    path=evidence['path'].split('/')
    if path[0] not in TEXT_FIELDS:
        raise ValueError('Evidence must come from event descriptions')
    if path[0]=='states' and (len(path)!=3 or path[2] not in STATE_FIELDS):
        raise ValueError('Evidence must come from a state description field')
    value=event
    for key in path:
        value=value[int(key)] if isinstance(value,list) else value[key]
    quote=evidence['quote']
    if not isinstance(value,str) or not isinstance(quote,str) or not quote or quote not in value:
        raise ValueError('Mention evidence does not match original event text')
    return value


def check_mapping(doc, mapping):
    if mapping.get('schema')!=SCHEMA or mapping.get('source_document_digest')!=digest(doc):
        raise ValueError('Description mapping belongs to another source document')
    ids={e['event_id'] for e in doc['search_event_states']['events']}
    if set(mapping['events'])!=ids:
        raise ValueError('Description mapping must cover exactly the original events')


def candidates_for_event(event, state, mapping, scene, catalog):
    from .contract import english_prompt
    selection=mapping['events'][event['event_id']]
    local_refs={f['entity_ref'] for s in event['states'] for f in s.get('focus_object_states',[])}
    frame_refs={f['entity_ref'] for f in state.get('focus_object_states',[])}
    result=[];seen=set()
    for mention in selection['candidates']:
        cid=mention['candidate_id']
        if cid in seen:raise ValueError('Duplicate event description candidate')
        seen.add(cid)
        evidence=mention['description_evidence']
        if not evidence:raise ValueError('Each candidate requires original description evidence')
        for item in evidence:evidence_text(event,item)
        kind=mention['reference_kind'];ref=mention['source_ref']
        if kind=='scene_object':
            if ref not in scene or cid!='scene:'+ref:raise ValueError('Unknown scene reference')
            label=scene[ref]['identity']['name'];links=[ref]
        elif kind=='event_entity':
            if ref not in local_refs or ref not in catalog or cid!='entity:'+ref:
                raise ValueError('Entity reference does not belong to this event')
            label=catalog[ref]['label'];links=catalog[ref].get('memory_entity_candidates',[])
            if not set(links)<=set(scene):raise ValueError('Unknown linked scene reference')
        elif kind=='description_mention':
            if not cid.startswith('mention:'+event['event_id']+':') or ref is not None:
                raise ValueError('New mentions must use an event-local extension namespace')
            label=mention['label'];links=[]
        else:raise ValueError('Unknown description reference kind')
        result.append({'candidate_id':cid,'reference_kind':kind,'source_ref':ref,'source_label':label,
            'existing_identity_status':'description_selected_identity_unverified',
            'scene_entity_candidates':links,'is_frame_focus':kind=='event_entity' and ref in frame_refs,
            'prompt':english_prompt(mention['prompt']),
            'selection_scope':'event_description_mentions',
            'description_event_id':event['event_id'], 'description_evidence':evidence,
            'description_role':mention['role'],
            'selection_review_status':mapping.get('policy',{}).get('selection_review_status','assistant_curated_not_human_semantic_reviewed')})
    for exclusion in selection.get('excluded_mentions',[]):
        if not exclusion.get('reason'):raise ValueError('Excluded mentions need an explanation')
        for item in exclusion['description_evidence']:evidence_text(event,item)
    return result
