"""Durable whole-episode ownership after Astra; stage routing stays compatible."""
from copy import deepcopy
from pathlib import Path
from .common import atomic, bind, now, read, sha

MODEL_STAGES = ('scene', 'events', 'temporal')
BUDGETS = {'scene': 10000, 'events': 6144, 'temporal': 3072}


def episode_mode(cfg):
    return cfg.get('dispatch', {}).get('mode', 'stage') == 'episode'


def endpoint_keys(cfg):
    if episode_mode(cfg):
        return list(cfg['dispatch']['endpoints'])
    return list(dict.fromkeys(cfg['routes'].values()))


def validate_config(cfg):
    dispatch = cfg.get('dispatch', {'mode': 'stage'})
    if dispatch.get('mode') not in ('stage', 'episode'):
        raise ValueError('Unknown dispatch mode')
    if episode_mode(cfg):
        if set(dispatch) != {'mode', 'endpoints'}:
            raise ValueError('Episode dispatch requires only mode and endpoints')
        keys = dispatch['endpoints']
        if not isinstance(keys, list) or not keys or len(set(keys)) != len(keys):
            raise ValueError('Episode endpoints must be a nonempty unique list')
        for key in keys:
            profile = cfg['endpoints'][key]
            if profile['context_tokens'] <= max(BUDGETS.values()):
                raise ValueError('Whole-episode endpoint cannot fit unchanged scene budget: ' + key)
    elif set(dispatch) != {'mode'}:
        raise ValueError('Stage dispatch accepts only mode')


def assignment_path(root, eid):
    return Path(root) / 'assignments' / (eid + '.json')


def config_path(root, eid):
    return Path(root) / 'episode_configs' / (eid + '.json')


def owned_config(cfg, key):
    value = deepcopy(cfg)
    value['routes'] = {stage: key for stage in MODEL_STAGES}
    return value


def read_assignment(root, cfg, eid):
    path = assignment_path(root, eid)
    if not path.exists():
        return None
    record = read(path)
    key = record['endpoint']
    if (not episode_mode(cfg) or record['episode_id'] != eid or
            key not in endpoint_keys(cfg) or record['profile'] != cfg['endpoints'][key]):
        raise ValueError('Episode ownership identity changed: ' + eid)
    worker_cfg = config_path(root, eid)
    actual=read(worker_cfg);expected=owned_config(cfg,key)
    legacy_ok=False
    legacy=Path(root)/'legacy_episode_configs.json'
    if actual!=expected and legacy.exists():
        manifest=read(Path(root)/'run_manifest.json')
        if sha(legacy)!=manifest.get('legacy_episode_configs_sha256'):
            raise ValueError('Legacy episode configuration registry changed')
        prior=read(legacy).get('episodes',{}).get(eid,{})
        normalized=deepcopy(actual);normalized['astra_config']=expected['astra_config']
        legacy_ok=(normalized==expected and prior.get('config_sha256')==sha(worker_cfg)
                   and prior.get('assignment_sha256')==sha(path))
    if (actual != expected and not legacy_ok) or sha(worker_cfg) != record['config_sha256']:
        raise ValueError('Owned worker configuration changed: ' + eid)
    return record


def assign(root, cfg, eid, key):
    """Caller holds the engine mutex and root OS lock. Publish owner last."""
    existing = read_assignment(root, cfg, eid)
    if existing:
        if existing['endpoint'] != key:
            raise ValueError('Cannot transfer an owned episode: ' + eid)
        return existing
    if key not in endpoint_keys(cfg):
        raise ValueError('Endpoint is not enabled for episode dispatch')
    path = config_path(root, eid)
    bind(path, owned_config(cfg, key))
    record = {'episode_id': eid, 'endpoint': key, 'profile': cfg['endpoints'][key],
              'assigned_at': now(), 'config_sha256': sha(path),
              'scope': 'scene, events, temporal and assembly; no automatic endpoint transfer'}
    atomic(assignment_path(root, eid), record)
    return record


def recover_assignment(root, cfg, eid):
    record = read_assignment(root, cfg, eid)
    if record:
        return record
    # A crash can leave the bound worker config before publishing ownership.
    path = config_path(root, eid)
    if path.exists():
        matches = [key for key in endpoint_keys(cfg) if read(path) == owned_config(cfg, key)]
        if len(matches) != 1:
            raise ValueError('Orphan worker configuration cannot be identified: ' + eid)
        return assign(root, cfg, eid, matches[0])
    return None


def receipt_metadata(root, cfg, eid):
    record = read_assignment(root, cfg, eid)
    if not record:
        raise ValueError('Missing whole-episode assignment: ' + eid)
    return {'endpoint': record['endpoint'], 'profile': record['profile'],
            'assignment_sha256': sha(assignment_path(root, eid)),
            'config_sha256': record['config_sha256']}


def verify_episode(root, cfg, eid):
    """No reclassification of old completed stage-routed episodes."""
    owner = read_assignment(root, cfg, eid)
    if not owner:
        migration = Path(root) / 'legacy_completed_episodes.json'
        manifest = read(Path(root) / 'run_manifest.json')
        if (not migration.exists() or sha(migration) != manifest.get('legacy_completed_episodes_sha256') or
                eid not in read(migration)['episode_ids']):
            raise ValueError('Final episode has no ownership record: ' + eid)
        return
    expected = receipt_metadata(root, cfg, eid)
    for stage in (*MODEL_STAGES, 'assemble'):
        receipt = read(Path(root) / 'receipts' / eid / (stage + '.json'))
        if receipt.get('dispatch') != expected:
            raise ValueError('Stage used a different episode owner: ' + stage)
        if stage in MODEL_STAGES:
            profile = read(Path(root) / 'annotations/episodes' / eid / (stage + '_profile.json'))
            if profile != owner['profile']:
                raise ValueError('Stage model profile differs from owner: ' + stage)
