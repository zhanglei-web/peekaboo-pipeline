import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import select_scene_keyframes  # sets provider import path
import astra_request_retry as retry

MODEL = {'name':'gpt-6-astra', 'transport':'responses', 'base_url':'http://unused', 'timeout_s':1}


def response(model='gpt-6-astra'):
    data = {'status':'completed', 'output':[{'type':'message','content':[{'type':'output_text','text':'{}'}]}]}
    if model is not None:
        data['model'] = model
    return json.dumps(data).encode()


def test_missing_identity_preserved_then_retried_with_fresh_id(tmp_path, monkeypatch):
    ids = []
    def post(*args):
        ids.append(args[-1])
        return response(None if len(ids)==1 else 'gpt-6-astra')
    monkeypatch.setattr(retry, 'post_json', post)
    monkeypatch.setattr(retry.time, 'sleep', lambda _:None)
    raw,text = retry.request_verified(tmp_path, MODEL, {}, 'secret', 'input')
    assert raw['model'] == MODEL['name'] and text == '{}'
    assert len(ids)==2 and ids[0] != ids[1]
    archived = list((tmp_path/'transport_attempts').glob('*.response.json'))
    assert len(archived)==2
    assert any('model' not in json.loads(p.read_text()) for p in archived)


def test_timeout_keeps_idempotency_id(tmp_path, monkeypatch):
    ids = []
    def post(*args):
        ids.append(args[-1])
        if len(ids)==1:
            raise retry.TransportError('timeout', True)
        return response()
    monkeypatch.setattr(retry, 'post_json', post)
    monkeypatch.setattr(retry.time, 'sleep', lambda _:None)
    retry.request_verified(tmp_path, MODEL, {}, 'secret', 'input')
    assert len(ids)==2 and ids[0]==ids[1]


def test_wrong_model_never_accepted_or_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(retry, 'post_json', lambda *args:response('other-model'))
    with pytest.raises(retry.ModelServiceError):
        retry.request_verified(tmp_path, MODEL, {}, 'secret', 'input')
    assert len(list((tmp_path/'transport_attempts').glob('*.response.json')))==1


def test_repeated_missing_identity_exhausts_without_result(tmp_path, monkeypatch):
    monkeypatch.setattr(retry, 'post_json', lambda *args:response(None))
    monkeypatch.setattr(retry.time, 'sleep', lambda _:None)
    with pytest.raises(retry.ModelServiceError):
        retry.request_verified(tmp_path, MODEL, {}, 'secret', 'input', attempts=2)
    assert len(list((tmp_path/'transport_attempts').glob('*.response.json')))==2
