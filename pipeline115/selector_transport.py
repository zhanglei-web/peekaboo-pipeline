"""Audited streaming Responses transport; completion and exact identity stay mandatory."""
import json
from pathlib import Path
import time
import uuid
import requests
from .common import atomic,bind,now

POLICY={'transport':'responses_sse','connect_timeout_s':30,'read_timeout_s':1200,
        'total_timeout_s':1800,'attempts':3,'completion_gate':'exact model identity and completed response',
        'evidence_policy':'original request plus stream=true; no image/prompt/schema/budget reduction'}


def stream_response(url,body,key,request_id,event_path,post=None):
    from astra_memory.provider import TransportError,ModelServiceError
    session=requests.Session();session.trust_env=False
    sender=post or session.post
    started=time.monotonic();payload={**body,'stream':True}
    try:
        with sender(url,json=payload,headers={'Authorization':'Bearer '+key,
                    'Idempotency-Key':request_id,'Content-Type':'application/json'},
                    stream=True,timeout=(POLICY['connect_timeout_s'],POLICY['read_timeout_s']),
                    allow_redirects=False) as response:
            if response.status_code!=200:
                raise TransportError('selector stream HTTP '+str(response.status_code),
                    response.status_code in (408,409,429) or response.status_code>=500)
            if 'text/event-stream' not in response.headers.get('content-type','').lower():
                raise ModelServiceError('Streaming selector returned a non-SSE response')
            data=[];response_id=None
            with Path(event_path).open('w') as log:
                for line in response.iter_lines(chunk_size=1024,decode_unicode=False):
                    if time.monotonic()-started>POLICY['total_timeout_s']:
                        raise TransportError('selector stream total timeout',True)
                    if line:
                        if line.startswith(b'data:'):data.append(line[5:].lstrip())
                        continue
                    if not data:continue
                    raw=b'\n'.join(data);data=[]
                    if raw==b'[DONE]':break
                    event=json.loads(raw)
                    clean=json.dumps(event,ensure_ascii=False).replace(key,'[REDACTED]')
                    log.write(clean+'\n');log.flush()
                    kind=event.get('type');obj=event.get('response',{})
                    if kind=='response.created':response_id=obj.get('id')
                    if kind=='response.completed':
                        if response_id and obj.get('id')!=response_id:
                            raise ModelServiceError('Streaming response identity changed')
                        return json.loads(json.dumps(obj).replace(key,'[REDACTED]'))
                    if kind in ('response.failed','response.incomplete','error'):
                        raise ModelServiceError('Streaming selector returned '+str(kind)+'; inspect saved event log')
            raise TransportError('selector stream ended before completed response',True)
    except requests.RequestException as exc:
        raise TransportError('selector stream network/timeout failure: '+type(exc).__name__,True) from None
    finally:session.close()


def request_verified(folder,model,body,key,signature):
    from astra_memory.provider import response_text,TransportError,ModelServiceError
    folder=Path(folder);archive=folder/'transport_attempts';archive.mkdir(exist_ok=True)
    bind(folder/'stream_transport_policy.json',POLICY)
    request_id=signature+'-stream-'+uuid.uuid4().hex
    if (folder/'response.json').exists():
        (archive/('prior-'+uuid.uuid4().hex+'.response.json')).write_bytes((folder/'response.json').read_bytes())
    for index in range(POLICY['attempts']):
        stem=uuid.uuid4().hex;record=archive/(stem+'.json');started=time.monotonic()
        entry={'request_id':request_id,'attempt':index+1,'started_at':now(),'status':'started',
               'transport':'responses_sse','read_timeout_s':POLICY['read_timeout_s']}
        atomic(record,entry);raw=None
        try:
            raw=stream_response(model['base_url'].rstrip('/')+'/responses',body,key,request_id,
                                archive/(stem+'.events.jsonl'))
            atomic(archive/(stem+'.response.json'),raw);atomic(folder/'response.json',raw)
            entry['response_file']=stem+'.response.json'
            text=response_text(raw,model)
            entry['status']='verified';entry['model']=raw['model']
            return raw,text
        except TransportError as exc:
            entry.update(status='transport_error',error=str(exc),retryable=exc.retryable,
                         server_completion_unknown=True)
            if not exc.retryable or index+1==POLICY['attempts']:raise
            # Reuse idempotency key when the service may still finish the request.
        except ModelServiceError as exc:
            missing_identity=isinstance(raw,dict) and not raw.get('model')
            entry.update(status='unverified_response',error=str(exc),retryable=missing_identity)
            if not missing_identity or index+1==POLICY['attempts']:raise
            request_id=signature+'-stream-'+uuid.uuid4().hex
        except Exception as exc:
            entry.update(status='invalid_response',error_type=type(exc).__name__,retryable=False);raise
        finally:
            entry['wall_seconds']=time.monotonic()-started;atomic(record,entry)
        time.sleep(min(10*(index+1),30))
