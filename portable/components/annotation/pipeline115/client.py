"""Immutable, model-specific OpenAI-compatible requests; no implicit fallback."""
from pathlib import Path
import base64
import json
import os
import time
import uuid
import jsonschema
import requests
from .common import atomic, bind, digest, now, read, sha

class CapacityError(ValueError): pass
class TransportPending(RuntimeError): pass
class YieldGPU(BaseException): pass

class Client:
    def __init__(self, profile, before_request=None):
        self.profile = profile
        self.before_request = before_request
        self.session = requests.Session(); self.session.trust_env = False

    def headers(self):
        env = self.profile.get('api_key_env')
        if env and not os.environ.get(env): raise ValueError('Missing credential environment: '+env)
        return {'Authorization':'Bearer '+os.environ[env]} if env else {}

    def probe(self, require_context=False):
        r = self.session.get(self.profile['base_url'].rstrip('/')+'/models',
                             headers=self.headers(), timeout=10, allow_redirects=False)
        if r.status_code != 200: raise RuntimeError('Model probe HTTP '+str(r.status_code))
        entries = r.json()['data']
        models = [m['id'] for m in entries]
        if self.profile['model'] not in models: raise ValueError('Configured model absent')
        entry = next(m for m in entries if m['id']==self.profile['model'])
        context = entry.get('meta',{}).get('n_ctx')
        if require_context and (not isinstance(context,int) or context < self.profile['context_tokens']):
            raise CapacityError('Remote context is unverified or below configured capacity: '+self.profile['name'])
        return {'model':self.profile['model'], 'available_models':models, 'context_tokens':context}

    def __call__(self, folder, prompt, images, schema, max_tokens=10000):
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        images = [Path(p) for p in images]
        descriptor = {'profile':self.profile, 'prompt':prompt, 'schema':schema,
                      'images':[{'path':str(p.resolve()), 'sha256':sha(p)} for p in images],
                      'max_tokens':max_tokens, 'temperature':0, 'seed':42, 'thinking':False}
        bind(folder/'inputs.json', descriptor)
        cached = folder/'result.json'
        if cached.exists():
            receipt = read(folder/'receipt.json')
            if receipt['input_key'] != digest(descriptor) or receipt['sha256'] != sha(cached):
                raise ValueError('Cached result identity/hash mismatch')
            data = read(cached); jsonschema.validate(data, schema); return data
        # Do not reduce output budgets to make a server accept a request. Image
        # tokens vary; the server performs the final full multimodal size check.
        if max_tokens >= self.profile['context_tokens']:
            raise CapacityError('Output budget exceeds endpoint context: '+self.profile['name'])
        if self.before_request: self.before_request()
        body = {'model':self.profile['model'], 'messages':[{'role':'user', 'content':[
            {'type':'text', 'text':prompt}]+[{'type':'image_url','image_url':{
                'url':'data:image/jpeg;base64,'+base64.b64encode(p.read_bytes()).decode()}} for p in images]}],
            'max_tokens':max_tokens, 'temperature':0, 'seed':42, 'cache_prompt':False,
            'chat_template_kwargs':{'enable_thinking':False},
            'response_format':{'type':'json_schema','json_schema':{'name':'annotation','strict':True,'schema':schema}}}
        attempt = folder/('http_'+str(time.time_ns())); attempt.mkdir()
        started = time.perf_counter(); key = digest(descriptor)
        atomic(attempt/'request.json', {'input_key':key, 'started_at':now(),
               'request_id':str(uuid.uuid4()), 'retry_policy':'manual_resume_after_transport_failure'})
        try:
            response = self.session.post(self.profile['base_url'].rstrip('/')+'/chat/completions',
                json=body, headers={**self.headers(), 'Idempotency-Key':key},
                timeout=self.profile.get('timeout_s',1200), allow_redirects=False)
        except requests.RequestException as exc:
            atomic(attempt/'failure.json', {'kind':type(exc).__name__, 'server_completion_unknown':True,
                   'seconds':time.perf_counter()-started})
            raise TransportPending('Transport failure; server completion unknown; inspect '+str(attempt)) from None
        # Responses may include request echoes. Redact credentials before saving.
        text = response.text
        env = self.profile.get('api_key_env'); secret = os.environ.get(env,'') if env else ''
        if secret: text = text.replace(secret, '[REDACTED]')
        (attempt/'response.json').write_text(text)
        atomic(attempt/'timing.json', {'seconds':time.perf_counter()-started, 'http_status':response.status_code,
               'finished_at':now(), 'model':self.profile['model']})
        if response.status_code != 200:
            if response.status_code == 400 and any(s in text.lower() for s in ('context','tokens','too long')):
                raise CapacityError('Endpoint cannot fit unchanged request budget; evidence: '+str(attempt))
            raise RuntimeError('Inference HTTP '+str(response.status_code)+'; evidence: '+str(attempt))
        raw = json.loads(text)
        if raw['model'] != self.profile['model'] or len(raw['choices']) != 1 or raw['choices'][0]['finish_reason'] != 'stop':
            raise ValueError('Wrong-model or incomplete response: '+str(attempt))
        data = json.loads(raw['choices'][0]['message']['content']); jsonschema.validate(data, schema)
        # Legacy temporal verifier hashes response.json at this exact location.
        (folder/'response.json').write_text(text)
        atomic(cached,data)
        atomic(folder/'receipt.json', {'input_key':key, 'sha256':sha(cached), 'attempt':str(attempt),
               'profile':self.profile, 'usage':raw.get('usage'), 'completed_at':now()})
        return data
