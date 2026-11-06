from copy import deepcopy
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import json
import os
import tempfile
import threading
import unittest
from pipeline115.client import Client,CapacityError,YieldGPU

SCHEMA={'type':'object','properties':{'value':{'type':'integer'}},'required':['value'],'additionalProperties':False}

class Server(BaseHTTPRequestHandler):
    response_model='test-model'; finish_reason='stop'; requests=0; status=200; content={'value':7}
    def log_message(self,*args): pass
    def do_GET(self):
        self.send_response(200);self.end_headers()
        self.wfile.write(json.dumps({'data':[{'id':'test-model','meta':{'n_ctx':1000}}]}).encode())
    def do_POST(self):
        cls=type(self);cls.requests+=1
        raw=self.rfile.read(int(self.headers['Content-Length']));body=json.loads(raw)
        assert body['max_tokens']==100 and body['chat_template_kwargs']=={'enable_thinking':False}
        assert self.headers['Authorization']=='Bearer test-secret-for-redaction'
        self.send_response(cls.status);self.end_headers()
        if cls.status!=200:self.wfile.write(b'context exceeded test-secret-for-redaction');return
        result={'model':cls.response_model,'choices':[{'finish_reason':cls.finish_reason,'message':{'content':json.dumps(cls.content)}}]}
        self.wfile.write(json.dumps(result).encode())

class Tests(unittest.TestCase):
    def setUp(self):
        Server.response_model='test-model';Server.finish_reason='stop';Server.requests=0;Server.status=200;Server.content={'value':7}
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Server)
        self.thread=threading.Thread(target=self.server.serve_forever);self.thread.start()
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.profile={'name':'fake','base_url':f'http://127.0.0.1:{self.server.server_port}/v1','model':'test-model','context_tokens':1000,'api_key_env':'PIPELINE115_TEST_KEY','placement':'remote'}
        os.environ['PIPELINE115_TEST_KEY']='test-secret-for-redaction'
    def tearDown(self):
        self.server.shutdown();self.thread.join();self.server.server_close();self.tmp.cleanup();os.environ.pop('PIPELINE115_TEST_KEY')
    def call(self,folder='one',profile=None,**kwargs):
        return Client(profile or self.profile,**kwargs)(self.root/folder,'prompt',[],SCHEMA,100)
    def test_cache_and_endpoint_identity(self):
        self.assertEqual(self.call(),{'value':7});self.call();self.assertEqual(Server.requests,1)
        p=deepcopy(self.profile);p['quantization']='another'
        with self.assertRaisesRegex(ValueError,'Immutable input'):self.call(profile=p)
    def test_episode_probe_checks_actual_server_capacity(self):
        self.assertEqual(Client(self.profile).probe(require_context=True)['context_tokens'],1000)
        p=deepcopy(self.profile);p['context_tokens']=2000
        with self.assertRaisesRegex(CapacityError,'below configured capacity'):
            Client(p).probe(require_context=True)
    def test_hash_mutation_rejected(self):
        self.call();(self.root/'one/result.json').write_text('{"value":8}')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.call()
    def test_wrong_model(self):
        Server.response_model='other'
        with self.assertRaisesRegex(ValueError,'Wrong-model'):self.call()
        self.assertFalse((self.root/'one/result.json').exists())
    def test_truncated(self):
        Server.finish_reason='length'
        with self.assertRaisesRegex(ValueError,'incomplete'):self.call()
    def test_schema_error(self):
        Server.content={'value':'not integer'}
        with self.assertRaises(Exception):self.call()
        self.assertFalse((self.root/'one/result.json').exists())
    def test_context_and_redaction(self):
        Server.status=400
        with self.assertRaises(CapacityError):self.call()
        for path in self.root.rglob('*'):
            if path.is_file():self.assertNotIn('test-secret-for-redaction',path.read_text())
    def test_redirect_not_followed(self):
        Server.status=302
        with self.assertRaisesRegex(RuntimeError,'HTTP 302'):self.call()
        self.assertEqual(Server.requests,1)
    def test_oversized_budget_before_http(self):
        p=deepcopy(self.profile);p['context_tokens']=100
        with self.assertRaises(CapacityError):self.call(profile=p)
        self.assertEqual(Server.requests,0)
    def test_request_boundary_yield_and_cached_resume(self):
        def yield_now():raise YieldGPU()
        with self.assertRaises(YieldGPU):self.call(before_request=yield_now)
        self.assertEqual(Server.requests,0);self.call()
        self.call(before_request=yield_now);self.assertEqual(Server.requests,1)

if __name__=='__main__':unittest.main()
