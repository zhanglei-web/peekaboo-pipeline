from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
import json
import sys
import tempfile
import threading
import unittest
from pipeline115.common import read
from pipeline115 import selector_transport as transport

sys.path.insert(0,'/home/baai/zl/egomemory_ws/src/astra-scene-memory/src')
from astra_memory.provider import TransportError,ModelServiceError


class Handler(BaseHTTPRequestHandler):
    mode='complete';requests=[]
    def log_message(self,*args):pass
    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        type(self).requests.append((body,self.headers.get('Idempotency-Key')))
        self.send_response(500 if self.mode=='http500' else 200)
        self.send_header('Content-Type','text/event-stream');self.end_headers()
        if self.mode=='http500':return
        created={'type':'response.created','response':{'id':'resp_test','model':'gpt-6.1-sol','status':'in_progress'}}
        self.wfile.write(('data: '+json.dumps(created)+'\n\n').encode())
        if self.mode=='truncated':return
        final={'id':'resp_test','model':'gpt-6.1-sol','status':'completed',
            'output':[{'type':'message','content':[{'type':'output_text','text':'{"ok": true}'}]}]}
        if self.mode=='missing':final.pop('model')
        if self.mode=='wrong':final['model']='other'
        if self.mode=='changed_id':final['id']='resp_other'
        if self.mode=='secret_echo':final['metadata']={'echo':'transport-test-secret'}
        event={'type':'response.completed','response':final}
        self.wfile.write(('data: '+json.dumps(event)+'\n\n').encode())


class Tests(unittest.TestCase):
    def setUp(self):
        Handler.mode='complete';Handler.requests=[]
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever);self.thread.start()
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.model={'name':'gpt-6.1-sol','base_url':f'http://127.0.0.1:{self.server.server_port}',
                    'transport':'responses'}
        self.body={'model':'gpt-6.1-sol','max_output_tokens':5000,'input':[{'evidence':'unchanged'}]}
    def tearDown(self):
        self.server.shutdown();self.thread.join();self.server.server_close();self.tmp.cleanup()
    def call(self):
        with patch.object(transport,'POLICY',{**transport.POLICY,'attempts':1}):
            return transport.request_verified(self.root,self.model,self.body,'transport-test-secret','signature')
    def test_completed_stream_keeps_request_and_exact_identity(self):
        raw,text=self.call();self.assertEqual(raw['model'],'gpt-6.1-sol');self.assertEqual(json.loads(text),{'ok':True})
        self.assertEqual(Handler.requests[0][0],{**self.body,'stream':True})
        entries=[read(p) for p in (self.root/'transport_attempts').glob('*.json') if '.response.' not in p.name]
        self.assertEqual(entries[0]['status'],'verified')
    def test_missing_model_rejected_even_when_created_event_has_model(self):
        Handler.mode='missing'
        with self.assertRaisesRegex(ModelServiceError,'omitted model identity'):self.call()
    def test_wrong_model_and_response_id_rejected(self):
        Handler.mode='wrong'
        with self.assertRaisesRegex(ModelServiceError,'exactly match'):self.call()
        Handler.mode='changed_id'
        with self.assertRaisesRegex(ModelServiceError,'identity changed'):self.call()
    def test_incomplete_stream_is_not_a_success(self):
        Handler.mode='truncated'
        with self.assertRaisesRegex(TransportError,'before completed'):self.call()
        self.assertFalse((self.root/'response.json').exists())
    def test_http_500_is_retryable(self):
        Handler.mode='http500'
        with self.assertRaises(TransportError) as caught:self.call()
        self.assertTrue(caught.exception.retryable)
    def test_credentials_redacted_from_saved_events(self):
        Handler.mode='secret_echo';self.call()
        for p in self.root.rglob('*'):
            if p.is_file():self.assertNotIn('transport-test-secret',p.read_text())

if __name__=='__main__':unittest.main()
