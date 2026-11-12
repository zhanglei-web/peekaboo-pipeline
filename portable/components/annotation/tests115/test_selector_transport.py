from pathlib import Path
REPOSITORY = next(p for p in Path(__file__).resolve().parents if (p / 'repository.marker').is_file())
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
import json
import requests
import sys
import tempfile
import threading
import time
import unittest
from pipeline115.common import read
from pipeline115 import selector_transport as transport

sys.path.insert(0,str(REPOSITORY / 'components/semantic/integrations/astra_pipeline/src'))
from astra_memory.provider import TransportError,ModelServiceError


class Handler(BaseHTTPRequestHandler):
    mode='complete';requests=[]
    def log_message(self,*args):pass
    def do_POST(self):
        if self.mode=='slow_upload':
            time.sleep(.3)
        try:
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        except (ValueError, OSError):
            return
        type(self).requests.append((body,self.headers.get('Idempotency-Key')))
        self.send_response(500 if self.mode=='http500' else 302 if self.mode=='redirect' else 200)
        self.send_header('Content-Type','application/json' if self.mode=='non_sse' else 'text/event-stream');self.end_headers()
        if self.mode in ('http500','redirect','non_sse'):return
        created={'type':'response.created','response':{'id':'resp_test','model':'gpt-6.1-sol','status':'in_progress'}}
        self.wfile.write(('data: '+json.dumps(created)+'\n\n').encode())
        if self.mode=='truncated':return
        final={'id':'resp_test','model':'gpt-6.1-sol','status':'completed',
            'output':[{'type':'message','content':[{'type':'output_text','text':'{"ok": true}'}]}]}
        if self.mode=='missing':final.pop('model')
        if self.mode=='wrong':final['model']='other'
        if self.mode=='changed_id':final['id']='resp_other'
        if self.mode=='secret_echo':final['metadata']={'echo':'transport-test-secret'}
        if self.mode in ('whitespace','whitespace_only','multiple_json','whitespace_refusal'):
            final['output'].insert(0,{'type':'message','content':[{'type':'output_text','text':' \n\t'}]})
        if self.mode=='whitespace_only':final['output'].pop()
        if self.mode=='multiple_json':
            final['output'].append({'type':'message','content':[{'type':'output_text','text':'{"other": true}'}]})
        if self.mode=='whitespace_refusal':
            final['output'][0]['content'].append({'type':'refusal','refusal':'refused'})
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
        entries=[read(p) for p in (self.root/'transport_attempts').glob('*.json')
                 if '.response.' not in p.name and '.http.' not in p.name]
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
    def test_whitespace_message_keeps_original_and_records_extraction(self):
        Handler.mode='whitespace'
        raw,text=self.call()
        self.assertEqual(json.loads(text),{'ok':True})
        self.assertEqual(len(raw['output']),2)
        self.assertEqual(read(self.root/'response.json'),raw)
        entries=[read(p) for p in (self.root/'transport_attempts').glob('*.json')
                 if '.response.' not in p.name and '.http.' not in p.name]
        self.assertEqual(entries[0]['ignored_whitespace_output_text'],
                         [{'output_index':0,'content_index':0,'characters':3}])
    def test_whitespace_does_not_accept_missing_or_multiple_substantive_outputs(self):
        for mode in ('whitespace_only','multiple_json'):
            Handler.mode=mode
            with self.assertRaisesRegex(ValueError,'exactly one'):self.call()
    def test_whitespace_does_not_hide_refusal(self):
        Handler.mode='whitespace_refusal'
        with self.assertRaisesRegex(ModelServiceError,'refused'):self.call()
    def test_curl_keeps_complete_large_upload_and_cleans_temporary_request(self):
        Handler.mode='slow_upload'
        body={**self.body,'large_upload':'x'*(8*1024*1024)}
        with patch.object(transport,'POLICY',{**transport.POLICY,'connect_timeout_s':.05,'read_timeout_s':5}):
            raw=transport.stream_response(self.model['base_url']+'/responses',body,'transport-test-secret','large-curl',self.root/'curl.events.jsonl')
        self.assertEqual(raw['status'],'completed')
        self.assertEqual(Handler.requests[-1][0],{**body,'stream':True})
        self.assertFalse(list(self.root.glob('.selector-upload-*')))
        self.assertEqual(read(self.root/'curl.events.http.json')['http_status'],200)
    def test_redirect_and_non_sse_are_rejected(self):
        Handler.mode='redirect'
        with self.assertRaises(TransportError) as caught:self.call()
        self.assertFalse(caught.exception.retryable)
        Handler.mode='non_sse'
        with self.assertRaisesRegex(ModelServiceError,'non-SSE'):self.call()
    def test_http2_policy_is_additive_to_original_policy(self):
        original={'transport':'responses_sse','connect_timeout_s':30}
        from pipeline115.common import atomic
        atomic(self.root/'stream_transport_policy.json',original)
        self.call()
        self.assertEqual(read(self.root/'stream_transport_policy.json'),original)
        self.assertEqual(read(self.root/'stream_transport_policy.http2.json')['transport'],'curl_http2_responses_sse')
    def test_upload_uses_connect_socket_deadline_until_request_is_written(self):
        Handler.mode='slow_upload'
        body={**self.body,'upload_regression':'x'*(8*1024*1024)}
        url=self.model['base_url']+'/responses'
        # An unconsumed large body exceeds the short socket deadline even
        # though the response read deadline is much longer.
        with patch.object(transport,'POLICY',{**transport.POLICY,'connect_timeout_s':.05,'read_timeout_s':5}):
            with self.assertRaises(TransportError):
                transport.stream_response(url,body,'transport-test-secret','upload-short',self.root/'short.events.jsonl',post=requests.Session().post)
        with patch.object(transport,'POLICY',{**transport.POLICY,'connect_timeout_s':5,'read_timeout_s':5}):
            raw=transport.stream_response(url,body,'transport-test-secret','upload-long',self.root/'long.events.jsonl',post=requests.Session().post)
        self.assertEqual(raw['status'],'completed')
        self.assertEqual(Handler.requests[-1][0],{**body,'stream':True})

if __name__=='__main__':unittest.main()
