from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import os
import tempfile
import unittest
from pipeline115.common import atomic,read
from pipeline115.selector import call


class Tests(unittest.TestCase):
    def test_configured_selector_keeps_evidence_budget_and_binds_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);image=root/'frame.jpg';image.write_bytes(b'evidence')
            cfg=root/'model.json';atomic(cfg,{'model':{'name':'gpt-6.1-sol','api_key_env':'SELECTOR_TEST_KEY','timeout_s':180}})
            result={'selected':[{'frame':7,'coverage':['table'],'why_needed':'clear','limitations':'occluded'}],
                    'coverage_gaps':[],'selection_rationale':'coverage'}
            from pipeline115.worker import setup
            setup({'output_root':tmp})
            import select_scene_keyframes as selector
            sent=[]
            def body(model,prompt,schema,context,images):
                sent.append((model,prompt,schema,context,images));return {'model':model['name']}
            def request(folder,model,body,key,signature):
                self.assertEqual(key,'test-secret');self.assertEqual(model['max_output_tokens'],5000)
                self.assertEqual(model['timeout_s'],300);self.assertEqual(body['model'],'gpt-6.1-sol')
                return {'model':'gpt-6.1-sol'},json.dumps(result)
            os.environ['SELECTOR_TEST_KEY']='test-secret'
            try:
                with patch.object(selector,'make_body',side_effect=body),patch('pipeline115.selector.request_verified',side_effect=request) as req:
                    self.assertEqual(call(selector,cfg,root,'coarse',[image],[7],1,'original prompt',{'camera':'prior'}),result)
                    call(selector,cfg,root,'coarse',[image],[7],1,'original prompt',{'camera':'prior'})
                    self.assertEqual(req.call_count,1)
                self.assertEqual(sent[0][1],'original prompt');self.assertEqual(sent[0][3]['tf_viewpoint_evidence'],{'camera':'prior'})
                self.assertEqual(sent[0][4][0]['path'],str(image))
                atomic(cfg,{'model':{'name':'other','api_key_env':'SELECTOR_TEST_KEY','timeout_s':180}})
                with self.assertRaisesRegex(ValueError,'Immutable input'):call(selector,cfg,root,'coarse',[image],[7],1,'original prompt',{'camera':'prior'})
            finally:os.environ.pop('SELECTOR_TEST_KEY',None)

    def test_initial_photo_uses_same_scene_and_preserves_actual_filename(self):
        from pipeline115.enroll_followup import initial_photo
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'原始位置照片/S09';p.mkdir(parents=True);(p/'胶水中间抽屉里面.jpg').write_bytes(b'image')
            result=initial_photo(root,'S09','胶水','1')
            self.assertEqual(result['filename'],'胶水中间抽屉里面.jpg');self.assertEqual(result['description'],'中间抽屉里面')
            with self.assertRaises(ValueError):initial_photo(root,'S08','胶水','1')
            with self.assertRaises(ValueError):initial_photo(root,'S09','胶水','黑盒里面')

if __name__=='__main__':unittest.main()
