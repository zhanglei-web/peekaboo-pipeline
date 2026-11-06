from pathlib import Path
import tempfile
import unittest
from pipeline115.common import atomic
from pipeline115.batch_sequence import predecessor_complete,summary,Sequence


class Tests(unittest.TestCase):
    def test_gate_requires_exact_count_and_final_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);state={'episodes':{'a':{'status':'complete'}}}
            self.assertFalse(predecessor_complete(state,root,1))
            atomic(root/'final_validation.json',{'status':'passed','validated':1})
            self.assertTrue(predecessor_complete(state,root,1))
            self.assertFalse(predecessor_complete(state,root,72))
            state['episodes']['a']['status']='needs_repair'
            self.assertFalse(predecessor_complete(state,root,1))

    def test_skips_are_reported_without_claiming_full_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            atomic(root/'state.json',{'episodes':{'a':{'status':'complete'},
                'b':{'status':'needs_repair','next':'cpu','error':'Frame identity mismatch'}}})
            atomic(root/'enrollment.json',{'skipped':[{'episode_id':'c','reason':'Missing image'}]})
            result=summary(root,root/'enrollment.json')
            self.assertEqual(result['status'],'processed_with_skips')
            self.assertEqual(result['complete_count'],1);self.assertEqual(result['skipped_count'],2)
            self.assertEqual(result['input_total'],3)

    def test_predecessor_identity_errors_are_not_transport_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'logs/a';folder.mkdir(parents=True)
            (folder/'astra-1.log').write_text('ValueError: model identity mismatch')
            self.assertFalse(Sequence.transport_failure(root,'a'))
            (folder/'astra-1.log').write_text('astra_memory.provider.TransportError: timeout')
            self.assertTrue(Sequence.transport_failure(root,'a'))

if __name__=='__main__':unittest.main()
