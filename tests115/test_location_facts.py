from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import jsonschema
from pipeline115.common import ROOT,read,sha
from pipeline115.location_facts import verify,attach,project_exports,contract

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.md=self.root/'episode.md';self.md.write_text('物体原始位置：1\n物体最后所在位置：黑色盒子里面\n')
        self.photo=self.root/'S03/S03便签纸——笔记本电脑旁边.jpg'
        self.photo.parent.mkdir();self.photo.write_bytes(b'fixture-photo-identity')
        self.doc=read(ROOT/'contracts/final_annotation.example.json')
        self.doc['metadata']['target']['description']='标签'
        self.row={'episode_id':self.doc['metadata']['episode_id'],'scene_group':'S03','target':'标签',
            'target_markdown':str(self.md),'target_markdown_sha256':sha(self.md),
            'event_index':{'events':self.doc['search_event_states']['events']}}
        self.facts={'schema':'owner_supplied_episode_locations.v1','episode_id':self.row['episode_id'],'scene_id':'S03','target_name':'标签',
            'initial_location':{'status':'recorded_filename','description':'笔记本电脑旁边','source_file':str(self.photo),
                'source_sha256':sha(self.photo),'filename':self.photo.name,'target_name_in_filename':'便签纸',
                'matching_rule':'same scene; owner-confirmed alias','raw_markdown_value':'1'},
            'actual_location':{'description':'黑色盒子里面','source_file':str(self.md),'source_sha256':sha(self.md),
                'field_name':'物体最后所在位置','source_type':'owner_confirmed_markdown',
                'temporal_scope':'episode_level_record_no_within_episode_relocation_time',
                'associated_event_ids':[e['event_id'] for e in self.row['event_index']['events']],
                'search_unit_id':None,'relocation_time_s':None},'interpretation':'Episode-level owner record, no unit or time inference'}
        self.row['dataset_location_facts']=self.facts
    def tearDown(self):self.tmp.cleanup()
    def test_preserve_visual_tree_outcomes_and_both_exports(self):
        baseline=deepcopy(self.doc);attach(self.doc,self.row)
        jsonschema.validate(self.doc,contract(read(ROOT/'contracts/final_annotation.schema.json')))
        for key in ('scene_memory','hidden','search_event_states','episode_outcome'):
            self.assertEqual(self.doc[key],baseline[key])
        a,b=project_exports(self.doc,{'hidden':{'true_unit_id':None}},{'hidden':{'true_unit_id':None}})
        for output in (a,b):
            self.assertEqual(output['hidden']['location_descriptions'],self.facts)
            self.assertIsNone(output['hidden']['true_unit_id'])
    def test_source_mutation_and_cross_scene_rejected(self):
        verify(self.row);self.photo.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'photo identity'):verify(self.row)
        self.photo.write_bytes(b'fixture-photo-identity');self.row['scene_group']='S04'
        with self.assertRaisesRegex(ValueError,'another episode'):verify(self.row)
    def test_event_scope_and_description_cannot_be_guessed(self):
        self.facts['actual_location']['description']='different location'
        with self.assertRaisesRegex(ValueError,'differs'):verify(self.row)
        self.facts['actual_location']['description']='黑色盒子里面'
        self.facts['actual_location']['associated_event_ids']=['another-episode:event']
        with self.assertRaisesRegex(ValueError,'association'):verify(self.row)

if __name__=='__main__':unittest.main()
