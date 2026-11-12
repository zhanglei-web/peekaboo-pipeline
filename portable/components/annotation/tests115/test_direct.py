from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import tempfile
import unittest
import cv2
import numpy as np
from pipeline115.common import atomic,sha
from pipeline115.direct import build,validate,stage_records

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.pre=self.root/'pre';self.out=self.root/'package'
        for name in ('audit','vio','select','rectify','depth'):
            (self.pre/name).mkdir(parents=True);atomic(self.pre/name/'stage.json',{'test_fixture':True})
        atomic(self.pre/'vio/native/vio_acceptance.json',{'passed':False,'gates':{'test':False}})
        atomic(self.pre/'rectify/rectified_camera.json',{'width':4,'height':3})
        self.frame={'keyframe_index':0,'source_idx':2,'left_video_frame_idx':2,'right_video_frame_idx':2,
            'sensor_time_ns':100,'reference_time_ns':100,'rectified_left_frame':'cam0_rect',
            'left_image_path':'left.png','right_image_path':'right.png'}
        self.pose={'keyframe_index':0,'source_idx':2,'sensor_time_ns':100,'reference_time_ns':100,
            'frame_id':'vio_world','child_frame_id':'cam0_rect','position_mm':[1000,2000,3000],
            'orientation_xyzw':[0,0,0,1],'pose_covariance_mm_rad':np.diag([100,200,300,1,2,3]).tolist()}
        self.depth={'keyframe_index':0,'source_idx':2,'sensor_time_ns':100,'reference_time_ns':100,
            'depth_path':'depth.png','confidence_path':'confidence.png','depth_unit':'mm',
            'encoding':'16UC1','confidence_encoding':'mono8'}
        for name in ('left.png','right.png'):cv2.imwrite(str(self.pre/'rectify'/name),np.zeros((3,4,3),np.uint8))
        cv2.imwrite(str(self.pre/'depth/depth.png'),np.full((3,4),1000,np.uint16))
        cv2.imwrite(str(self.pre/'depth/confidence.png'),np.full((3,4),255,np.uint8))
        self.row={'episode_id':'test','fps':30,'metadata_sha256':'test-meta','target_markdown_sha256':'test-md','source':'test-capture'}
    def tearDown(self):self.tmp.cleanup()
    def run_build(self):
        import json
        for path,value in [('rectify/frame_manifest.jsonl',self.frame),('rectify/pose_samples_mm.jsonl',self.pose),('depth/depth_manifest.jsonl',self.depth)]:
            (self.pre/path).write_text(json.dumps(value)+'\n')
        stages={name:{'artifacts':{str(p.relative_to(self.pre/name)):sha(p) for p in (self.pre/name).rglob('*') if p.is_file() and p.name!='stage.json'}} for name in ('audit','vio','select','rectify','depth')}
        with patch('pipeline115.direct.stage_records',return_value=stages):return build(self.row,self.pre,self.out)
    def test_quality_preserved_and_references_immutable(self):
        self.assertFalse(self.run_build()['vio_acceptance_passed']);validate(self.out)
        (self.pre/'rectify/left.png').write_bytes(b'changed')
        with self.assertRaises(ValueError):validate(self.out)
    def test_bad_timestamp_not_published(self):
        self.pose['sensor_time_ns']=101
        with self.assertRaisesRegex(ValueError,'synchronization'):self.run_build()
        self.assertFalse(self.out.exists())
    def test_depth_encoding_not_relaxed(self):
        cv2.imwrite(str(self.pre/'depth/depth.png'),np.ones((3,4),np.uint8))
        with self.assertRaisesRegex(ValueError,'encoding'):self.run_build()
        self.assertFalse(self.out.exists())
    def test_no_ros_import_and_units(self):
        self.run_build()
        from pipeline115.common import rows
        record=rows(self.out/'all_frames.jsonl')[0]
        self.assertEqual(record['transforms'][0]['translation_m'],[1.,2.,3.])
        self.assertAlmostEqual(record['direct_evidence']['pose_covariance_m_rad'][0][0],.0001)
        self.assertFalse((self.pre/'ros_output').exists())
    def test_reuse_rejects_changed_original_capture(self):
        import sys
        from pipeline115.common import CORE
        sys.path.insert(0,str(CORE/'src'))
        raw=self.root/'capture.bin';raw.write_bytes(b'original')
        atomic(self.pre/'audit/audit_report.json',{'input_sha256':{str(raw):sha(raw)}})
        with patch('daaam.preprocessing.common.verify_stage',return_value={'status':'complete'}):
            self.assertEqual(len(stage_records(self.pre)),5)
            raw.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'capture changed'):stage_records(self.pre)

if __name__=='__main__':unittest.main()
