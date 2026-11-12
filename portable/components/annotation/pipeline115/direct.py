"""Publish validated frame/pose/depth references without writing or reading ROS bags."""
from pathlib import Path
import math
import os
import sys
import tempfile
import cv2
import numpy as np
from .common import CORE, atomic, digest, lock, now, read, rows, sha

SCHEMA = 'ego.direct_frames.v1'

def stage_records(preprocessed):
    # This function verifies artifact contents, not geometric accuracy. Preserve
    # VIO acceptance separately even when the upstream stage is complete.
    sys.path.insert(0, str(CORE/'src'))
    from daaam.preprocessing.common import verify_stage
    states = {name:verify_stage(Path(preprocessed)/name)
              for name in ('audit','vio','select','rectify','depth')}
    # A valid old output alone does not prove that the raw capture is unchanged.
    report = read(Path(preprocessed)/'audit/audit_report.json')
    if not report.get('input_sha256'): raise ValueError('Missing original capture hashes')
    for path,checksum in report['input_sha256'].items():
        if sha(path)!=checksum: raise ValueError('Preprocessed capture changed: '+path)
    return states

def indexed(values, label):
    result = {int(v['keyframe_index']):v for v in values}
    if len(result) != len(values): raise ValueError('Duplicate '+label+' keyframe index')
    return result

def reference(path):
    path = Path(path).resolve()
    return {'path':str(path), 'sha256':sha(path)}

def image(path, shape, dtype, channels=None):
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    expected = tuple(shape) if channels is None else (*shape,channels)
    if value is None or value.shape != expected or value.dtype != dtype:
        raise ValueError('Invalid image shape/encoding: '+str(path))
    return value

def validate(folder, expected_key=None):
    folder = Path(folder); receipt = read(folder/'ready.json')
    if receipt['schema'] != SCHEMA or receipt['status'] != 'passed':
        raise ValueError('Invalid direct-package readiness receipt')
    if expected_key is not None and receipt['input_key'] != expected_key:
        raise ValueError('Direct-package inputs changed')
    actual = {str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*'))
              if p.is_file() and p.name != 'ready.json'}
    if actual != receipt['files']: raise ValueError('Direct-package files changed')
    for path, checksum in receipt['external_artifacts'].items():
        if sha(path) != checksum: raise ValueError('Direct-package upstream changed: '+path)
    records = rows(folder/'all_frames.jsonl')
    if not records or len(records) != receipt['frame_count']:
        raise ValueError('Direct-package frame count mismatch')
    last_frame = last_ns = -1
    for index, record in enumerate(records):
        if record['keyframe_index'] != index or record['sensor_time_ns'] <= last_ns or record['original_video_frame_index'] <= last_frame:
            raise ValueError('Invalid direct-package order')
        if sha(folder/record['rgb_path']) != record['rgb_sha256']:
            raise ValueError('Direct-package RGB hash mismatch')
        for key in ('rectified_left','rectified_right','depth','confidence'):
            ref = record['direct_evidence'][key]
            if sha(ref['path']) != ref['sha256']: raise ValueError('Referenced evidence changed')
        last_frame = record['original_video_frame_index']; last_ns = record['sensor_time_ns']
    return receipt

def build(row, preprocessed, folder, jpeg_quality=90):
    """Keep legacy model-visible JPEG encoding and metre-based TF exactly."""
    preprocessed = Path(preprocessed).resolve(); folder = Path(folder).resolve()
    folder.parent.mkdir(parents=True, exist_ok=True)
    with lock(folder.parent/('.'+folder.name+'.publish.lock')):
        stages = stage_records(preprocessed)
        descriptor = {'schema':SCHEMA, 'episode_id':row['episode_id'],
            'preprocessed':str(preprocessed), 'stages':stages,
            'fps':row['fps'], 'jpeg_quality':jpeg_quality,
            'source_metadata_sha256':row['metadata_sha256'],
            'target_markdown_sha256':row['target_markdown_sha256']}
        key = digest(descriptor)
        if folder.exists(): return validate(folder,key)
        if not 1 <= jpeg_quality <= 100: raise ValueError('Invalid JPEG quality')
        if not math.isfinite(row['fps']) or row['fps'] <= 0: raise ValueError('Invalid FPS')
        rectify = preprocessed/'rectify'; depth = preprocessed/'depth'
        frames = rows(rectify/'frame_manifest.jsonl')
        poses = indexed(rows(rectify/'pose_samples_mm.jsonl'),'pose')
        depths = indexed(rows(depth/'depth_manifest.jsonl'),'depth')
        expected = set(range(len(frames)))
        if not frames or set(poses) != expected or set(depths) != expected:
            raise ValueError('RGB, pose and depth keyframe sets differ')
        camera = read(rectify/'rectified_camera.json')
        shape = (int(camera['height']),int(camera['width']))
        pending = Path(tempfile.mkdtemp(prefix='.'+folder.name+'-', dir=folder.parent))
        try:
            atomic(pending/'inputs.json', descriptor)
            atomic(pending/'camera.json',camera)
            acceptance = read(preprocessed/'vio/native/vio_acceptance.json')
            atomic(pending/'vio_acceptance.json',acceptance)
            output = []; last_ns = last_frame = -1
            for index, frame in enumerate(frames):
                if frame['keyframe_index'] != index: raise ValueError('Frame index is not contiguous')
                pose, d = poses[index], depths[index]; ns = int(frame['sensor_time_ns'])
                frame_no = int(frame['left_video_frame_idx'])
                if ns <= last_ns or frame_no <= last_frame: raise ValueError('Nonmonotonic frames/timestamps')
                if any(int(v['sensor_time_ns']) != ns or v['source_idx'] != frame['source_idx'] for v in (pose,d)):
                    raise ValueError('Pose/depth synchronization mismatch')
                if frame['reference_time_ns'] != ns or pose['reference_time_ns'] != ns or d['reference_time_ns'] != ns:
                    raise ValueError('Reference timestamp mismatch')
                if pose['frame_id'] != 'vio_world' or pose['child_frame_id'] != 'cam0_rect' or frame['rectified_left_frame'] != 'cam0_rect':
                    raise ValueError('Unexpected camera/reference frame')
                left = rectify/frame['left_image_path']; right = rectify/frame['right_image_path']
                depth_path = depth/d['depth_path']; confidence_path = depth/d['confidence_path']
                image(left,shape,np.uint8,3); image(right,shape,np.uint8,3)
                image(depth_path,shape,np.uint16); image(confidence_path,shape,np.uint8)
                if d['depth_unit'] != 'mm' or d['encoding'] != '16UC1' or d['confidence_encoding'] != 'mono8':
                    raise ValueError('Depth/confidence encoding mismatch')
                # Matches ros_output.compressed: IMREAD_COLOR + imencode JPEG90.
                ok, encoded = cv2.imencode('.jpg',cv2.imread(str(left),cv2.IMREAD_COLOR),
                                           [cv2.IMWRITE_JPEG_QUALITY,jpeg_quality])
                if not ok: raise ValueError('JPEG encoding failed')
                name = f'f{frame_no:07d}.jpg'; (pending/name).write_bytes(encoded.tobytes())
                position = (np.asarray(pose['position_mm'],float)/1000.).tolist()
                quaternion = list(map(float,pose['orientation_xyzw']))
                covariance = np.asarray(pose['pose_covariance_mm_rad'],float).reshape(6,6)
                scale = np.diag([.001,.001,.001,1.,1.,1.])
                covariance_m = scale @ covariance @ scale.T
                if not np.isfinite(covariance_m).all() or not all(math.isfinite(x) for x in position+quaternion):
                    raise ValueError('Nonfinite pose/covariance')
                if abs(sum(x*x for x in quaternion)-1) > .02: raise ValueError('Invalid pose quaternion')
                record = {'frame_id':row['episode_id']+f':direct:{index:06d}',
                    'keyframe_index':index, 'original_video_frame_index':frame_no,
                    'original_video_seconds':frame_no/row['fps'],
                    'rgb_path':name, 'rgb_sha256':sha(pending/name),
                    'width':shape[1], 'height':shape[0], 'sensor_time_ns':ns,
                    'camera_frame':'cam0_rect', 'transforms':[{'parent':'vio_world',
                    'child':'cam0_rect', 'sensor_time_ns':ns, 'translation_m':position,
                    'quaternion_xyzw':quaternion}],
                    'pose':{'max_position_std_m':math.sqrt(max(0,*np.diag(covariance_m)[:3])),
                            'quality':'unverified_VIO_covariance_valid'},
                    'direct_evidence':{'rectified_left':reference(left),
                        'rectified_right':reference(right), 'depth':reference(depth_path),
                        'confidence':reference(confidence_path), 'depth_unit':'mm',
                        'confidence_encoding':'mono8', 'frame_metadata':frame,
                        'pose_mm_rad':pose, 'depth_metadata':d,
                        'pose_covariance_m_rad':covariance_m.tolist()}}
                output.append(record); last_ns = ns; last_frame = frame_no
            import json
            (pending/'all_frames.jsonl').write_text(''.join(json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n' for v in output))
            atomic(pending/'source_manifest.json', {'schema':SCHEMA,
                'preprocessed':str(preprocessed), 'frame_count':len(output),
                'source_capture':row['source'], 'source_metadata_sha256':row['metadata_sha256'],
                'geometry_scope':'Unverified VIO camera prior; not metric object localization.',
                'transport':'direct_frames_no_rosbag', 'vio_acceptance_passed':acceptance['passed']})
            external = {}
            for name,state in stages.items():
                external[str(preprocessed/name/'stage.json')] = sha(preprocessed/name/'stage.json')
                for relative,checksum in state['artifacts'].items():
                    external[str(preprocessed/name/relative)] = checksum
            receipt = {'schema':SCHEMA,'status':'passed','input_key':key,
                'frame_count':len(output),'vio_acceptance_passed':acceptance['passed'],
                'files':{str(p.relative_to(pending)):sha(p) for p in sorted(pending.rglob('*')) if p.is_file()},
                'external_artifacts':external,'completed_at':now(),
                'validation_scope':'Direct-package integrity, decoding, synchronization, units and hashes; no ROS deserialization claim.'}
            atomic(pending/'ready.json',receipt)
            validate(pending,key)
            os.rename(pending,folder)
            return receipt
        except BaseException as exc:
            atomic(pending/'failure.json',{'error':repr(exc),'at':now()})
            raise
