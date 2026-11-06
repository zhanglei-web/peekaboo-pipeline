from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from .common import covariance_units, execute_stage, read_json, read_rows, write_json


def _messages(rectify, depth, *, prefix, image_format, jpeg_quality):
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
    from sensor_msgs.msg import CameraInfo, CompressedImage, Image
    from std_msgs.msg import String
    from tf2_msgs.msg import TFMessage

    rectified = read_json(rectify/'rectified_camera.json')
    frames = read_rows(rectify/'frame_manifest.jsonl')
    poses = {int(x['keyframe_index']): x for x in read_rows(rectify/'pose_samples_mm.jsonl')}
    depths = {int(x['keyframe_index']): x for x in read_rows(depth/'depth_manifest.jsonl')}
    width, height = int(rectified['width']), int(rectified['height'])
    if set(poses) != set(range(len(frames))) or set(depths) != set(range(len(frames))):
        raise ValueError('rectified image, pose and depth frame sets are not identical')
    topics = {'left': f'{prefix}/left/image_rect/compressed',
              'right': f'{prefix}/right/image_rect/compressed',
              'left_info': f'{prefix}/left/camera_info', 'right_info': f'{prefix}/right/camera_info',
              'depth': f'{prefix}/depth/image', 'confidence': f'{prefix}/depth/confidence',
              'pose': f'{prefix}/pose', 'tf': '/tf', 'metadata': f'{prefix}/frame_metadata'}

    def stamp(ns):
        value = Time(); value.sec = int(ns//1_000_000_000); value.nanosec = int(ns%1_000_000_000)
        return value

    def camera_info(ns, frame_id, projection):
        msg = CameraInfo(); msg.header.stamp = stamp(ns); msg.header.frame_id = frame_id
        msg.width = width; msg.height = height; msg.distortion_model = 'plumb_bob'
        msg.d = [0., 0., 0., 0., 0.]
        K = rectified['K']
        msg.k = [K[0][0], K[0][1], K[0][2], K[1][0], K[1][1], K[1][2], K[2][0], K[2][1], K[2][2]]
        msg.r = [1.,0.,0.,0.,1.,0.,0.,0.,1.]
        P = np.asarray(projection, float).copy(); P[:, 3] /= 1000.
        msg.p = P.reshape(-1).tolist()
        return msg

    def compressed(path, ns, frame_id):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (height, width):
            raise ValueError(f'invalid rectified image: {path}')
        ext = '.jpg' if image_format == 'jpeg' else '.png'
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if image_format == 'jpeg' else [cv2.IMWRITE_PNG_COMPRESSION, 1]
        ok, buffer = cv2.imencode(ext, image, params)
        if not ok:
            raise ValueError(f'image compression failed: {path}')
        msg = CompressedImage(); msg.header.stamp = stamp(ns); msg.header.frame_id = frame_id
        msg.format = image_format; msg.data = buffer.tobytes()
        return msg

    def raw_image(path, ns, frame_id, encoding):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape[:2] != (height, width):
            raise ValueError(f'invalid depth product: {path}')
        if encoding == '16UC1' and (image.dtype != np.uint16 or image.ndim != 2):
            raise ValueError(f'depth must be uint16 millimetres: {path}')
        if encoding == 'mono8' and (image.dtype != np.uint8 or image.ndim != 2):
            raise ValueError(f'confidence must be uint8 mono: {path}')
        image = np.ascontiguousarray(image)
        msg = Image(); msg.header.stamp = stamp(ns); msg.header.frame_id = frame_id
        msg.height, msg.width = height, width; msg.encoding = encoding; msg.is_bigendian = 0
        msg.step = width * image.dtype.itemsize; msg.data = image.tobytes()
        return msg

    for index, frame in enumerate(frames):
        pose, depth_row = poses[index], depths[index]
        ns = int(frame['sensor_time_ns'])
        if any(int(row['sensor_time_ns']) != ns for row in (pose, depth_row)):
            raise ValueError(f'timestamp mismatch at keyframe {index}')
        left_frame, right_frame = 'cam0_rect', 'cam1_rect'
        left = compressed(rectify/frame['left_image_path'], ns, left_frame)
        right = compressed(rectify/frame['right_image_path'], ns, right_frame)
        left_info = camera_info(ns, left_frame, rectified['P_left_mm'])
        right_info = camera_info(ns, right_frame, rectified['P_right_mm'])
        dpath = depth/depth_row['depth_path']; cpath = depth/depth_row['confidence_path']
        depth_msg = raw_image(dpath, ns, left_frame, '16UC1')
        confidence = raw_image(cpath, ns, left_frame, 'mono8')
        pose_msg = PoseWithCovarianceStamped(); pose_msg.header.stamp = stamp(ns); pose_msg.header.frame_id = 'vio_world'
        position = np.asarray(pose['position_mm'], float)/1000.
        pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y, pose_msg.pose.pose.position.z = position.tolist()
        q = pose['orientation_xyzw']
        pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w = map(float, q)
        pose_msg.pose.covariance = covariance_units(pose['pose_covariance_mm_rad'], .001).reshape(-1).tolist()
        transform = TransformStamped(); transform.header.stamp = stamp(ns)
        transform.header.frame_id = 'vio_world'; transform.child_frame_id = left_frame
        transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = position.tolist()
        transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w = map(float, q)
        tf = TFMessage(); tf.transforms = [transform]
        metadata = String(); metadata.data = json.dumps({'schema': 'ego.frame_metadata.v1',
                  'frame_id': frame['frame_id'], 'source_idx': frame['source_idx'],
                  'left_source_idx': frame['left_source_idx'], 'right_source_idx': frame['right_source_idx'],
                  'left_video_frame_idx': frame['left_video_frame_idx'],
                  'right_video_frame_idx': frame['right_video_frame_idx'],
                  'sensor_time_ns': ns, 'reference_time_ns': ns,
                  'processing_completed_time_ns': depth_row['processing_completed_time_ns'],
                  'depth_unit': 'mm', 'confidence_encoding': 'mono8'})
        yield ns, [(topics['left'], left), (topics['right'], right),
                   (topics['left_info'], left_info), (topics['right_info'], right_info),
                   (topics['depth'], depth_msg), (topics['confidence'], confidence),
                   (topics['pose'], pose_msg), (topics['tf'], tf), (topics['metadata'], metadata)]


def run(rectify, depth, output, *, mode='bag', prefix='/preprocess', image_format='jpeg',
        jpeg_quality=90, publish_rate=1., storage_id='sqlite3', force=False, resume=False):
    rectify, depth = Path(rectify).resolve(), Path(depth).resolve()
    if mode not in ('bag', 'topic', 'both'):
        raise ValueError('mode must be bag, topic or both')
    if not prefix.startswith('/') or prefix.endswith('/') or '//' in prefix:
        raise ValueError('topic prefix must be an absolute ROS namespace without a trailing slash')
    if not 1 <= jpeg_quality <= 100 or publish_rate <= 0:
        raise ValueError('jpeg_quality must be [1,100] and publish_rate positive')
    config = {'rectify': str(rectify), 'depth': str(depth), 'mode': mode, 'topic_prefix': prefix,
              'image_format': image_format, 'jpeg_quality': jpeg_quality,
              'publish_rate': publish_rate, 'storage_id': storage_id,
              'boundary_units': {'pose': 'metres', 'CameraInfo_P': 'metres', 'depth': 'millimetres'}}

    def action(out):
        import rclpy
        import rosbag2_py
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from rclpy.serialization import serialize_message
        from sensor_msgs.msg import CameraInfo, CompressedImage, Image
        from geometry_msgs.msg import PoseWithCovarianceStamped
        from std_msgs.msg import String
        from tf2_msgs.msg import TFMessage

        type_by_topic = {}
        bag = None
        if mode in ('bag', 'both'):
            uri = out/'rosbag'
            bag = rosbag2_py.SequentialWriter()
            bag.open(rosbag2_py.StorageOptions(uri=str(uri), storage_id=storage_id),
                     rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'))
            # Topic type names are taken from the concrete ROS messages below.
        node = None
        publishers = {}
        spin = None
        if mode in ('topic', 'both'):
            rclpy.init(args=None)
            node = Node('ego_preprocess_output')
            spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True); spin.start()
            qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
            for _, events in _messages(rectify, depth, prefix=prefix, image_format=image_format, jpeg_quality=jpeg_quality):
                for topic, msg in events:
                    if topic not in publishers:
                        publishers[topic] = node.create_publisher(type(msg), topic, qos)
                break
            time.sleep(.2)
        count, first_ns, last_ns = 0, None, None
        wall_anchor = None
        try:
            for ns, events in _messages(rectify, depth, prefix=prefix, image_format=image_format, jpeg_quality=jpeg_quality):
                if last_ns is not None and ns <= last_ns:
                    raise ValueError('frame acquisition timestamps must increase strictly')
                if mode in ('topic', 'both'):
                    if wall_anchor is None:
                        wall_anchor = (time.monotonic(), ns)
                    target = wall_anchor[0] + (ns-wall_anchor[1])/1e9/publish_rate
                    delay = target-time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                for topic, msg in events:
                    package = msg.__class__.__module__.split('.', 1)[0]
                    type_name = f'{package}/msg/{msg.__class__.__name__}'
                    if bag is not None:
                        if topic not in type_by_topic:
                            bag.create_topic(rosbag2_py.TopicMetadata(id=len(type_by_topic) + 1,
                                                                      name=topic, type=type_name,
                                                                      serialization_format='cdr'))
                            type_by_topic[topic] = type_name
                        bag.write(topic, serialize_message(msg), ns)
                    if node is not None:
                        publishers[topic].publish(msg)
                first_ns = ns if first_ns is None else first_ns
                last_ns, count = ns, count+1
        finally:
            if node is not None:
                node.destroy_node()
                if rclpy.ok(): rclpy.shutdown()
                spin.join(timeout=2.)
            if bag is not None:
                bag.close()
        report = {'mode': mode, 'frames': count, 'first_sensor_time_ns': first_ns,
                  'last_sensor_time_ns': last_ns, 'topics': {k: v for k, v in type_by_topic.items()},
                  'rosbag_uri': 'rosbag' if bag is not None else None,
                  'rosbag_storage_id': storage_id if bag is not None else None,
                  'publication_note': 'ROS messages are separate deliveries with equal acquisition stamps, not an atomic transaction'}
        write_json(out/'ros_output_report.json', report)
        return report

    return execute_stage('ros_output', config, output, [rectify, depth], action,
                         resume=resume, force=force)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Write ROS 2 bag and/or publish synchronized preprocessing outputs.')
    parser.add_argument('--rectify', required=True, type=Path); parser.add_argument('--depth', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path); parser.add_argument('--mode', choices=('bag','topic','both'), default='bag')
    parser.add_argument('--prefix', default='/preprocess'); parser.add_argument('--image-format', choices=('jpeg','png'), default='jpeg')
    parser.add_argument('--jpeg-quality', type=int, default=90); parser.add_argument('--publish-rate', type=float, default=1.)
    parser.add_argument('--storage-id', choices=('sqlite3','mcap'), default='sqlite3')
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--force', action='store_true')
    a = parser.parse_args(argv)
    run(a.rectify, a.depth, a.output, mode=a.mode, prefix=a.prefix, image_format=a.image_format,
        jpeg_quality=a.jpeg_quality, publish_rate=a.publish_rate, storage_id=a.storage_id,
        resume=a.resume, force=a.force)


if __name__ == '__main__':
    main()
