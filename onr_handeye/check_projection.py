import argparse
import json
from typing import List

import numpy as np
import pytransform3d.transformations as pt
from scipy.spatial.transform import Rotation as R


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Back-project solved base_T_cam to camera frame and compare predicted vs measured cam_T_target.'
    )
    parser.add_argument(
        '--samples',
        default='',
        help='Path to handeye_samples.json. If omitted, infer from result JSON or default to handeye_samples.json.',
    )
    parser.add_argument(
        '--result',
        default='base_T_cam_result.json',
        help='Path to base_T_cam_result.json. Default: base_T_cam_result.json',
    )
    parser.add_argument(
        '--tool-t-tag',
        nargs=16,
        type=float,
        default=[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
        metavar=('m00', 'm01', 'm02', 'm03', 'm10', 'm11', 'm12', 'm13',
                 'm20', 'm21', 'm22', 'm23', 'm30', 'm31', 'm32', 'm33'),
        help='4x4 tool_T_tag/tool_T_target row-major matrix. Default is identity.',
    )
    parser.add_argument(
        '--tool-tag-offset-mm',
        type=float,
        default=0.0,
        help='Extra fixed translation from tool origin to calibration target origin in millimeters.',
    )
    parser.add_argument(
        '--tool-tag-axis',
        choices=['x', 'y', 'z', '-x', '-y', '-z'],
        default='z',
        help='Axis of the extra tool->target translation offset.',
    )
    parser.add_argument('--top-k', type=int, default=5, help='Print top-k worst samples by combined score.')
    parser.add_argument('--print-cam-t-tool', action='store_true', help='Print cam_T_tool matrix per sample.')
    parser.add_argument(
        '--publish-rviz-current-robot-pose',
        action='store_true',
        help='Start ROS2 subscriber and publish current robot pose in camera frame to TF for RViz.',
    )
    parser.add_argument(
        '--robot-pose-topic',
        default='/right/manip/measured/tool_int_pose',
        help='Robot pose topic carrying base_T_tool as geometry_msgs/PoseStamped.',
    )
    parser.add_argument(
        '--camera-frame',
        default='zed_left_camera_frame_optical',
        help='Parent frame id for published TF when --publish-rviz-current-robot-pose is enabled.',
    )
    parser.add_argument(
        '--robot-frame',
        default='robot_current',
        help='Child frame id for published TF when --publish-rviz-current-robot-pose is enabled.',
    )
    return parser.parse_args()


def axis_to_unit_vector(axis_name: str) -> np.ndarray:
    mapping = {
        'x': np.array([1.0, 0.0, 0.0]),
        'y': np.array([0.0, 1.0, 0.0]),
        'z': np.array([0.0, 0.0, 1.0]),
        '-x': np.array([-1.0, 0.0, 0.0]),
        '-y': np.array([0.0, -1.0, 0.0]),
        '-z': np.array([0.0, 0.0, -1.0]),
    }
    return mapping[axis_name]


def mat_from_list(values: List[float]) -> np.ndarray:
    mat = np.array(values, dtype=float).reshape(4, 4)
    if not np.allclose(mat[3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
        raise ValueError('Input transform last row must be [0,0,0,1].')
    return mat


def load_samples(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    return payload['samples']


def load_base_t_cam(path: str) -> np.ndarray:
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    if 'base_T_cam' in payload:
        return mat_from_list(payload['base_T_cam'])
    if 'base_T_cam_matrix' in payload:
        return np.array(payload['base_T_cam_matrix'], dtype=float).reshape(4, 4)
    raise KeyError('Result JSON must contain base_T_cam or base_T_cam_matrix.')


def load_result_payload(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _run_rviz_pose_publisher(
    base_t_cam: np.ndarray,
    robot_pose_topic: str,
    camera_frame: str,
    robot_frame: str,
) -> None:
    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped, TransformStamped
        from rclpy.node import Node
        from tf2_ros import TransformBroadcaster
    except ImportError as exc:
        raise RuntimeError(
            'ROS2 runtime dependencies are required for --publish-rviz-current-robot-pose '
            '(rclpy, geometry_msgs, tf2_ros). Run with sourced ROS2 environment.'
        ) from exc

    def _pose_to_matrix(msg: PoseStamped) -> np.ndarray:
        quat = [
            float(msg.pose.orientation.x),
            float(msg.pose.orientation.y),
            float(msg.pose.orientation.z),
            float(msg.pose.orientation.w),
        ]
        rot = R.from_quat(quat).as_matrix()
        trans = [
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
        ]
        return pt.transform_from(rot, trans)

    class _RobotPoseInCameraTfNode(Node):
        def __init__(self) -> None:
            super().__init__('check_projection_rviz_pose_publisher')
            self._cam_t_base = pt.invert_transform(base_t_cam)
            self._camera_frame = str(camera_frame)
            self._robot_frame = str(robot_frame)
            self._tf_broadcaster = TransformBroadcaster(self)
            self.create_subscription(PoseStamped, robot_pose_topic, self._pose_cb, 20)
            self.get_logger().info(
                f'publishing current robot pose in camera frame: {self._camera_frame} -> {self._robot_frame}'
            )
            self.get_logger().info(f'subscribed robot pose topic: {robot_pose_topic}')

        def _pose_cb(self, msg: PoseStamped) -> None:
            base_t_tool = _pose_to_matrix(msg)
            cam_t_tool = self._cam_t_base @ base_t_tool

            tf_msg = TransformStamped()
            tf_msg.header.stamp = msg.header.stamp
            tf_msg.header.frame_id = self._camera_frame
            tf_msg.child_frame_id = self._robot_frame
            tf_msg.transform.translation.x = float(cam_t_tool[0, 3])
            tf_msg.transform.translation.y = float(cam_t_tool[1, 3])
            tf_msg.transform.translation.z = float(cam_t_tool[2, 3])
            qx, qy, qz, qw = R.from_matrix(cam_t_tool[:3, :3]).as_quat()
            tf_msg.transform.rotation.x = float(qx)
            tf_msg.transform.rotation.y = float(qy)
            tf_msg.transform.rotation.z = float(qz)
            tf_msg.transform.rotation.w = float(qw)
            self._tf_broadcaster.sendTransform(tf_msg)

    rclpy.init()
    node = _RobotPoseInCameraTfNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> None:
    args = parse_args()
    result_payload = load_result_payload(args.result)
    base_t_cam = pt.check_transform(load_base_t_cam(args.result))

    samples_path = str(args.samples).strip()
    if not samples_path:
        samples_path = str(result_payload.get('samples_file', 'handeye_samples.json'))
        print(f'using samples file: {samples_path}')

    samples = load_samples(samples_path)

    use_cli_tool_t_tag = (
        not np.allclose(np.array(args.tool_t_tag, dtype=float).reshape(4, 4), np.eye(4), atol=1e-12)
        or abs(float(args.tool_tag_offset_mm)) > 1e-12
    )

    if ('tool_T_tag_used' in result_payload or 'tool_T_tag_used_matrix' in result_payload) and not use_cli_tool_t_tag:
        if 'tool_T_tag_used' in result_payload:
            tool_t_tag = mat_from_list(result_payload['tool_T_tag_used'])
        else:
            tool_t_tag = np.array(result_payload['tool_T_tag_used_matrix'], dtype=float).reshape(4, 4)
        tool_t_tag = pt.check_transform(tool_t_tag)
        print('using tool_T_tag from solve result file')
    else:
        tool_t_tag = pt.check_transform(np.array(args.tool_t_tag, dtype=float).reshape(4, 4))
        if abs(float(args.tool_tag_offset_mm)) > 1e-12:
            offset_m = float(args.tool_tag_offset_mm) / 1000.0
            axis = axis_to_unit_vector(args.tool_tag_axis)
            t_extra = pt.transform_from(np.eye(3), axis * offset_m)
            tool_t_tag = tool_t_tag @ t_extra
            print(
                f'apply tool->tag offset: {args.tool_tag_offset_mm:.3f} mm '
                f'along {args.tool_tag_axis} axis'
            )

    cam_t_base = pt.invert_transform(base_t_cam)

    rows = []
    for i, s in enumerate(samples):
        base_t_tool = mat_from_list(s['base_T_tool'])
        cam_t_tag_meas = mat_from_list(s['cam_T_tag'])

        cam_t_tool = cam_t_base @ base_t_tool
        cam_t_tag_pred = cam_t_tool @ tool_t_tag

        t_err_vec = cam_t_tag_pred[:3, 3] - cam_t_tag_meas[:3, 3]
        t_err = float(np.linalg.norm(t_err_vec))

        rot_delta = pt.invert_transform(cam_t_tag_meas) @ cam_t_tag_pred
        r_err_deg = float(np.linalg.norm(np.rad2deg(R.from_matrix(rot_delta[:3, :3]).as_rotvec())))

        rows.append((i, t_err, r_err_deg, cam_t_tool, cam_t_tag_pred, cam_t_tag_meas))

    t_arr = np.array([x[1] for x in rows], dtype=float)
    r_arr = np.array([x[2] for x in rows], dtype=float)

    print(f'samples: {len(rows)}')
    print(f'translation error mean (m): {float(np.mean(t_arr)):.6f}')
    print(f'translation error std  (m): {float(np.std(t_arr)):.6f}')
    print(f'rotation error mean (deg): {float(np.mean(r_arr)):.4f}')
    print(f'rotation error std  (deg): {float(np.std(r_arr)):.4f}')

    score = t_arr + np.deg2rad(r_arr) * 0.1
    order = np.argsort(-score)
    top_k = max(1, min(int(args.top_k), len(rows)))

    print('\nWorst samples:')
    for rank in range(top_k):
        idx = int(order[rank])
        i, t_err, r_err_deg, cam_t_tool, cam_t_tag_pred, cam_t_tag_meas = rows[idx]
        print(f'  #{rank + 1} sample[{i}]  t_err={t_err:.4f} m, r_err={r_err_deg:.2f} deg')
        print(f'     pred_tag_t={cam_t_tag_pred[:3, 3].tolist()}')
        print(f'     meas_tag_t={cam_t_tag_meas[:3, 3].tolist()}')
        if args.print_cam_t_tool:
            print('     cam_T_tool:')
            for row in cam_t_tool:
                print(f'       {row.tolist()}')

    if args.publish_rviz_current_robot_pose:
        print(
            '\nstarting rviz TF publisher: '
            f'{args.camera_frame} -> {args.robot_frame}, topic={args.robot_pose_topic}'
        )
        _run_rviz_pose_publisher(
            base_t_cam=base_t_cam,
            robot_pose_topic=args.robot_pose_topic,
            camera_frame=args.camera_frame,
            robot_frame=args.robot_frame,
        )


if __name__ == '__main__':
    main()
