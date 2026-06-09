import json
import os
from typing import Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
import pytransform3d.transformations as pt
from scipy.spatial.transform import Rotation as R
from std_srvs.srv import Trigger
from message_filters import Subscriber, ApproximateTimeSynchronizer


class HandEyeCollector(Node):
    def __init__(self) -> None:
        super().__init__('handeye_collector')

        self.declare_parameter('image_topic', '/zed/zed_node/left/color/rect/image')
        self.declare_parameter('camera_info_topic', '/zed/zed_node/left/color/rect/camera_info')
        self.declare_parameter('robot_pose_topic', '/right/manip/measured/tool_int_pose')
        self.declare_parameter('aruco_dictionary', 'DICT_6X6_250')
        self.declare_parameter('charuco_squares_x', 8)
        self.declare_parameter('charuco_squares_y', 12)
        self.declare_parameter('charuco_square_length_m', 0.01)
        self.declare_parameter('charuco_marker_length_m', 0.007)
        self.declare_parameter('charuco_legacy_pattern', True)
        self.declare_parameter('min_charuco_corners', 4)
        self.declare_parameter('samples_file', 'handeye_samples.json')
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('debug_image_topic', '/onr_handeye/debug_image')
        self.declare_parameter('debug_axis_scale', 0.5)
        self.declare_parameter('online_solve_enabled', True)
        self.declare_parameter('online_solve_min_samples', 8)
        self.declare_parameter('online_solve_every_n', 1)
        self.declare_parameter('handeye_method', 'PARK')
        self.declare_parameter(
            'tool_t_tag',
            [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        )


        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        robot_pose_topic = self.get_parameter('robot_pose_topic').get_parameter_value().string_value
        dictionary_name = self.get_parameter('aruco_dictionary').get_parameter_value().string_value
        self.charuco_squares_x = int(self.get_parameter('charuco_squares_x').get_parameter_value().integer_value)
        self.charuco_squares_y = int(self.get_parameter('charuco_squares_y').get_parameter_value().integer_value)
        self.charuco_square_length_m = self.get_parameter('charuco_square_length_m').get_parameter_value().double_value
        self.charuco_marker_length_m = self.get_parameter('charuco_marker_length_m').get_parameter_value().double_value
        self.charuco_legacy_pattern = self.get_parameter('charuco_legacy_pattern').get_parameter_value().bool_value
        self.min_charuco_corners = int(self.get_parameter('min_charuco_corners').get_parameter_value().integer_value)
        self.samples_file = self.get_parameter('samples_file').get_parameter_value().string_value
        self.publish_debug_image = self.get_parameter('publish_debug_image').get_parameter_value().bool_value
        self.debug_image_topic = self.get_parameter('debug_image_topic').get_parameter_value().string_value
        self.debug_axis_scale = self.get_parameter('debug_axis_scale').get_parameter_value().double_value
        self.online_solve_enabled = self.get_parameter('online_solve_enabled').get_parameter_value().bool_value
        self.online_solve_min_samples = int(self.get_parameter('online_solve_min_samples').get_parameter_value().integer_value)
        self.online_solve_every_n = max(1, int(self.get_parameter('online_solve_every_n').get_parameter_value().integer_value))
        self.handeye_method = self.get_parameter('handeye_method').get_parameter_value().string_value.upper()
        tool_t_tag_raw = self.get_parameter('tool_t_tag').value
        self.tool_t_tag = pt.check_transform(np.asarray(tool_t_tag_raw, dtype=float).reshape(4, 4))

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self._aruco_dictionary_id(dictionary_name))
        self.charuco_board = cv2.aruco.CharucoBoard(
            (self.charuco_squares_x, self.charuco_squares_y),
            float(self.charuco_square_length_m),
            float(self.charuco_marker_length_m),
            self.aruco_dict,
        )
        if hasattr(self.charuco_board, 'setLegacyPattern'):
            self.charuco_board.setLegacyPattern(bool(self.charuco_legacy_pattern))
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.latest_cam_t_tag: Optional[np.ndarray] = None
        self.latest_base_t_tool: Optional[np.ndarray] = None
        self.latest_marker_count = 0
        self.latest_charuco_count = 0
        self.latest_image_frame: str = ''
        self.sample_count = 0
        self.samples: list[dict] = []
        self.online_base_t_cam: Optional[np.ndarray] = None
        self.online_metrics_text: Optional[str] = None

        # Subscribe to camera_info independently (low frequency)
        self.create_subscription(CameraInfo, camera_info_topic, self._camera_info_cb, 10)
        
        # Use message_filters for time-synchronized image + pose subscription
        self.img_sub = Subscriber(self, Image, image_topic)
        self.pose_sub = Subscriber(self, PoseStamped, robot_pose_topic)
        self.sync = ApproximateTimeSynchronizer(
            [self.img_sub, self.pose_sub],
            queue_size=10,
            slop=0.1  # 100ms tolerance for time sync
        )
        self.sync.registerCallback(self._image_pose_cb)

        self.debug_image_pub = None
        if self.publish_debug_image:
            self.debug_image_pub = self.create_publisher(Image, self.debug_image_topic, 10)

        self.create_service(Trigger, 'capture_sample', self._capture_sample_cb)
        self.create_service(Trigger, 'save_samples', self._save_samples_cb)

        self.get_logger().info(f'listening image: {image_topic}')
        self.get_logger().info(f'listening camera_info: {camera_info_topic}')
        self.get_logger().info(f'listening robot_pose: {robot_pose_topic}')
        self.get_logger().info(
            f'ChArUco board: dict={dictionary_name}, squares='
            f'{self.charuco_squares_x}x{self.charuco_squares_y}, '
            f'square={self.charuco_square_length_m:.4f}m, '
            f'marker={self.charuco_marker_length_m:.4f}m, '
            f'legacy={self.charuco_legacy_pattern}'
        )
        self.get_logger().info('call /capture_sample when robot is steady at each pose')
        if self.publish_debug_image:
            self.get_logger().info(f'publishing debug image: {self.debug_image_topic}')
            self.get_logger().info(f'debug axis scale: {self.debug_axis_scale:.2f} (relative to square length)')
        if self.online_solve_enabled:
            self.get_logger().info(
                f'online solve enabled: min_samples={self.online_solve_min_samples}, '
                f'every_n={self.online_solve_every_n}, method={self.handeye_method}'
            )

    @staticmethod
    def _aruco_dictionary_id(name: str) -> int:
        dictionary_id = getattr(cv2.aruco, str(name), None)
        if dictionary_id is None:
            valid = sorted(k for k in dir(cv2.aruco) if k.startswith('DICT_'))
            raise ValueError(f'Unknown ArUco dictionary "{name}". Valid examples: {valid[:8]} ...')
        return int(dictionary_id)

    @staticmethod
    def _handeye_method_id(name: str) -> int:
        mapping = {
            'TSAI': cv2.CALIB_HAND_EYE_TSAI,
            'PARK': cv2.CALIB_HAND_EYE_PARK,
            'HORAUD': cv2.CALIB_HAND_EYE_HORAUD,
            'ANDREFF': cv2.CALIB_HAND_EYE_ANDREFF,
            'DANIILIDIS': cv2.CALIB_HAND_EYE_DANIILIDIS,
        }
        method = str(name).upper()
        if method not in mapping:
            raise ValueError(f'Unknown handeye_method "{name}". Valid: {sorted(mapping)}')
        return mapping[method]

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if self.camera_matrix is not None:
            return
        fx = msg.k[0]
        fy = msg.k[4]
        cx = msg.k[2]
        cy = msg.k[5]
        self.camera_matrix = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        if len(msg.d) > 0:
            self.dist_coeffs = np.asarray(msg.d, dtype=np.float64).reshape(-1, 1)
        else:
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        self.get_logger().info(f'camera intrinsics ready: fx={fx:.2f}, fy={fy:.2f}')

    def _image_pose_cb(self, img_msg: Image, pose_msg: PoseStamped) -> None:
        """Handle synchronized image and pose together"""
        self.latest_image_frame = img_msg.header.frame_id
        
        # Update pose from synchronized message
        self.latest_base_t_tool = pose_to_matrix(
            pose_msg.pose.position.x,
            pose_msg.pose.position.y,
            pose_msg.pose.position.z,
            pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z,
            pose_msg.pose.orientation.w
        )

        if self.camera_matrix is None or self.dist_coeffs is None:
            if self.publish_debug_image:
                debug_img = self._image_to_bgr(img_msg)
                if debug_img is not None:
                    self._draw_status(debug_img, 'waiting camera_info', (0, 140, 255))
                    self._publish_debug_image(debug_img, img_msg)
            return

        gray = self._image_to_gray(img_msg)
        debug_img = self._image_to_bgr(img_msg) if self.publish_debug_image else None

        if gray is None:
            if debug_img is not None:
                self._draw_status(debug_img, f'unsupported image encoding: {img_msg.encoding}', (0, 0, 255))
                self._publish_debug_image(debug_img, img_msg)
            return

        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.aruco_params,
        )
        self.latest_cam_t_tag = None
        self.latest_marker_count = 0 if marker_ids is None else len(marker_ids)
        self.latest_charuco_count = 0

        if marker_ids is None or len(marker_ids) == 0:
            if debug_img is not None:
                self._draw_status(debug_img, 'ChArUco markers NOT detected', (0, 0, 255))
                self._publish_debug_image(debug_img, img_msg)
            return

        count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            self.charuco_board,
            self.camera_matrix,
            self.dist_coeffs,
        )
        self.latest_charuco_count = int(count)

        if debug_img is not None:
            cv2.aruco.drawDetectedMarkers(debug_img, marker_corners, marker_ids)

        if charuco_ids is None or count < self.min_charuco_corners:
            if debug_img is not None:
                self._draw_status(
                    debug_img,
                    f'ArUco={self.latest_marker_count}, ChArUco={self.latest_charuco_count} need>={self.min_charuco_corners}',
                    (0, 0, 255),
                )
                self._publish_debug_image(debug_img, img_msg)
            return

        if debug_img is not None:
            cv2.aruco.drawDetectedCornersCharuco(debug_img, charuco_corners, charuco_ids)

        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners,
            charuco_ids,
            self.charuco_board,
            self.camera_matrix,
            self.dist_coeffs,
            None,
            None,
        )
        if not ok:
            if debug_img is not None:
                self._draw_status(
                    debug_img,
                    f'ChArUco pose failed: markers={self.latest_marker_count}, corners={self.latest_charuco_count}',
                    (0, 0, 255),
                )
                self._publish_debug_image(debug_img, img_msg)
            return

        rot, _ = cv2.Rodrigues(rvec)
        self.latest_cam_t_tag = pt.transform_from(
            np.asarray(rot, dtype=float),
            np.asarray(tvec, dtype=float).reshape(3),
        )

        if debug_img is not None:
            cv2.drawFrameAxes(
                debug_img,
                self.camera_matrix,
                self.dist_coeffs,
                rvec,
                tvec,
                length=max(float(self.charuco_square_length_m) * float(self.debug_axis_scale), 1e-4),
            )
            self._draw_status(debug_img, f'ChArUco pose OK: markers={self.latest_marker_count}, corners={self.latest_charuco_count}', (0, 255, 0))
            self._publish_debug_image(debug_img, img_msg)

    def _capture_sample_cb(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if self.latest_base_t_tool is None:
            response.success = False
            response.message = 'No robot pose received yet.'
            return response

        if self.latest_cam_t_tag is None:
            response.success = False
            response.message = (
                'No ChArUco board pose detected yet '
                f'(markers={self.latest_marker_count}, corners={self.latest_charuco_count}).'
            )
            return response

        sample = {
            'index': self.sample_count,
            'target_type': 'charuco_board',
            'marker_count': int(self.latest_marker_count),
            'charuco_corner_count': int(self.latest_charuco_count),
            'base_T_tool': self.latest_base_t_tool.reshape(-1).tolist(),
            'cam_T_tag': self.latest_cam_t_tag.reshape(-1).tolist(),
        }
        self.samples.append(sample)
        self.sample_count += 1

        response.success = True
        response.message = f'Captured sample #{sample["index"]} (total={len(self.samples)}).'
        self.get_logger().info(response.message)
        self._run_online_solve_if_needed()
        return response

    def _save_samples_cb(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if not self.samples:
            response.success = False
            response.message = 'No samples to save.'
            return response

        out_path = os.path.expanduser(self.samples_file)
        payload = {
            'meta': {
                'target_type': 'charuco_board',
                'aruco_dictionary': self.get_parameter('aruco_dictionary').get_parameter_value().string_value,
                'charuco_squares_x': int(self.charuco_squares_x),
                'charuco_squares_y': int(self.charuco_squares_y),
                'charuco_square_length_m': float(self.charuco_square_length_m),
                'charuco_marker_length_m': float(self.charuco_marker_length_m),
                'charuco_legacy_pattern': bool(self.charuco_legacy_pattern),
                'num_samples': len(self.samples),
            },
            'samples': self.samples,
        }

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

        response.success = True
        response.message = f'Saved {len(self.samples)} samples to {out_path}'
        self.get_logger().info(response.message)
        return response

    def _image_to_gray(self, msg: Image) -> Optional[np.ndarray]:
        enc = msg.encoding.lower()
        h = msg.height
        w = msg.width

        data = np.frombuffer(msg.data, dtype=np.uint8)

        if enc == 'mono8':
            if data.size != h * w:
                return None
            return data.reshape((h, w))

        if enc in ('rgb8', 'bgr8'):
            if data.size != h * w * 3:
                return None
            img = data.reshape((h, w, 3))
            code = cv2.COLOR_RGB2GRAY if enc == 'rgb8' else cv2.COLOR_BGR2GRAY
            return cv2.cvtColor(img, code)

        if enc in ('rgba8', 'bgra8'):
            if data.size != h * w * 4:
                return None
            img = data.reshape((h, w, 4))
            code = cv2.COLOR_RGBA2GRAY if enc == 'rgba8' else cv2.COLOR_BGRA2GRAY
            return cv2.cvtColor(img, code)

        self.get_logger().warn(f'Unsupported image encoding: {msg.encoding}')
        return None

    def _image_to_bgr(self, msg: Image) -> Optional[np.ndarray]:
        enc = msg.encoding.lower()
        h = msg.height
        w = msg.width
        data = np.frombuffer(msg.data, dtype=np.uint8)

        if enc == 'mono8':
            if data.size != h * w:
                return None
            gray = data.reshape((h, w))
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if enc == 'bgr8':
            if data.size != h * w * 3:
                return None
            return data.reshape((h, w, 3)).copy()

        if enc == 'rgb8':
            if data.size != h * w * 3:
                return None
            rgb = data.reshape((h, w, 3))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if enc == 'bgra8':
            if data.size != h * w * 4:
                return None
            bgra = data.reshape((h, w, 4))
            return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

        if enc == 'rgba8':
            if data.size != h * w * 4:
                return None
            rgba = data.reshape((h, w, 4))
            return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

        return None

    def _draw_status(self, image_bgr: np.ndarray, text: str, color_bgr: tuple[int, int, int]) -> None:
        cv2.rectangle(image_bgr, (8, 8), (520, 40), (0, 0, 0), -1)
        cv2.putText(
            image_bgr,
            text,
            (14, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color_bgr,
            2,
            cv2.LINE_AA,
        )

        if self.online_metrics_text is not None:
            cv2.rectangle(image_bgr, (8, 42), (740, 74), (0, 0, 0), -1)
            cv2.putText(
                image_bgr,
                self.online_metrics_text,
                (14, 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )

    def _publish_debug_image(self, image_bgr: np.ndarray, src_msg: Image) -> None:
        if self.debug_image_pub is None:
            return
        out = Image()
        out.header = src_msg.header
        out.height = int(image_bgr.shape[0])
        out.width = int(image_bgr.shape[1])
        out.encoding = 'bgr8'
        out.is_bigendian = 0
        out.step = int(image_bgr.shape[1] * 3)
        out.data = image_bgr.tobytes()
        self.debug_image_pub.publish(out)

    def _se3_from_list(self, values: list[float]) -> np.ndarray:
        return np.asarray(values, dtype=float).reshape(4, 4)

    def _evaluate_errors(self, base_t_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        trans_err = []
        rot_err_deg = []
        cam_t_base = pt.invert_transform(base_t_cam)
        for s in self.samples:
            base_t_tool = self._se3_from_list(s['base_T_tool'])
            cam_t_tag_meas = self._se3_from_list(s['cam_T_tag'])
            cam_t_tag_pred = cam_t_base @ base_t_tool @ self.tool_t_tag

            te = np.linalg.norm(np.asarray(cam_t_tag_pred[:3, 3], dtype=float) - np.asarray(cam_t_tag_meas[:3, 3], dtype=float))
            rot_delta = pt.invert_transform(cam_t_tag_meas) @ cam_t_tag_pred
            re = R.from_matrix(rot_delta[:3, :3]).as_rotvec()
            trans_err.append(float(te))
            rot_err_deg.append(float(np.linalg.norm(np.rad2deg(re))))

        return np.array(trans_err, dtype=float), np.array(rot_err_deg, dtype=float)

    def _solve_base_t_cam_handeye(self) -> np.ndarray:
        r_gripper2base = []
        t_gripper2base = []
        r_target2cam = []
        t_target2cam = []

        for s in self.samples:
            base_t_tool = self._se3_from_list(s['base_T_tool'])
            cam_t_tag = self._se3_from_list(s['cam_T_tag'])
            base_t_tag = base_t_tool @ self.tool_t_tag
            tag_t_base = pt.invert_transform(base_t_tag)

            r_gripper2base.append(np.asarray(tag_t_base[:3, :3], dtype=np.float64))
            t_gripper2base.append(np.asarray(tag_t_base[:3, 3], dtype=np.float64).reshape(3, 1))
            r_target2cam.append(np.asarray(cam_t_tag[:3, :3], dtype=np.float64))
            t_target2cam.append(np.asarray(cam_t_tag[:3, 3], dtype=np.float64).reshape(3, 1))

        rot, trans = cv2.calibrateHandEye(
            r_gripper2base,
            t_gripper2base,
            r_target2cam,
            t_target2cam,
            method=self._handeye_method_id(self.handeye_method),
        )
        return pt.transform_from(
            np.asarray(rot, dtype=float),
            np.asarray(trans, dtype=float).reshape(3),
        )

    def _run_online_solve_if_needed(self) -> None:
        if not self.online_solve_enabled:
            return

        n = len(self.samples)
        if n < self.online_solve_min_samples:
            self.online_metrics_text = f'online n={n}/{self.online_solve_min_samples} waiting...'
            return

        if ((n - self.online_solve_min_samples) % self.online_solve_every_n) != 0:
            return

        try:
            self.online_base_t_cam = self._solve_base_t_cam_handeye()
        except cv2.error as exc:
            self.get_logger().warn(f'online cv2.calibrateHandEye failed: {exc}')
            self.online_metrics_text = f'online n={n} solve failed'
            return

        trans_err, rot_err_deg = self._evaluate_errors(self.online_base_t_cam)
        t_mean = float(np.mean(trans_err))
        t_std = float(np.std(trans_err))
        r_mean = float(np.mean(rot_err_deg))
        r_std = float(np.std(rot_err_deg))

        self.online_metrics_text = (
            f'online n={n} t={t_mean:.3f}+-{t_std:.3f}m '
            f'r={r_mean:.1f}+-{r_std:.1f}deg'
        )
        self.get_logger().info(
            f'[online cv2.calibrateHandEye:{self.handeye_method}] n={n}, '
            f'trans={t_mean:.4f}+-{t_std:.4f} m, rot={r_mean:.2f}+-{r_std:.2f} deg'
        )


def pose_to_matrix(px: float, py: float, pz: float,
                   qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    rot = R.from_quat([float(qx), float(qy), float(qz), float(qw)]).as_matrix()
    return pt.transform_from(rot, [float(px), float(py), float(pz)])


def main() -> None:
    rclpy.init()
    node = HandEyeCollector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
