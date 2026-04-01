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
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from std_srvs.srv import Trigger
from message_filters import Subscriber, ApproximateTimeSynchronizer

try:
    from pupil_apriltags import Detector
except ImportError:  # pragma: no cover - runtime dependency
    Detector = None


class HandEyeCollector(Node):
    def __init__(self) -> None:
        super().__init__('handeye_collector')

        self.declare_parameter('image_topic', '/zed/zed_node/left/color/rect/image')
        self.declare_parameter('camera_info_topic', '/zed/zed_node/left/color/rect/camera_info')
        self.declare_parameter('robot_pose_topic', '/right/manip/measured/tool_int_pose')
        self.declare_parameter('tag_size_m', 0.08)
        self.declare_parameter('tag_family', 'tag36h11')
        self.declare_parameter('target_tag_id', -1)
        self.declare_parameter('samples_file', 'handeye_samples.json')
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('debug_image_topic', '/onr_handeye/debug_image')
        self.declare_parameter('debug_axis_scale', 0.5)
        self.declare_parameter('online_solve_enabled', True)
        self.declare_parameter('online_solve_min_samples', 8)
        self.declare_parameter('online_solve_every_n', 1)
        self.declare_parameter('online_rot_weight', 0.2)
        self.declare_parameter('online_multistart', 6)
        self.declare_parameter('online_seed', 42)
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
        self.tag_size_m = self.get_parameter('tag_size_m').get_parameter_value().double_value
        self.target_tag_id = self.get_parameter('target_tag_id').get_parameter_value().integer_value
        tag_family = self.get_parameter('tag_family').get_parameter_value().string_value
        self.samples_file = self.get_parameter('samples_file').get_parameter_value().string_value
        self.publish_debug_image = self.get_parameter('publish_debug_image').get_parameter_value().bool_value
        self.debug_image_topic = self.get_parameter('debug_image_topic').get_parameter_value().string_value
        self.debug_axis_scale = self.get_parameter('debug_axis_scale').get_parameter_value().double_value
        self.online_solve_enabled = self.get_parameter('online_solve_enabled').get_parameter_value().bool_value
        self.online_solve_min_samples = int(self.get_parameter('online_solve_min_samples').get_parameter_value().integer_value)
        self.online_solve_every_n = max(1, int(self.get_parameter('online_solve_every_n').get_parameter_value().integer_value))
        self.online_rot_weight = self.get_parameter('online_rot_weight').get_parameter_value().double_value
        self.online_multistart = max(1, int(self.get_parameter('online_multistart').get_parameter_value().integer_value))
        self.online_seed = int(self.get_parameter('online_seed').get_parameter_value().integer_value)
        tool_t_tag_raw = self.get_parameter('tool_t_tag').value
        self.tool_t_tag = pt.check_transform(np.asarray(tool_t_tag_raw, dtype=float).reshape(4, 4))

        if Detector is None:
            raise RuntimeError(
                'Missing dependency pupil_apriltags. Install with: pip install pupil-apriltags'
            )

        self.detector = Detector(families=tag_family, nthreads=2, quad_decimate=1.0)

        self.camera_params: Optional[tuple[float, float, float, float]] = None
        self.latest_cam_t_tag: Optional[np.ndarray] = None
        self.latest_base_t_tool: Optional[np.ndarray] = None
        self.latest_tag_id: Optional[int] = None
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
        self.get_logger().info('call /capture_sample when robot is steady at each pose')
        if self.publish_debug_image:
            self.get_logger().info(f'publishing debug image: {self.debug_image_topic}')
            self.get_logger().info(f'debug axis scale: {self.debug_axis_scale:.2f} (relative to tag_size_m)')
        if self.online_solve_enabled:
            self.get_logger().info(
                f'online solve enabled: min_samples={self.online_solve_min_samples}, '
                f'every_n={self.online_solve_every_n}, multistart={self.online_multistart}'
            )

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if self.camera_params is not None:
            return
        fx = msg.k[0]
        fy = msg.k[4]
        cx = msg.k[2]
        cy = msg.k[5]
        self.camera_params = (float(fx), float(fy), float(cx), float(cy))
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

        if self.camera_params is None:
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

        detections = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=float(self.tag_size_m),
        )
        if not detections:
            if debug_img is not None:
                self._draw_status(debug_img, 'AprilTag NOT detected', (0, 0, 255))
                self._publish_debug_image(debug_img, img_msg)
            return

        selected = None
        if self.target_tag_id >= 0:
            for det in detections:
                if int(det.tag_id) == self.target_tag_id:
                    selected = det
                    break
        else:
            selected = detections[0]

        if selected is None:
            if debug_img is not None:
                self._draw_status(debug_img, f'tag id {self.target_tag_id} not found', (0, 0, 255))
                self._publish_debug_image(debug_img, img_msg)
            return

        cam_t_tag = pt.transform_from(
            np.asarray(selected.pose_R, dtype=float),
            np.asarray(selected.pose_t, dtype=float).reshape(3),
        )

        self.latest_cam_t_tag = cam_t_tag
        self.latest_tag_id = int(selected.tag_id)

        if debug_img is not None:
            self._draw_detection(debug_img, selected)
            self._draw_tag_axes(debug_img, selected)
            self._draw_status(debug_img, f'detected tag id={int(selected.tag_id)}', (0, 255, 0))
            self._publish_debug_image(debug_img, img_msg)

    def _capture_sample_cb(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if self.latest_base_t_tool is None:
            response.success = False
            response.message = 'No robot pose received yet.'
            return response

        if self.latest_cam_t_tag is None:
            response.success = False
            response.message = 'No AprilTag pose detected yet.'
            return response

        sample = {
            'index': self.sample_count,
            'tag_id': int(self.latest_tag_id if self.latest_tag_id is not None else -1),
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
                'tag_size_m': float(self.tag_size_m),
                'target_tag_id': int(self.target_tag_id),
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

    def _draw_detection(self, image_bgr: np.ndarray, det: object) -> None:
        corners = np.asarray(det.corners, dtype=float).reshape(-1, 2).astype(int)
        if corners.shape[0] >= 4:
            for i in range(4):
                p0 = tuple(corners[i])
                p1 = tuple(corners[(i + 1) % 4])
                cv2.line(image_bgr, p0, p1, (0, 255, 0), 2)
        center = np.asarray(det.center, dtype=float).reshape(2).astype(int)
        cv2.circle(image_bgr, tuple(center), 4, (0, 255, 255), -1)
        cv2.putText(
            image_bgr,
            f'id={int(det.tag_id)}',
            (int(center[0]) + 8, int(center[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    def _draw_tag_axes(self, image_bgr: np.ndarray, det: object) -> None:
        if self.camera_params is None:
            return

        axis_len = max(float(self.tag_size_m) * float(self.debug_axis_scale), 1e-4)
        obj_pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [axis_len, 0.0, 0.0],
                [0.0, axis_len, 0.0],
                [0.0, 0.0, axis_len],
            ],
            dtype=np.float32,
        )

        fx, fy, cx, cy = self.camera_params
        cam_k = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        rvec, _ = cv2.Rodrigues(np.asarray(det.pose_R, dtype=np.float64))
        tvec = np.asarray(det.pose_t, dtype=np.float64).reshape(3, 1)
        img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, cam_k, np.zeros((4, 1), dtype=np.float64))
        img_pts = img_pts.reshape(-1, 2).astype(int)

        o = tuple(img_pts[0])
        px = tuple(img_pts[1])
        py = tuple(img_pts[2])
        pz = tuple(img_pts[3])

        cv2.line(image_bgr, o, px, (0, 0, 255), 2)
        cv2.line(image_bgr, o, py, (0, 255, 0), 2)
        cv2.line(image_bgr, o, pz, (255, 0, 0), 2)
        cv2.putText(image_bgr, 'X', (px[0] + 4, px[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(image_bgr, 'Y', (py[0] + 4, py[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(image_bgr, 'Z', (pz[0] + 4, pz[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

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

    def _se3_to_params(self, t_base_cam: np.ndarray) -> np.ndarray:
        t = np.asarray(t_base_cam[:3, 3], dtype=float)
        rotvec = R.from_matrix(t_base_cam[:3, :3]).as_rotvec()
        return np.array([t[0], t[1], t[2], rotvec[0], rotvec[1], rotvec[2]], dtype=float)

    def _params_to_se3(self, x: np.ndarray) -> np.ndarray:
        rot = R.from_rotvec(np.asarray(x[3:6], dtype=float)).as_matrix()
        return pt.transform_from(rot, np.asarray(x[:3], dtype=float))

    def _estimate_base_t_cam_from_one(self, sample: dict) -> np.ndarray:
        base_t_tool = self._se3_from_list(sample['base_T_tool'])
        cam_t_tag = self._se3_from_list(sample['cam_T_tag'])
        return pt.concat(pt.concat(base_t_tool, self.tool_t_tag), pt.invert_transform(cam_t_tag))

    def _residual_vector(self, x: np.ndarray) -> np.ndarray:
        base_t_cam = self._params_to_se3(x)
        cam_t_base = pt.invert_transform(base_t_cam)
        residuals: list[float] = []

        for s in self.samples:
            base_t_tool = self._se3_from_list(s['base_T_tool'])
            cam_t_tag_meas = self._se3_from_list(s['cam_T_tag'])
            cam_t_tag_pred = pt.concat(pt.concat(cam_t_base, base_t_tool), self.tool_t_tag)

            t_err = np.asarray(cam_t_tag_pred[:3, 3], dtype=float) - np.asarray(cam_t_tag_meas[:3, 3], dtype=float)
            rot_delta = pt.concat(pt.invert_transform(cam_t_tag_meas), cam_t_tag_pred)
            rot_err = R.from_matrix(rot_delta[:3, :3]).as_rotvec()
            residuals.extend(t_err.tolist())
            residuals.extend((float(self.online_rot_weight) * rot_err).tolist())

        return np.array(residuals, dtype=float)

    def _evaluate_errors(self, base_t_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        trans_err = []
        rot_err_deg = []
        cam_t_base = pt.invert_transform(base_t_cam)
        for s in self.samples:
            base_t_tool = self._se3_from_list(s['base_T_tool'])
            cam_t_tag_meas = self._se3_from_list(s['cam_T_tag'])
            cam_t_tag_pred = pt.concat(pt.concat(cam_t_base, base_t_tool), self.tool_t_tag)

            te = np.linalg.norm(np.asarray(cam_t_tag_pred[:3, 3], dtype=float) - np.asarray(cam_t_tag_meas[:3, 3], dtype=float))
            rot_delta = pt.concat(pt.invert_transform(cam_t_tag_meas), cam_t_tag_pred)
            re = R.from_matrix(rot_delta[:3, :3]).as_rotvec()
            trans_err.append(float(te))
            rot_err_deg.append(float(np.linalg.norm(np.rad2deg(re))))

        return np.array(trans_err, dtype=float), np.array(rot_err_deg, dtype=float)

    def _build_start_points(self) -> list[np.ndarray]:
        starts: list[np.ndarray] = []
        if self.online_base_t_cam is not None:
            starts.append(self._se3_to_params(self.online_base_t_cam))

        seed = self._estimate_base_t_cam_from_one(self.samples[0])
        starts.append(self._se3_to_params(seed))

        for s in self.samples[1:]:
            if len(starts) >= self.online_multistart:
                break
            starts.append(self._se3_to_params(self._estimate_base_t_cam_from_one(s)))

        rng = np.random.default_rng(self.online_seed + len(self.samples))
        while len(starts) < self.online_multistart:
            x = starts[0].copy()
            x[:3] += rng.normal(0.0, 0.03, size=3)
            x[3:6] += rng.normal(0.0, 0.15, size=3)
            starts.append(x)

        return starts

    def _run_online_solve_if_needed(self) -> None:
        if not self.online_solve_enabled:
            return

        n = len(self.samples)
        if n < self.online_solve_min_samples:
            self.online_metrics_text = f'online n={n}/{self.online_solve_min_samples} waiting...'
            return

        if ((n - self.online_solve_min_samples) % self.online_solve_every_n) != 0:
            return

        best = None
        for x0 in self._build_start_points():
            opt = least_squares(
                self._residual_vector,
                x0,
                method='trf',
                loss='huber',
                f_scale=0.01,
                max_nfev=300,
            )
            if not opt.success:
                continue
            if best is None or opt.cost < best.cost:
                best = opt

        if best is None:
            self.get_logger().warn('online solve failed for all start points.')
            self.online_metrics_text = f'online n={n} solve failed'
            return

        self.online_base_t_cam = self._params_to_se3(best.x)
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
            f'[online solve] n={n}, cost={float(best.cost):.6f}, '
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
