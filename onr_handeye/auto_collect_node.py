from dataclasses import dataclass
from itertools import product
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from std_srvs.srv import Trigger


@dataclass
class TargetPose:
    label: str
    pose: PoseStamped


class AutoCollector(Node):
    def __init__(self) -> None:
        super().__init__('auto_handeye_collector')

        # You can replace command_topic with your real robot command topic.
        self.declare_parameter('robot_pose_topic', '/right/manip/measured/tool_int_pose')
        self.declare_parameter('command_topic', '/righthand/pose')
        self.declare_parameter('start_position', [
            0.4484192132949829,
            0.3618442118167877,
            0.20653684437274933,
        ])
        self.declare_parameter('start_orientation', [
            -0.3472360339746712,
            0.6331430428480898,
            0.5757953789455686,
            0.383427575413574,
        ])
        self.declare_parameter('capture_service', '/capture_sample')
        self.declare_parameter('save_service', '/save_samples')
        self.declare_parameter('auto_save_at_end', True)

        self.declare_parameter('tick_hz', 100.0)
        self.declare_parameter('settle_time_sec', 0.7)
        self.declare_parameter('retry_interval_sec', 0.6)
        self.declare_parameter('max_capture_retries', 2)
        self.declare_parameter('samples_per_pose', 1)

        # 30cm range (±15cm), 1cm step
        translation_vals = [float(x) for x in np.arange(-0.20, 0.20, 0.05)]
        self.declare_parameter('translation_offsets_m', translation_vals)
        # 45deg range (±22.5deg), 1deg step
        rotation_vals = [float(x) for x in np.arange(-30, 30, 10)]
        self.declare_parameter('rotation_offsets_deg', rotation_vals)
        self.declare_parameter('do_translation_sweep', True)
        self.declare_parameter('do_rotation_sweep', True)

        robot_pose_topic = self.get_parameter('robot_pose_topic').value
        command_topic = self.get_parameter('command_topic').value
        self.start_position = np.asarray(self.get_parameter('start_position').value, dtype=float)
        self.start_orientation = np.asarray(self.get_parameter('start_orientation').value, dtype=float)
        self.capture_service_name = self.get_parameter('capture_service').value
        self.save_service_name = self.get_parameter('save_service').value
        self.auto_save_at_end = bool(self.get_parameter('auto_save_at_end').value)

        self.tick_hz = float(self.get_parameter('tick_hz').value)
        self.settle_time_sec = float(self.get_parameter('settle_time_sec').value)
        self.retry_interval_sec = float(self.get_parameter('retry_interval_sec').value)
        self.max_capture_retries = int(self.get_parameter('max_capture_retries').value)
        self.samples_per_pose = int(self.get_parameter('samples_per_pose').value)

        self.translation_offsets_m = [float(x) for x in self.get_parameter('translation_offsets_m').value]
        self.rotation_offsets_deg = [float(x) for x in self.get_parameter('rotation_offsets_deg').value]
        self.do_translation_sweep = bool(self.get_parameter('do_translation_sweep').value)
        self.do_rotation_sweep = bool(self.get_parameter('do_rotation_sweep').value)

        self.latest_robot_pose: Optional[PoseStamped] = None
        self.base_pose: Optional[PoseStamped] = None
        self.targets: list[TargetPose] = []

        self.current_idx = -1
        self.current_target_started_ns = 0
        self.current_pose_capture_count = 0
        self.capture_retry_count = 0
        self.retry_wait_until_ns = 0

        self.capture_future = None
        self.save_future = None

        self.state = 'WAIT_READY'

        self.create_subscription(PoseStamped, robot_pose_topic, self._robot_pose_cb, 10)
        self.cmd_pub = self.create_publisher(PoseStamped, command_topic, 10)

        self.capture_client = self.create_client(Trigger, self.capture_service_name)
        self.save_client = self.create_client(Trigger, self.save_service_name)

        self.timer = self.create_timer(1.0 / max(self.tick_hz, 1.0), self._tick)

        self.get_logger().info(f'robot pose topic: {robot_pose_topic}')
        self.get_logger().info(f'command topic: {command_topic}')
        self.get_logger().info(f'capture service: {self.capture_service_name}')
        self.get_logger().info('Waiting for robot pose + services to become ready...')

    def _robot_pose_cb(self, msg: PoseStamped) -> None:
        self.latest_robot_pose = msg

    def _tick(self) -> None:
        now_ns = self.get_clock().now().nanoseconds

        if self.state == 'WAIT_READY':
            self._tick_wait_ready()
            return

        if self.state == 'MOVE_AND_SETTLE':
            target = self.targets[self.current_idx]
            self.cmd_pub.publish(target.pose)
            elapsed = (now_ns - self.current_target_started_ns) * 1e-9
            if elapsed >= self.settle_time_sec:
                self._send_capture_request()
                self.state = 'WAIT_CAPTURE'
            return

        if self.state == 'WAIT_CAPTURE':
            self.cmd_pub.publish(self.targets[self.current_idx].pose)
            if self.capture_future is None or not self.capture_future.done():
                return

            resp = self.capture_future.result()
            if resp is not None and resp.success:
                self.current_pose_capture_count += 1
                self.capture_retry_count = 0
                self.get_logger().info(
                    f'[{self.current_idx + 1}/{len(self.targets)}] captured '
                    f'{self.current_pose_capture_count}/{self.samples_per_pose} '
                    f'at {self.targets[self.current_idx].label}'
                )
                if self.current_pose_capture_count >= self.samples_per_pose:
                    self._advance_target(now_ns)
                else:
                    self._send_capture_request()
                return

            self.capture_retry_count += 1
            msg = resp.message if resp is not None else 'service call failed'
            self.get_logger().warn(f'capture failed: {msg}')
            if self.capture_retry_count > self.max_capture_retries:
                self.get_logger().warn('max retries exceeded, skip this pose')
                self._advance_target(now_ns)
                return

            self.retry_wait_until_ns = now_ns + int(self.retry_interval_sec * 1e9)
            self.state = 'RETRY_WAIT'
            return

        if self.state == 'RETRY_WAIT':
            self.cmd_pub.publish(self.targets[self.current_idx].pose)
            if now_ns >= self.retry_wait_until_ns:
                self._send_capture_request()
                self.state = 'WAIT_CAPTURE'
            return

        if self.state == 'SAVE':
            if self.save_future is None:
                self.save_future = self.save_client.call_async(Trigger.Request())
                return
            if not self.save_future.done():
                return
            resp = self.save_future.result()
            if resp is not None:
                self.get_logger().info(f'save_samples: success={resp.success}, msg="{resp.message}"')
            self.state = 'DONE'
            return

        if self.state == 'DONE':
            return

    def _tick_wait_ready(self) -> None:
        if self.latest_robot_pose is None:
            return
        if not self.capture_client.wait_for_service(timeout_sec=0.0):
            return
        if self.auto_save_at_end and not self.save_client.wait_for_service(timeout_sec=0.0):
            return

        self.base_pose = self._make_seed_pose(self.latest_robot_pose.header.frame_id)
        self.targets = self._build_targets(self.base_pose)

        self.current_idx = 0
        self.current_pose_capture_count = 0
        self.capture_retry_count = 0
        self.current_target_started_ns = self.get_clock().now().nanoseconds
        self.state = 'MOVE_AND_SETTLE'

        self.get_logger().info(f'base pose ready, generated {len(self.targets)} target poses')
        self.get_logger().info('Auto collection started')

    def _send_capture_request(self) -> None:
        self.capture_future = self.capture_client.call_async(Trigger.Request())

    def _advance_target(self, now_ns: int) -> None:
        self.current_idx += 1
        self.current_pose_capture_count = 0
        self.capture_retry_count = 0

        if self.current_idx >= len(self.targets):
            if self.auto_save_at_end:
                self.get_logger().info('all targets done, saving samples...')
                self.state = 'SAVE'
            else:
                self.get_logger().info('all targets done')
                self.state = 'DONE'
            return

        self.current_target_started_ns = now_ns
        self.state = 'MOVE_AND_SETTLE'
        self.get_logger().info(
            f'[{self.current_idx + 1}/{len(self.targets)}] moving to {self.targets[self.current_idx].label}'
        )

    def _build_targets(self, base_pose: PoseStamped) -> list[TargetPose]:
        targets: list[TargetPose] = []

        base_pos = np.array([
            float(base_pose.pose.position.x),
            float(base_pose.pose.position.y),
            float(base_pose.pose.position.z),
        ])
        base_quat = np.array([
            float(base_pose.pose.orientation.x),
            float(base_pose.pose.orientation.y),
            float(base_pose.pose.orientation.z),
            float(base_pose.pose.orientation.w),
        ])
        base_rot = R.from_quat(base_quat)

        targets.append(TargetPose(label='base', pose=self._make_pose(base_pose, base_pos, base_rot)))

        if self.do_rotation_sweep:
            axes = [('rx', np.array([1.0, 0.0, 0.0])), ('ry', np.array([0.0, 1.0, 0.0])), ('rz', np.array([0.0, 0.0, 1.0]))]
            for name, axis in axes:
                for deg in self.rotation_offsets_deg:
                    if abs(deg) < 1e-9:
                        continue
                    d_rot = R.from_rotvec(axis * np.deg2rad(deg))
                    rot = base_rot * d_rot
                    targets.append(
                        TargetPose(
                            label=f'{name}:{deg:+.1f}deg',
                            pose=self._make_pose(base_pose, base_pos, rot),
                        )
                    )

        if self.do_translation_sweep:
            axes = [('x', np.array([1.0, 0.0, 0.0])), ('y', np.array([0.0, 1.0, 0.0])), ('z', np.array([0.0, 0.0, 1.0]))]
            for name, axis in axes:
                for d in self.translation_offsets_m:
                    if abs(d) < 1e-9:
                        continue
                    pos = base_pos + axis * d
                    targets.append(
                        TargetPose(
                            label=f'{name}:{d:+.3f}m',
                            pose=self._make_pose(base_pose, pos, base_rot),
                        )
                    )

        # Optional small mixed motions for better excitation.
        if self.do_translation_sweep and self.do_rotation_sweep:
            for d, deg in product(self.translation_offsets_m, self.rotation_offsets_deg):
                if abs(d) < 1e-9 or abs(deg) < 1e-9:
                    continue
                pos = base_pos + np.array([d, 0.0, 0.0])
                rot = base_rot * R.from_rotvec(np.array([0.0, 0.0, np.deg2rad(deg)]))
                targets.append(
                    TargetPose(
                        label=f'mix:x{d:+.3f}_rz{deg:+.1f}',
                        pose=self._make_pose(base_pose, pos, rot),
                    )
                )

        return targets

    def _make_seed_pose(self, frame_id: str) -> PoseStamped:
        msg = PoseStamped()
        msg.header.frame_id = frame_id
        msg.pose.position.x = float(self.start_position[0])
        msg.pose.position.y = float(self.start_position[1])
        msg.pose.position.z = float(self.start_position[2])
        msg.pose.orientation.x = float(self.start_orientation[0])
        msg.pose.orientation.y = float(self.start_orientation[1])
        msg.pose.orientation.z = float(self.start_orientation[2])
        msg.pose.orientation.w = float(self.start_orientation[3])
        return msg

    @staticmethod
    def _make_pose(pos: np.ndarray, quat: np.ndarray) -> PoseStamped:
        msg = PoseStamped()
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        return msg

    @staticmethod
    def _make_pose(base_pose: PoseStamped, pos: np.ndarray, rot: R) -> PoseStamped:
        msg = PoseStamped()
        msg.header.frame_id = base_pose.header.frame_id
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        qx, qy, qz, qw = rot.as_quat()
        msg.pose.orientation.x = float(qx)
        msg.pose.orientation.y = float(qy)
        msg.pose.orientation.z = float(qz)
        msg.pose.orientation.w = float(qw)
        return msg


def main() -> None:
    rclpy.init()
    node = AutoCollector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
