# onr_handeye (minimal)

Minimal ROS2 Python package for collecting paired samples:
- AprilTag pose in camera frame (`cam_T_tag`)
- Robot tool pose in base frame (`base_T_tool`)

Then solve `base_T_cam` with a simple rigid assumption:
`base_T_cam = base_T_tool * tool_T_tag * inv(cam_T_tag)`.

## Quick Start (4 steps)

```bash
# 1) start collector
ros2 launch onr_handeye collect_handeye.launch.py

# 2) capture several poses, then save
ros2 service call /capture_sample std_srvs/srv/Trigger {}
ros2 service call /save_samples std_srvs/srv/Trigger {}

# 3) solve with tool-tag offset (example: +60 mm on tool z)
ros2 run onr_handeye solve_base_camera \
  --samples /home/hfiengineering/Documents/sxj749/ros2_ws/handeye_samples.json \
  --tool-tag-offset-mm 60 \
  --tool-tag-axis z \
  --out /home/hfiengineering/Documents/sxj749/ros2_ws/base_T_cam_result.json

# 4) check residuals (auto-read result/samples metadata)
ros2 run onr_handeye check_projection
```

## 1) Install dependency

`pupil_apriltags`, `pytransform3d`, and `scipy` are Python dependencies:

```bash
pip install pupil-apriltags pytransform3d scipy
```

## 2) Build

```bash
cd ~/Documents/sxj749/ros2_ws
colcon build --packages-select onr_handeye
source install/setup.bash
```

## 3) Start collector

```bash
ros2 launch onr_handeye collect_handeye.launch.py
```

Notes:
- Collector now does **not** apply tool-tag offset parameters. Keep sampling raw `base_T_tool` + `cam_T_tag` pairs.
- Tool/tag rigid transform handling is done at solve/check stage.
- Collector no longer publishes RViz prediction TF (`tag_meas/tag_pred`).
- RViz visualization of current robot pose in camera frame is provided by `check_projection` (see section 6).

Current default topics in node params are:
- image: `/zed/zed_node/left/color/rect/image`
- camera_info: `/zed/zed_node/left/color/rect/camera_info`
- robot_pose (PoseStamped): `/right/manip/measured/tool_int_pose`

## 4) Capture and save samples

To visualize detection result in real time:

```bash
ros2 run rqt_image_view rqt_image_view
```

Then select topic `/onr_handeye/debug_image`.

Overlay meanings:
- Green box + `id=...`: tag detected.
- Red text `AprilTag NOT detected`: no tag in current frame.
- Red text `tag id X not found`: detection exists but not your target id.

At each robot pose (steady), call:

```bash
ros2 service call /capture_sample std_srvs/srv/Trigger {}
```

After collecting >= 5 samples:

```bash
ros2 service call /save_samples std_srvs/srv/Trigger {}
```

This writes `handeye_samples.json` in your current working directory (or set `samples_file` parameter).

## 4.5) Auto collect samples (optional)

`auto_collect_samples` publishes a sequence of target poses and calls `/capture_sample` automatically.

Important:
- Start `collect_samples` first (so `/capture_sample` and `/save_samples` services exist).
- `command_topic` must match your robot command interface topic.
- It does not apply tool-tag offset in collection stage (same policy as manual flow).

Basic run:

```bash
ros2 run onr_handeye auto_collect_samples
```

Typical run with explicit topics/services:

```bash
ros2 run onr_handeye auto_collect_samples --ros-args \
  -p robot_pose_topic:=/right/manip/measured/tool_int_pose \
  -p command_topic:=/righthand/pose \
  -p capture_service:=/capture_sample \
  -p save_service:=/save_samples \
  -p auto_save_at_end:=true
```

Common optional parameters:
- `tick_hz` (default `100.0`): auto collector loop rate.
- `settle_time_sec` (default `1.5`): wait time after sending target pose before capture.
- `samples_per_pose` (default `1`): number of captures at each target pose.
- `max_capture_retries` (default `3`), `retry_interval_sec` (default `0.6`).
- `do_translation_sweep` / `do_rotation_sweep` (both default `true`).
- `translation_offsets_m` (default roughly `[-0.16, 0.04, 0.24]`).
- `rotation_offsets_deg` (default from `-22` to `22` step `2`).

Example: lighter sweep for quick validation:

```bash
ros2 run onr_handeye auto_collect_samples --ros-args \
  -p settle_time_sec:=1.0 \
  -p samples_per_pose:=1 \
  -p do_translation_sweep:=true \
  -p do_rotation_sweep:=true \
  -p translation_offsets_m:="[-0.05, 0.0, 0.05]" \
  -p rotation_offsets_deg:="[-10.0, 0.0, 10.0]"
```

## 5) Solve base-camera transform

If tag frame equals tool frame, use identity `tool_T_tag`:

```bash
ros2 run onr_handeye solve_base_camera --samples handeye_samples.json
```

If tag is offset from tool, pass row-major 4x4 `tool_T_tag`:

```bash
ros2 run onr_handeye solve_base_camera --samples handeye_samples.json \
  --tool-t-tag 1 0 0 0.02 0 1 0 0 0 0 1 0.04 0 0 0 1
```

Result is saved to `base_T_cam_result.json`.

You can tune orientation-vs-translation weighting in error reporting/output metadata:

```bash
ros2 run onr_handeye solve_base_camera --samples handeye_samples.json --rot-weight 0.2
```

### Optional parameters for `solve_base_camera`

```bash
ros2 run onr_handeye solve_base_camera --help
```

Most useful options:
- `--samples`: input sample JSON.
- `--out`: output result JSON path. Default `base_T_cam_result.json`.
- `--tool-t-tag`: 4x4 row-major `tool_T_tag`.
- `--tool-tag-offset-mm`: extra translation (mm) from tool origin to tag origin.
- `--tool-tag-axis`: axis for `--tool-tag-offset-mm` (`x|y|z|-x|-y|-z`).
- `--seed`, `--multistart`, `--rot-weight`: kept for compatibility and metadata.
- `--prune-outliers`: accepted but currently ignored in notebook-style averaging mode.

Example with offset applied only at solve stage:

```bash
ros2 run onr_handeye solve_base_camera \
  --samples /home/hfiengineering/Documents/sxj749/ros2_ws/handeye_samples.json \
  --tool-t-tag 1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1 \
  --tool-tag-offset-mm 60 \
  --tool-tag-axis z \
  --out /home/hfiengineering/Documents/sxj749/ros2_ws/base_T_cam_result.json
```

## 6) Projection check (recommended)

`check_projection` now supports two methods:
- Method A: offline residual statistics (predicted tag vs measured tag on recorded samples).
- Method B: live RViz TF for current robot pose in camera frame.

### Method A: offline residual statistics

After solve, validate predicted vs measured tag poses:

```bash
ros2 run onr_handeye check_projection
```

This default command resolves paths as:
- `--result`: defaults to `base_T_cam_result.json`.
- `--samples`: if omitted, first tries `samples_file` from result JSON, else falls back to `handeye_samples.json`.

Explicit-path version:

```bash
ros2 run onr_handeye check_projection \
  --samples /home/hfiengineering/Documents/sxj749/ros2_ws/handeye_samples.json \
  --result /home/hfiengineering/Documents/sxj749/ros2_ws/base_T_cam_result.json \
  --top-k 10
```

`check_projection` will automatically use `tool_T_tag_used` stored in solve result if available.
You can still override from CLI by passing `--tool-t-tag` and/or `--tool-tag-offset-mm`.

### Method B: live RViz TF (current robot pose in camera frame)

To visualize current robot pose in camera frame in RViz (live TF):

```bash
ros2 run onr_handeye check_projection \
  --publish-rviz-current-robot-pose \
  --result /home/hfiengineering/Documents/sxj749/ros2_ws/base_T_cam_result.json \
  --robot-pose-topic /right/manip/measured/tool_int_pose \
  --camera-frame zed_left_camera_frame_optical \
  --robot-frame robot_current
```

This mode publishes `camera_frame -> robot_frame` with:

```text
cam_T_tool = inv(base_T_cam) @ base_T_tool
```

Notes for live mode:
- Subscribe topic type: `geometry_msgs/PoseStamped` (`--robot-pose-topic`).
- Process keeps running to stream TF; press `Ctrl+C` to stop.

## Notes

- This solver now follows notebook-style averaging from per-sample `cam_T_base` estimates, then inverts to `base_T_cam`.
- This minimal implementation assumes the AprilTag is rigidly mounted on the robot tool.
- If your setup is eye-in-hand or fixed-world target workflow, this collector is still useful, but the solver should be swapped to the corresponding calibration model.
- Image conversion avoids `cv_bridge` so it is robust with NumPy 2 environments.
