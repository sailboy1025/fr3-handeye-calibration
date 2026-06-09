import argparse
import json
from typing import List

import cv2
import numpy as np
import pytransform3d.transformations as pt
from scipy.spatial.transform import Rotation as R


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Solve base_T_cam from hand-eye samples under assumption that '
            'the calibration target is rigidly attached to robot tool with known tool_T_tag.'
        )
    )
    parser.add_argument('--samples', required=True, help='Path to handeye_samples.json')
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
        help='4x4 tool_T_tag row-major matrix. Default is identity.',
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
    parser.add_argument(
        '--handeye-method',
        choices=['TSAI', 'PARK', 'HORAUD', 'ANDREFF', 'DANIILIDIS'],
        default='PARK',
        help='OpenCV cv2.calibrateHandEye method. Default: PARK.',
    )
    parser.add_argument('--out', default='base_T_cam_result.json', help='Output result path')
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


def handeye_method_id(name: str) -> int:
    mapping = {
        'TSAI': cv2.CALIB_HAND_EYE_TSAI,
        'PARK': cv2.CALIB_HAND_EYE_PARK,
        'HORAUD': cv2.CALIB_HAND_EYE_HORAUD,
        'ANDREFF': cv2.CALIB_HAND_EYE_ANDREFF,
        'DANIILIDIS': cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    return mapping[str(name).upper()]


def load_samples(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    return payload['samples']


def mat_from_list(values: List[float]) -> np.ndarray:
    mat = np.array(values, dtype=float).reshape(4, 4)
    if not np.allclose(mat[3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
        raise ValueError('Input transform last row must be [0,0,0,1].')
    return mat


def se3_from_list(values: List[float]) -> np.ndarray:
    return mat_from_list(values)


def evaluate_sample_errors(
    base_t_cam: np.ndarray,
    samples: List[dict],
    tool_t_tag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    trans_err = []
    rot_err_deg = []
    cam_t_base = pt.invert_transform(base_t_cam)
    for s in samples:
        base_t_tool = se3_from_list(s['base_T_tool'])
        cam_t_tag_meas = se3_from_list(s['cam_T_tag'])
        cam_t_tag_pred = cam_t_base @ base_t_tool @ tool_t_tag

        te = np.linalg.norm(
            np.asarray(cam_t_tag_pred[:3, 3], dtype=float)
            - np.asarray(cam_t_tag_meas[:3, 3], dtype=float)
        )
        rot_delta = pt.invert_transform(cam_t_tag_meas) @ cam_t_tag_pred
        re = R.from_matrix(rot_delta[:3, :3]).as_rotvec()
        trans_err.append(float(te))
        rot_err_deg.append(float(np.linalg.norm(np.rad2deg(re))))

    return np.array(trans_err, dtype=float), np.array(rot_err_deg, dtype=float)


def solve_base_t_cam(
    samples: List[dict],
    tool_t_tag: np.ndarray,
    handeye_method: str,
) -> np.ndarray:
    # OpenCV's calibrateHandEye is normally written for eye-in-hand:
    #     base_T_gripper * gripper_T_cam * cam_T_target = base_T_target
    # For this setup the camera is fixed and the ChArUco board is attached
    # to the tool:
    #     base_T_tool * tool_T_tag = base_T_cam * cam_T_tag
    # Feeding tag_T_base as "gripper2base" and cam_T_tag as "target2cam"
    # makes OpenCV's returned cam2gripper transform equal to base_T_cam.
    r_gripper2base = []
    t_gripper2base = []
    r_target2cam = []
    t_target2cam = []

    for s in samples:
        base_t_tool = se3_from_list(s['base_T_tool'])
        cam_t_tag = se3_from_list(s['cam_T_tag'])
        base_t_tag = base_t_tool @ tool_t_tag
        tag_t_base = pt.invert_transform(base_t_tag)

        r_gripper2base.append(np.asarray(tag_t_base[:3, :3], dtype=np.float64))
        t_gripper2base.append(np.asarray(tag_t_base[:3, 3], dtype=np.float64).reshape(3, 1))
        r_target2cam.append(np.asarray(cam_t_tag[:3, :3], dtype=np.float64))
        t_target2cam.append(np.asarray(cam_t_tag[:3, 3], dtype=np.float64).reshape(3, 1))

    r_base_t_cam, t_base_t_cam = cv2.calibrateHandEye(
        r_gripper2base,
        t_gripper2base,
        r_target2cam,
        t_target2cam,
        method=handeye_method_id(handeye_method),
    )
    base_t_cam = pt.transform_from(
        np.asarray(r_base_t_cam, dtype=float),
        np.asarray(t_base_t_cam, dtype=float).reshape(3),
    )

    trans_err, rot_err_deg = evaluate_sample_errors(base_t_cam, samples, tool_t_tag)

    print(f'solver: cv2.calibrateHandEye ({handeye_method})')
    print(f'samples: {len(samples)}')
    print(f'translation error mean (m): {float(np.mean(trans_err)):.6f}')
    print(f'translation error std  (m): {float(np.std(trans_err)):.6f}')
    print(f'rotation error mean (deg): {float(np.mean(rot_err_deg)):.4f}')
    print(f'rotation error std  (deg): {float(np.std(rot_err_deg)):.4f}')

    return base_t_cam


def main() -> None:
    args = parse_args()
    samples = load_samples(args.samples)
    if len(samples) < 5:
        raise ValueError('Need at least 5 samples for a stable solve.')

    tool_t_tag = np.array(args.tool_t_tag, dtype=float).reshape(4, 4)
    tool_t_tag = pt.check_transform(tool_t_tag)
    if abs(float(args.tool_tag_offset_mm)) > 1e-12:
        offset_m = float(args.tool_tag_offset_mm) / 1000.0
        axis = axis_to_unit_vector(args.tool_tag_axis)
        t_extra = pt.transform_from(np.eye(3), axis * offset_m)
        tool_t_tag = tool_t_tag @ t_extra
        print(
            f'apply tool->tag offset: {args.tool_tag_offset_mm:.3f} mm '
            f'along {args.tool_tag_axis} axis'
        )

    base_t_cam = solve_base_t_cam(
        samples=samples,
        tool_t_tag=tool_t_tag,
        handeye_method=args.handeye_method,
    )

    result = {
        'samples_file': str(args.samples),
        'base_T_cam': base_t_cam.reshape(-1).tolist(),
        'base_T_cam_matrix': base_t_cam.tolist(),
        'tool_T_tag_used': tool_t_tag.reshape(-1).tolist(),
        'tool_T_tag_used_matrix': tool_t_tag.tolist(),
        'target_type': 'charuco_board',
        'tool_tag_offset_mm': float(args.tool_tag_offset_mm),
        'tool_tag_axis': str(args.tool_tag_axis),
        'solver': 'cv2.calibrateHandEye',
        'handeye_method': str(args.handeye_method),
    }

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    print('base_T_cam solved and written to:', args.out)


if __name__ == '__main__':
    main()
