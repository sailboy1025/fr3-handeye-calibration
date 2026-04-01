import argparse
import json
from typing import List

import numpy as np
import pytransform3d.transformations as pt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Solve base_T_cam from hand-eye samples under assumption that '
            'tag is rigidly attached to robot tool with known tool_T_tag.'
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
        help='Extra fixed translation from tool origin to tag origin in millimeters.',
    )
    parser.add_argument(
        '--tool-tag-axis',
        choices=['x', 'y', 'z', '-x', '-y', '-z'],
        default='z',
        help='Axis of the extra tool->tag translation offset.',
    )
    parser.add_argument(
        '--rot-weight',
        type=float,
        default=0.2,
        help='Weight for orientation residual (rad) relative to translation residual (m).',
    )
    parser.add_argument(
        '--multistart',
        type=int,
        default=8,
        help='Number of initial guesses for robust optimization. Set 1 to disable.',
    )
    parser.add_argument(
        '--prune-outliers',
        action='store_true',
        help='Enable MAD-based outlier pruning and re-solve on inliers.',
    )
    parser.add_argument(
        '--mad-scale',
        type=float,
        default=3.0,
        help='MAD scale threshold used when --prune-outliers is enabled.',
    )
    parser.add_argument(
        '--max-prune-ratio',
        type=float,
        default=0.25,
        help='Maximum ratio of samples that can be pruned as outliers.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed used for multi-start perturbations.',
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


def se3_to_params(t_base_cam: np.ndarray) -> np.ndarray:
    t = np.asarray(t_base_cam[:3, 3], dtype=float)
    rotvec = R.from_matrix(t_base_cam[:3, :3]).as_rotvec()
    return np.array([t[0], t[1], t[2], rotvec[0], rotvec[1], rotvec[2]], dtype=float)


def params_to_se3(x: np.ndarray) -> np.ndarray:
    rot = R.from_rotvec(np.asarray(x[3:6], dtype=float)).as_matrix()
    return pt.transform_from(rot, np.asarray(x[:3], dtype=float))


def initial_guess(samples: List[dict], tool_t_tag: np.ndarray) -> np.ndarray:
    est = []
    for s in samples:
        base_t_tool = se3_from_list(s['base_T_tool'])
        cam_t_tag = se3_from_list(s['cam_T_tag'])
        est.append(base_t_tool @ tool_t_tag @ pt.invert_transform(cam_t_tag))

    t_mean = np.mean(np.array([e[:3, 3] for e in est], dtype=float), axis=0)
    seed = est[0].copy()
    seed[:3, 3] = t_mean
    return seed


def sample_based_estimates(samples: List[dict], tool_t_tag: np.ndarray) -> List[np.ndarray]:
    est = []
    for s in samples:
        base_t_tool = se3_from_list(s['base_T_tool'])
        cam_t_tag = se3_from_list(s['cam_T_tag'])
        est.append(base_t_tool @ tool_t_tag @ pt.invert_transform(cam_t_tag))
    return est


def residual_vector(x: np.ndarray, samples: List[dict], tool_t_tag: np.ndarray, rot_weight: float) -> np.ndarray:
    base_t_cam = params_to_se3(x)
    cam_t_base = pt.invert_transform(base_t_cam)
    residuals = []

    for s in samples:
        base_t_tool = se3_from_list(s['base_T_tool'])
        cam_t_tag_meas = se3_from_list(s['cam_T_tag'])
        cam_t_tag_pred = cam_t_base @ base_t_tool @ tool_t_tag

        t_err = np.asarray(cam_t_tag_pred[:3, 3], dtype=float) - np.asarray(cam_t_tag_meas[:3, 3], dtype=float)
        rot_delta = pt.invert_transform(cam_t_tag_meas) @ cam_t_tag_pred
        rot_err = R.from_matrix(rot_delta[:3, :3]).as_rotvec()

        residuals.extend(t_err.tolist())
        residuals.extend((rot_weight * rot_err).tolist())

    return np.array(residuals, dtype=float)


def _run_optimizer(x0: np.ndarray, samples: List[dict], tool_t_tag: np.ndarray, rot_weight: float):
    return least_squares(
        residual_vector,
        x0,
        args=(samples, tool_t_tag, rot_weight),
        method='trf',
        loss='huber',
        f_scale=0.01,
        max_nfev=500,
    )


def _build_start_points(
    samples: List[dict],
    tool_t_tag: np.ndarray,
    multistart: int,
    seed: int,
) -> List[np.ndarray]:
    starts = [se3_to_params(initial_guess(samples, tool_t_tag))]
    for est in sample_based_estimates(samples, tool_t_tag):
        starts.append(se3_to_params(est))

    if multistart <= len(starts):
        return starts[:multistart]

    rng = np.random.default_rng(seed)
    base = starts[0]
    while len(starts) < multistart:
        x = base.copy()
        x[:3] += rng.normal(loc=0.0, scale=0.03, size=3)
        x[3:6] += rng.normal(loc=0.0, scale=0.15, size=3)
        starts.append(x)

    return starts


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


def _mad_scale(arr: np.ndarray) -> float:
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return 1.4826 * mad + 1e-9


def prune_outliers(
    samples: List[dict],
    trans_err: np.ndarray,
    rot_err_deg: np.ndarray,
    mad_scale: float,
    max_prune_ratio: float,
) -> tuple[List[dict], List[int]]:
    med_t = float(np.median(trans_err))
    med_r = float(np.median(rot_err_deg))
    sig_t = _mad_scale(trans_err)
    sig_r = _mad_scale(rot_err_deg)

    outlier_mask = (
        (trans_err > (med_t + mad_scale * sig_t))
        | (rot_err_deg > (med_r + mad_scale * sig_r))
    )

    outlier_indices = np.where(outlier_mask)[0].tolist()
    max_prune = int(np.floor(max_prune_ratio * len(samples)))
    if len(outlier_indices) > max_prune:
        severity = (trans_err - med_t) / sig_t + (rot_err_deg - med_r) / sig_r
        ranked = np.argsort(-severity)
        outlier_indices = sorted([int(i) for i in ranked[:max_prune]])
        outlier_mask = np.zeros(len(samples), dtype=bool)
        outlier_mask[outlier_indices] = True

    inliers = [s for i, s in enumerate(samples) if not outlier_mask[i]]
    return inliers, outlier_indices


def solve_base_t_cam(
    samples: List[dict],
    tool_t_tag: np.ndarray,
    rot_weight: float,
    multistart: int,
    seed: int,
) -> np.ndarray:
    # Match notebook logic:
    # cam_T_base_i = cam_T_tag_i @ inv(base_T_tool_i @ tool_T_tag)
    # then average rotation (Markley) and translation.
    cam_t_base_samples = []
    for s in samples:
        base_t_tool = se3_from_list(s['base_T_tool'])
        cam_t_tag = se3_from_list(s['cam_T_tag'])
        base_t_tag = base_t_tool @ tool_t_tag
        cam_t_base = cam_t_tag @ pt.invert_transform(base_t_tag)
        cam_t_base_samples.append(cam_t_base)

    cam_t_base_samples = np.array(cam_t_base_samples, dtype=float)
    rs = cam_t_base_samples[:, :3, :3]
    ts = cam_t_base_samples[:, :3, 3]

    quats = R.from_matrix(rs).as_quat()  # [x, y, z, w]
    quats_wxyz = np.column_stack([quats[:, 3], quats[:, 0], quats[:, 1], quats[:, 2]])
    m = np.zeros((4, 4), dtype=float)
    for q in quats_wxyz:
        m += np.outer(q, q)
    eigvals, eigvecs = np.linalg.eig(m)
    q_avg = eigvecs[:, np.argmax(eigvals)].real
    q_avg /= np.linalg.norm(q_avg)
    q_avg_xyzw = np.array([q_avg[1], q_avg[2], q_avg[3], q_avg[0]], dtype=float)

    cam_t_base_avg = np.eye(4, dtype=float)
    cam_t_base_avg[:3, :3] = R.from_quat(q_avg_xyzw).as_matrix()
    cam_t_base_avg[:3, 3] = np.mean(ts, axis=0)

    base_t_cam = pt.invert_transform(cam_t_base_avg)

    trans_err, rot_err_deg = evaluate_sample_errors(base_t_cam, samples, tool_t_tag)

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
        rot_weight=args.rot_weight,
        multistart=args.multistart,
        seed=args.seed,
    )

    inlier_indices = list(range(len(samples)))
    outlier_indices: List[int] = []

    if args.prune_outliers:
        print('note: --prune-outliers is ignored in notebook-style averaging mode')

    result = {
        'samples_file': str(args.samples),
        'base_T_cam': base_t_cam.reshape(-1).tolist(),
        'base_T_cam_matrix': base_t_cam.tolist(),
        'tool_T_tag_used': tool_t_tag.reshape(-1).tolist(),
        'tool_T_tag_used_matrix': tool_t_tag.tolist(),
        'tool_tag_offset_mm': float(args.tool_tag_offset_mm),
        'tool_tag_axis': str(args.tool_tag_axis),
        'rot_weight': float(args.rot_weight),
        'multistart': int(args.multistart),
        'prune_outliers': bool(args.prune_outliers),
        'inlier_indices': inlier_indices,
        'outlier_indices': outlier_indices,
    }

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    print('base_T_cam solved and written to:', args.out)


if __name__ == '__main__':
    main()
