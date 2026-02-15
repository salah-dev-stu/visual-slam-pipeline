#!/usr/bin/env python3

import argparse
import os
import sys
import struct
import bisect
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree


# TUM Freiburg2 camera intrinsics
FX = 525.0
FY = 525.0
CX = 319.5
CY = 239.5
IMG_W = 640
IMG_H = 480
DEPTH_SCALE = 5000.0  # 16-bit PNG value -> meters


def parse_tum_file(filepath):
    entries = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            entries.append(line.split())
    return entries


def load_ground_truth(dataset_path):
    gt_file = os.path.join(dataset_path, 'groundtruth.txt')
    if not os.path.exists(gt_file):
        print(f"ERROR: Ground truth file not found: {gt_file}")
        return []
    entries = parse_tum_file(gt_file)
    poses = []
    for e in entries:
        if len(e) >= 8:
            ts = float(e[0])
            tx, ty, tz = float(e[1]), float(e[2]), float(e[3])
            qx, qy, qz, qw = float(e[4]), float(e[5]), float(e[6]), float(e[7])
            poses.append((ts, tx, ty, tz, qx, qy, qz, qw))
    poses.sort(key=lambda x: x[0])
    return poses


def load_depth_list(dataset_path):
    depth_file = os.path.join(dataset_path, 'depth.txt')
    if not os.path.exists(depth_file):
        print(f"ERROR: Depth file not found: {depth_file}")
        return []
    entries = parse_tum_file(depth_file)
    depths = []
    for e in entries:
        if len(e) >= 2:
            ts = float(e[0])
            path = e[1]
            depths.append((ts, path))
    depths.sort(key=lambda x: x[0])
    return depths


def load_rgb_list(dataset_path):
    rgb_file = os.path.join(dataset_path, 'rgb.txt')
    if not os.path.exists(rgb_file):
        return []
    entries = parse_tum_file(rgb_file)
    rgbs = []
    for e in entries:
        if len(e) >= 2:
            ts = float(e[0])
            path = e[1]
            rgbs.append((ts, path))
    rgbs.sort(key=lambda x: x[0])
    return rgbs


def find_closest_timestamp(target_ts, sorted_list, max_diff=0.02):
    if not sorted_list:
        return None
    timestamps = [x[0] for x in sorted_list]
    idx = bisect.bisect_left(timestamps, target_ts)
    best = None
    best_diff = float('inf')
    for i in [idx - 1, idx]:
        if 0 <= i < len(sorted_list):
            diff = abs(sorted_list[i][0] - target_ts)
            if diff < best_diff:
                best_diff = diff
                best = i
    if best is not None and best_diff <= max_diff:
        return sorted_list[best]
    return None


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n < 1e-10:
        return np.eye(3)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n

    R = np.array([
        [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),       1 - 2*(qx*qx + qz*qz),   2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)]
    ])
    return R


def load_depth_image(filepath):
    import cv2
    depth = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    depth_m = depth.astype(np.float64) / DEPTH_SCALE
    depth_m[depth == 0] = 0.0
    return depth_m


def load_ply(filepath):
    if not os.path.exists(filepath):
        print(f"PLY file not found: {filepath}")
        return None

    with open(filepath, 'rb') as f:
        is_binary_le = False
        is_binary_be = False
        vertex_count = 0
        header_lines = []
        properties = []

        while True:
            line = f.readline()
            if not line:
                break
            line_str = line.decode('ascii', errors='ignore').strip()
            header_lines.append(line_str)

            if line_str.startswith('format'):
                if 'binary_little_endian' in line_str:
                    is_binary_le = True
                elif 'binary_big_endian' in line_str:
                    is_binary_be = True
            elif line_str.startswith('element vertex'):
                vertex_count = int(line_str.split()[-1])
            elif line_str.startswith('property'):
                parts = line_str.split()
                if len(parts) >= 3:
                    properties.append((parts[1], parts[2]))
            elif line_str == 'end_header':
                break

        if vertex_count == 0:
            print(f"No vertices found in PLY: {filepath}")
            return None

        prop_names = [p[1] for p in properties]
        try:
            xi = prop_names.index('x')
            yi = prop_names.index('y')
            zi = prop_names.index('z')
        except ValueError:
            print(f"PLY missing x/y/z properties: {filepath}")
            return None

        points = np.zeros((vertex_count, 3), dtype=np.float64)

        if is_binary_le or is_binary_be:
            # Determine byte layout from property types
            endian = '<' if is_binary_le else '>'
            fmt_map = {
                'float': 'f', 'float32': 'f',
                'double': 'd', 'float64': 'd',
                'int': 'i', 'int32': 'i',
                'uint': 'I', 'uint32': 'I',
                'short': 'h', 'int16': 'h',
                'ushort': 'H', 'uint16': 'H',
                'char': 'b', 'int8': 'b',
                'uchar': 'B', 'uint8': 'B',
            }
            fmt_str = endian
            for ptype, pname in properties:
                fmt_str += fmt_map.get(ptype, 'f')
            row_size = struct.calcsize(fmt_str)

            for i in range(vertex_count):
                data = f.read(row_size)
                if len(data) < row_size:
                    break
                vals = struct.unpack(fmt_str, data)
                points[i, 0] = vals[xi]
                points[i, 1] = vals[yi]
                points[i, 2] = vals[zi]
        else:
            # ASCII format
            for i in range(vertex_count):
                line = f.readline().decode('ascii', errors='ignore').strip()
                if not line:
                    break
                vals = line.split()
                points[i, 0] = float(vals[xi])
                points[i, 1] = float(vals[yi])
                points[i, 2] = float(vals[zi])

    return points


def save_ply(filepath, points, colors=None):
    n = len(points)
    has_color = colors is not None and len(colors) == n
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"
            if has_color:
                line += f" {int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}"
            f.write(line + "\n")


def build_gt_point_cloud(dataset_path, frame_step=30, pixel_step=4,
                         max_depth=8.0, gt_ply_path=None, colored=True):
    print("\nBuilding GT point cloud...")

    gt_poses = load_ground_truth(dataset_path)
    depth_list = load_depth_list(dataset_path)
    rgb_list = load_rgb_list(dataset_path) if colored else []

    if not gt_poses:
        print("ERROR: No ground truth poses loaded")
        return None
    if not depth_list:
        print("ERROR: No depth images loaded")
        return None

    if colored:
        try:
            import cv2
        except ImportError:
            print("  WARNING: OpenCV not available, disabling color")
            colored = False

    us = np.arange(0, IMG_W, pixel_step)
    vs = np.arange(0, IMG_H, pixel_step)
    uu, vv = np.meshgrid(us, vs)
    uu = uu.flatten()
    vv = vv.flatten()

    all_points = []
    all_colors = []

    for i in range(0, len(depth_list), frame_step):
        depth_ts, depth_rel_path = depth_list[i]
        gt_match = find_closest_timestamp(depth_ts, gt_poses, max_diff=0.05)
        if gt_match is None:
            continue

        _, tx, ty, tz, qx, qy, qz, qw = gt_match
        R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
        t = np.array([tx, ty, tz])

        # Load depth image
        depth_path = os.path.join(dataset_path, depth_rel_path)
        depth_m = load_depth_image(depth_path)
        if depth_m is None:
            continue

        # Load color image if available
        color_img = None
        if colored and rgb_list:
            rgb_match = find_closest_timestamp(depth_ts, rgb_list, max_diff=0.05)
            if rgb_match is not None:
                import cv2
                rgb_path = os.path.join(dataset_path, rgb_match[1])
                color_img = cv2.imread(rgb_path)
                if color_img is not None:
                    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        # Sample depth at grid points
        z_vals = depth_m[vv, uu]

        valid = (z_vals > 0) & (z_vals < max_depth)
        z_valid = z_vals[valid]
        u_valid = uu[valid].astype(np.float64)
        v_valid = vv[valid].astype(np.float64)

        if len(z_valid) == 0:
            continue

        x_cam = (u_valid - CX) * z_valid / FX
        y_cam = (v_valid - CY) * z_valid / FY
        pts_cam = np.stack([x_cam, y_cam, z_valid], axis=1)  # Nx3

        pts_world = (R @ pts_cam.T).T + t  # Nx3

        all_points.append(pts_world)

        if colored and color_img is not None:
            u_int = uu[valid].astype(int)
            v_int = vv[valid].astype(int)
            colors = color_img[v_int, u_int]  # Nx3 (RGB)
            all_colors.append(colors)
        elif colored:
            all_colors.append(np.full((len(z_valid), 3), 128, dtype=np.uint8))

    if not all_points:
        print("ERROR: No valid depth frames processed")
        return None

    points = np.vstack(all_points)
    colors = np.vstack(all_colors) if colored and all_colors else None

    points, colors = voxel_downsample(points, voxel_size=0.01, colors=colors)

    if gt_ply_path is None:
        gt_ply_path = os.path.join(dataset_path, 'gt_pointcloud.ply')
    save_ply(gt_ply_path, points, colors=colors)

    return points


def voxel_downsample(points, voxel_size=0.01, colors=None):
    if len(points) == 0:
        return points, colors

    voxel_indices = np.floor(points / voxel_size).astype(np.int64)

    seen = {}
    for i in range(len(voxel_indices)):
        key = (voxel_indices[i, 0], voxel_indices[i, 1], voxel_indices[i, 2])
        if key not in seen:
            seen[key] = i

    indices = sorted(seen.values())
    ds_points = points[indices]
    ds_colors = colors[indices] if colors is not None else None
    return ds_points, ds_colors


def umeyama_alignment(src, dst):
    n = len(src)
    assert n >= 3 and len(dst) == n

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_c = src - src_mean
    dst_c = dst - dst_mean

    sigma_src = np.mean(np.sum(src_c**2, axis=1))

    H = (dst_c.T @ src_c) / n

    U, S, Vt = np.linalg.svd(H)

    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1

    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / sigma_src
    t = dst_mean - scale * R @ src_mean

    return scale, R, t


def align_trajectory_to_gt(est_traj, gt_poses, max_diff=0.05):
    est_pts = []
    gt_pts = []

    for ep in est_traj:
        ts = ep[0]
        gt_match = find_closest_timestamp(ts, gt_poses, max_diff=max_diff)
        if gt_match is None:
            continue
        est_pts.append([ep[1], ep[2], ep[3]])
        gt_pts.append([gt_match[1], gt_match[2], gt_match[3]])

    if len(est_pts) < 3:
        return None

    est_pts = np.array(est_pts)
    gt_pts = np.array(gt_pts)

    scale, R, t = umeyama_alignment(est_pts, gt_pts)

    aligned = scale * (R @ est_pts.T).T + t
    errors = np.sqrt(np.sum((aligned - gt_pts)**2, axis=1))
    ate_rmse = np.sqrt(np.mean(errors**2))

    print(f"  Alignment: scale={scale:.4f}, ATE RMSE={ate_rmse:.4f}m, "
          f"using {len(est_pts)} pose correspondences")

    return scale, R, t


def load_tum_trajectory(filepath):
    entries = parse_tum_file(filepath)
    traj = []
    for e in entries:
        if len(e) >= 4:
            ts = float(e[0])
            tx, ty, tz = float(e[1]), float(e[2]), float(e[3])
            traj.append((ts, tx, ty, tz))
    return traj


def load_tum_trajectory_full(filepath):
    entries = parse_tum_file(filepath)
    traj = []
    for e in entries:
        if len(e) >= 8:
            ts = float(e[0])
            tx, ty, tz = float(e[1]), float(e[2]), float(e[3])
            qx, qy, qz, qw = float(e[4]), float(e[5]), float(e[6]), float(e[7])
            traj.append((ts, tx, ty, tz, qx, qy, qz, qw))
    return traj


def build_dense_map_from_trajectory(est_traj_full, dataset_path, alignment,
                                     frame_step=3, pixel_step=4, max_depth=5.0):
    print(f"\nBuilding dense map from {len(est_traj_full)} poses...")

    a_scale, a_R, a_t = alignment
    depth_list = load_depth_list(dataset_path)

    import cv2

    us = np.arange(0, IMG_W, pixel_step)
    vs = np.arange(0, IMG_H, pixel_step)
    uu, vv = np.meshgrid(us, vs)
    uu = uu.flatten()
    vv = vv.flatten()

    all_points = []

    for i in range(0, len(est_traj_full), frame_step):
        ts, tx, ty, tz, qx, qy, qz, qw = est_traj_full[i]

        # Find closest depth image
        depth_match = find_closest_timestamp(ts, depth_list, max_diff=0.05)
        if depth_match is None:
            continue

        depth_path = os.path.join(dataset_path, depth_match[1])
        depth_m = load_depth_image(depth_path)
        if depth_m is None:
            continue

        # Estimated pose (camera in SLAM world frame)
        R_est = quaternion_to_rotation_matrix(qx, qy, qz, qw)
        t_est = np.array([tx, ty, tz])

        # Align pose to GT frame:
        # - Position: scale applies to camera position only
        # - Rotation: alignment rotation applied
        # - Depth is already in true meters (Kinect), so NOT scaled
        R_aligned = a_R @ R_est
        t_aligned = a_scale * (a_R @ t_est) + a_t

        # Sample depth at grid points
        z_vals = depth_m[vv, uu]
        valid = (z_vals > 0) & (z_vals < max_depth)
        z_valid = z_vals[valid]
        u_valid = uu[valid].astype(np.float64)
        v_valid = vv[valid].astype(np.float64)

        if len(z_valid) == 0:
            continue

        x_cam = (u_valid - CX) * z_valid / FX
        y_cam = (v_valid - CY) * z_valid / FY
        pts_cam = np.stack([x_cam, y_cam, z_valid], axis=1)  # Nx3

        pts_gt = (R_aligned @ pts_cam.T).T + t_aligned

        all_points.append(pts_gt)

    if not all_points:
        print("  ERROR: No frames processed")
        return None

    points = np.vstack(all_points)

    points, _ = voxel_downsample(points, voxel_size=0.01)

    return points


def compute_metrics(est_points, gt_points, thresholds_cm=[1.0, 2.0, 5.0, 10.0]):
    gt_tree = cKDTree(gt_points)
    est_tree = cKDTree(est_points)

    acc_dists, _ = gt_tree.query(est_points, k=1, workers=-1)

    comp_dists, _ = est_tree.query(gt_points, k=1, workers=-1)

    results = {}

    results['accuracy_mean_m'] = float(np.mean(acc_dists))
    results['accuracy_median_m'] = float(np.median(acc_dists))
    results['accuracy_std_m'] = float(np.std(acc_dists))
    results['accuracy_90pct_m'] = float(np.percentile(acc_dists, 90))
    results['completeness_mean_m'] = float(np.mean(comp_dists))
    results['completeness_median_m'] = float(np.median(comp_dists))
    results['completeness_std_m'] = float(np.std(comp_dists))

    margin = 0.5
    est_min = est_points.min(axis=0) - margin
    est_max = est_points.max(axis=0) + margin
    in_region = np.all((gt_points >= est_min) & (gt_points <= est_max), axis=1)
    gt_in_region = gt_points[in_region]
    comp_dists_region = comp_dists[in_region]
    results['gt_in_region'] = int(np.sum(in_region))
    if len(gt_in_region) > 0:
        results['completeness_region_mean_m'] = float(np.mean(comp_dists_region))
        results['completeness_region_median_m'] = float(np.median(comp_dists_region))
    else:
        results['completeness_region_mean_m'] = float('inf')
        results['completeness_region_median_m'] = float('inf')

    for thr_cm in thresholds_cm:
        thr_m = thr_cm / 100.0
        precision = float(np.mean(acc_dists < thr_m))     # % of est points within thr of GT
        recall = float(np.mean(comp_dists < thr_m))        # % of GT points within thr of est
        if precision + recall > 0:
            fscore = 2 * precision * recall / (precision + recall)
        else:
            fscore = 0.0
        results[f'precision_{thr_cm}cm'] = precision
        results[f'recall_{thr_cm}cm'] = recall
        results[f'fscore_{thr_cm}cm'] = fscore

        if len(gt_in_region) > 0:
            recall_reg = float(np.mean(comp_dists_region < thr_m))
            if precision + recall_reg > 0:
                fscore_reg = 2 * precision * recall_reg / (precision + recall_reg)
            else:
                fscore_reg = 0.0
            results[f'recall_region_{thr_cm}cm'] = recall_reg
            results[f'fscore_region_{thr_cm}cm'] = fscore_reg

    results['num_est_points'] = len(est_points)
    results['num_gt_points'] = len(gt_points)

    return results


def print_metrics(results, label=""):
    print(f"\n{'='*60}")
    if label:
        print(f"  {label}")
        print(f"{'='*60}")
    print(f"  Estimated points: {results['num_est_points']:,}")
    print(f"  GT points:        {results['num_gt_points']:,}")
    gt_region = results.get('gt_in_region', 0)
    if gt_region > 0:
        print(f"  GT in est region: {gt_region:,} ({gt_region/results['num_gt_points']*100:.1f}%)")
    print()
    print(f"  Accuracy (est->gt): how close are our points to true geometry")
    print(f"    Mean:   {results['accuracy_mean_m']*100:.2f} cm")
    print(f"    Median: {results['accuracy_median_m']*100:.2f} cm")
    print(f"    Std:    {results['accuracy_std_m']*100:.2f} cm")
    print(f"    90th %%: {results['accuracy_90pct_m']*100:.2f} cm")
    print()
    print(f"  Completeness (gt->est): how much of the scene is covered")
    print(f"    Full scene:")
    print(f"      Mean:   {results['completeness_mean_m']*100:.2f} cm")
    print(f"      Median: {results['completeness_median_m']*100:.2f} cm")
    if gt_region > 0:
        print(f"    Within estimated map region only ({gt_region:,} GT pts):")
        print(f"      Mean:   {results['completeness_region_mean_m']*100:.2f} cm")
        print(f"      Median: {results['completeness_region_median_m']*100:.2f} cm")
    print()
    print(f"  F-score (Precision / Recall / F1):")
    print(f"    Full scene:")
    for key in sorted(results.keys()):
        if key.startswith('fscore_') and 'region' not in key:
            thr = key.replace('fscore_', '').replace('cm', '')
            p = results[f'precision_{thr}cm']
            r = results[f'recall_{thr}cm']
            f = results[key]
            print(f"      {thr:>5s} cm:  P={p*100:6.2f}%  R={r*100:6.2f}%  F1={f*100:6.2f}%")
    # Region-limited F-scores
    has_region = any(k.startswith('fscore_region_') for k in results)
    if has_region:
        print(f"    Within estimated map region:")
        for key in sorted(results.keys()):
            if key.startswith('fscore_region_'):
                thr = key.replace('fscore_region_', '').replace('cm', '')
                p = results[f'precision_{thr}cm']
                r = results.get(f'recall_region_{thr}cm', 0)
                f = results[key]
                print(f"      {thr:>5s} cm:  P={p*100:6.2f}%  R={r*100:6.2f}%  F1={f*100:6.2f}%")
    print(f"{'='*60}")


def print_comparison_table(all_results):
    if len(all_results) < 2:
        return

    print(f"\n{'='*80}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*80}")

    # Header
    systems = list(all_results.keys())
    header = f"{'Metric':<30s}"
    for s in systems:
        header += f"  {s:>18s}"
    print(header)
    print("-" * 80)

    # Rows
    rows = [
        ("# Points", 'num_est_points', '{:>18,d}'),
        ("Accuracy Mean (cm)", 'accuracy_mean_m', '{:>17.2f}'),
        ("Accuracy Median (cm)", 'accuracy_median_m', '{:>17.2f}'),
        ("Completeness Mean (cm)", 'completeness_mean_m', '{:>17.2f}'),
        ("Completeness Median (cm)", 'completeness_median_m', '{:>17.2f}'),
    ]

    for label, key, fmt in rows:
        line = f"{label:<30s}"
        for s in systems:
            val = all_results[s].get(key, 0)
            if 'cm' in label.lower() and '_m' in key:
                val = val * 100  # convert m to cm
            if isinstance(val, int) or (isinstance(val, float) and val == int(val) and val > 100):
                line += f"  {int(val):>18,d}"
            else:
                line += f"  {val:>18.2f}"
        print(line)

    # F-scores
    thresholds = []
    for key in sorted(list(all_results.values())[0].keys()):
        if key.startswith('fscore_'):
            thresholds.append(key)

    for key in thresholds:
        thr = key.replace('fscore_', '').replace('cm', '')
        label = f"F-score @ {thr}cm (%)"
        line = f"{label:<30s}"
        for s in systems:
            val = all_results[s].get(key, 0) * 100
            line += f"  {val:>17.2f}%"
        print(line)

    print(f"{'='*80}")


def build_orbslam3_pointcloud(orbslam3_dir, dataset_path, gt_poses):
    ply_path = os.path.join(orbslam3_dir, 'map_points.ply')
    if os.path.exists(ply_path):
        print(f"  Found ORB-SLAM3 PLY: {ply_path}")
        return load_ply(ply_path)

    # Try to reconstruct from keyframe trajectory + depth images
    kf_traj_path = os.path.join(orbslam3_dir, 'KeyFrameTrajectory.txt')
    if not os.path.exists(kf_traj_path):
        print(f"  No keyframe trajectory found: {kf_traj_path}")
        return None

    print("  Reconstructing ORB-SLAM3 point cloud from keyframe trajectory + depth...")

    # Load keyframe trajectory (TUM format: timestamp tx ty tz qx qy qz qw)
    entries = parse_tum_file(kf_traj_path)
    kf_poses = []
    for e in entries:
        if len(e) >= 8:
            ts = float(e[0])
            tx, ty, tz = float(e[1]), float(e[2]), float(e[3])
            qx, qy, qz, qw = float(e[4]), float(e[5]), float(e[6]), float(e[7])
            kf_poses.append((ts, tx, ty, tz, qx, qy, qz, qw))

    if not kf_poses:
        print("  ERROR: No keyframe poses loaded")
        return None

    kf_traj_xyz = [(p[0], p[1], p[2], p[3]) for p in kf_poses]
    alignment = align_trajectory_to_gt(kf_traj_xyz, gt_poses)
    if alignment is None:
        print("  ERROR: Could not align ORB-SLAM3 trajectory to GT")
        return None
    a_scale, a_R, a_t = alignment

    # Load depth list
    depth_list = load_depth_list(dataset_path)
    rgb_list = load_rgb_list(dataset_path)

    try:
        import cv2
    except ImportError:
        print("  ERROR: OpenCV not available")
        return None

    # For each keyframe, backproject depth using aligned pose
    pixel_step = 4
    max_depth = 8.0

    us = np.arange(0, IMG_W, pixel_step)
    vs = np.arange(0, IMG_H, pixel_step)
    uu, vv = np.meshgrid(us, vs)
    uu = uu.flatten()
    vv = vv.flatten()

    all_points = []

    for kf in kf_poses:
        ts, tx, ty, tz, qx, qy, qz, qw = kf

        # Find closest depth image
        depth_match = find_closest_timestamp(ts, depth_list, max_diff=0.05)
        if depth_match is None:
            continue

        depth_path = os.path.join(dataset_path, depth_match[1])
        depth_m = load_depth_image(depth_path)
        if depth_m is None:
            continue

        # Use the ORB-SLAM3 estimated pose (apply alignment)
        R_est = quaternion_to_rotation_matrix(qx, qy, qz, qw)
        t_est = np.array([tx, ty, tz])

        # Align to GT frame
        R_aligned = a_R @ R_est
        t_aligned = a_scale * (a_R @ t_est) + a_t

        # Sample depth
        z_vals = depth_m[vv, uu]
        valid = (z_vals > 0) & (z_vals < max_depth)
        z_valid = z_vals[valid]
        u_valid = uu[valid].astype(np.float64)
        v_valid = vv[valid].astype(np.float64)

        if len(z_valid) == 0:
            continue

        x_cam = (u_valid - CX) * z_valid / FX
        y_cam = (v_valid - CY) * z_valid / FY
        pts_cam = np.stack([x_cam, y_cam, z_valid], axis=1)

        pts_world = a_scale * (R_aligned @ pts_cam.T).T + t_aligned

        all_points.append(pts_world)

    if not all_points:
        print("  ERROR: No depth frames processed for ORB-SLAM3")
        return None

    points = np.vstack(all_points)

    points, _ = voxel_downsample(points, voxel_size=0.01)

    return points


def main():
    parser = argparse.ArgumentParser(
        description="3D Map Evaluation: GT point cloud construction and SLAM map comparison")

    parser.add_argument('--dataset', type=str,
                        default='/media/salah/VIVADO_SSD/hw3_full_slam/rgbd_dataset_freiburg2_pioneer_slam3/',
                        help='Path to TUM RGB-D dataset directory')
    parser.add_argument('--build-gt-only', action='store_true',
                        help='Only build and save the GT point cloud, skip evaluation')
    parser.add_argument('--gt-ply', type=str, default=None,
                        help='Path to save/load GT point cloud PLY (default: dataset/gt_pointcloud.ply)')
    parser.add_argument('--estimated', type=str, default=None,
                        help='Path to estimated map PLY from our SLAM (default: HW3/build/estimated_map.ply)')
    parser.add_argument('--orbslam3', type=str, default=None,
                        help='Path to ORB-SLAM3 directory (default: HW2/ORB_SLAM3/)')
    parser.add_argument('--frame-step', type=int, default=30,
                        help='Use every Nth depth frame for GT cloud (default: 30)')
    parser.add_argument('--pixel-step', type=int, default=4,
                        help='Use every Nth pixel for GT cloud (default: 4)')
    parser.add_argument('--max-depth', type=float, default=8.0,
                        help='Maximum depth in meters (default: 8.0)')
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=[1.0, 2.0, 5.0, 10.0],
                        help='F-score thresholds in cm (default: 1 2 5 10)')
    parser.add_argument('--no-color', action='store_true',
                        help='Skip color in GT point cloud')
    parser.add_argument('--no-orbslam3-recon', action='store_true',
                        help='Skip ORB-SLAM3 reconstruction from depth')

    args = parser.parse_args()

    dataset_path = args.dataset
    if not dataset_path.endswith('/'):
        dataset_path += '/'

    if not os.path.isdir(dataset_path):
        print(f"ERROR: Dataset directory not found: {dataset_path}")
        sys.exit(1)

    # Determine default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    hw3_dir = script_dir  # HW3/
    hw2_dir = os.path.join(os.path.dirname(hw3_dir), 'HW2')

    gt_ply_path = args.gt_ply or os.path.join(dataset_path, 'gt_pointcloud.ply')
    est_ply_path = args.estimated or os.path.join(hw3_dir, 'estimated_map.ply')
    orbslam3_dir = args.orbslam3 or os.path.join(hw2_dir, 'ORB_SLAM3')

    # Step 1: GT Point Cloud
    if os.path.exists(gt_ply_path) and not args.build_gt_only:
        print(f"\nLoading existing GT point cloud: {gt_ply_path}")
        gt_points = load_ply(gt_ply_path)
    else:
        gt_points = build_gt_point_cloud(
            dataset_path,
            frame_step=args.frame_step,
            pixel_step=args.pixel_step,
            max_depth=args.max_depth,
            gt_ply_path=gt_ply_path,
            colored=not args.no_color
        )

    if gt_points is None:
        print("ERROR: Failed to build GT point cloud")
        sys.exit(1)

    if args.build_gt_only:
        print(f"\nGT point cloud saved to: {gt_ply_path}")
        print(f"Total points: {len(gt_points):,}")
        sys.exit(0)

    # Step 2: Load GT poses for alignment
    gt_poses = load_ground_truth(dataset_path)

    # Step 3: Evaluate our SLAM system
    all_results = {}

    if os.path.exists(est_ply_path):
        print(f"\nEvaluating SLAM map...")
        print(f"  Loading estimated map: {est_ply_path}")
        est_points = load_ply(est_ply_path)
        if est_points is not None and len(est_points) > 0:
            # Align estimated map to GT frame using trajectory Umeyama
            est_traj_path = os.path.join(hw3_dir, 'estimated_trajectory.txt')
            if os.path.exists(est_traj_path):
                est_traj = load_tum_trajectory(est_traj_path)
                alignment = align_trajectory_to_gt(est_traj, gt_poses)
                if alignment is not None:
                    a_scale, a_R, a_t = alignment
                    est_points = a_scale * (est_points @ a_R.T) + a_t
                    print(f"  Aligned {len(est_points):,} map points to GT frame")

            print(f"  Estimated points: {len(est_points):,}")
            print(f"  GT points: {len(gt_points):,}")

            results = compute_metrics(est_points, gt_points, thresholds_cm=args.thresholds)
            print_metrics(results, label="Sparse Map (triangulated features)")
        else:
            print("  WARNING: No estimated points loaded")
    else:
        print(f"\n  Our SLAM estimated map not found: {est_ply_path}")
        print("  Run the SLAM system first to generate estimated_map.ply")

    # Step 4: Dense map from estimated trajectory + Kinect depth
    est_traj_full_path = os.path.join(hw3_dir, 'estimated_trajectory_full.txt')
    est_traj_path = os.path.join(hw3_dir, 'estimated_trajectory.txt')
    if os.path.exists(est_traj_full_path):
        est_traj_full = load_tum_trajectory_full(est_traj_full_path)
        est_traj_xyz = load_tum_trajectory(est_traj_full_path)
        alignment = align_trajectory_to_gt(est_traj_xyz, gt_poses)

        if alignment is not None and len(est_traj_full) > 0:
            dense_points = build_dense_map_from_trajectory(
                est_traj_full, dataset_path, alignment,
                frame_step=3, pixel_step=4, max_depth=5.0
            )
            if dense_points is not None:
                # Save dense map PLY
                dense_ply_path = os.path.join(hw3_dir, 'dense_map_estimated.ply')
                save_ply(dense_ply_path, dense_points)

                results_dense = compute_metrics(dense_points, gt_points, thresholds_cm=args.thresholds)
                print_metrics(results_dense, label="Dense Map (est. trajectory + Kinect depth)")

    print("\nDone.")


if __name__ == '__main__':
    main()
