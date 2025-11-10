# -*- coding: utf-8 -*-
import os, json, re, math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tensorflow.core.example import example_pb2

# ====================================
# Configuration (paths, plotting, metrics, constants)
# ------------------------------------
# JSON_PATH: path to a JSON file that lists which TFRecord files to read
# and which segment indices (examples) to extract from each file. The JSON
# is expected to be a list of objects like:
#   [{
#       "filename": "training.tfrecord",
#       "segment_indices": [0, 5, 12]
#     }, ...]
# TFREC_DIR: root directory containing the TFRecord files referenced by the JSON.
# OUT_DIR: output directory for CSV files and optional scene images.
# USE_LOCAL_FRAME: if True, transform positions/velocities into the AV-centric
# local coordinate frame (origin at AV current pose; x forward, y left when yaw available).
# ONLY_VEHICLES: if True, filter agents to vehicles only (skip pedestrians/cyclists/others).
# XLIM, YLIM: axis limits for generated scene plots (meters in the chosen frame).
# EXPORT_* flags: control whether to write HV/AV CSVs and save per-scene PNGs.
# CALC_* and PLOT_* toggles: enable metric computation (e.g., lane-center offset)
# and aggregate plots (e.g., boxplots).
# ARROW_*: parameters for drawing direction arrows representing current velocity.
# COLOR_MAP: colors for roadgraph, stop signs, AV/HV trajectories.
# N_AGENTS & N_PAST/N_CUR/N_FUT: fixed shapes matching WOMD TFExample layout.
# LANE_TYPES: roadgraph type codes to be treated as lane/road-center/edge lines.
# STOP_SIGN_TYPES: roadgraph type codes corresponding to stop signs (as confirmed).
# STOP_AS_SINGLE_POINT: if True, reduce each stop sign group into a single representative
# point for distance calculations and plotting.
# STOP_POINT_REDUCER: strategy for reducing multiple stop-sample points ("mean" or "median").
# TIME_STEP_SECONDS: time delta between frames (10Hz => 0.1s).
# STOP_SPEED_RADIUS_M: radius around a stop point within which to search for minimal speed.
# ====================================
JSON_PATH = "./summary.json"
TFREC_DIR = "/mnt/home/Files/Lab/waymo_motion/dataset"
OUT_DIR   = "./extract_out"

USE_LOCAL_FRAME = True
ONLY_VEHICLES   = True
XLIM = (-120, 120)
YLIM = (-120, 120)

EXPORT_HV_CSV = True
EXPORT_AV_CSV = True
EXPORT_SCENE_PNG = True

# Metrics toggles
CALC_STOP_DISTANCE = False
PLOT_STOP_DISTANCE = False
CALC_LANE_CENTER_OFFSET = True
PLOT_LANE_CENTER_OFFSET = True

# Direction arrow drawing parameters (for current velocity visualization)
DRAW_DIR_ARROWS = True
ARROW_LEN   = 6.0
ARROW_WIDTH = 0.006

COLOR_MAP = {
    "lanes": "#666666",
    "stops": "#d62728",
    "av":    "#ff7f0e",
    "hv":    "#1f77b4",
}

# WOMD TFExample fixed shapes
N_AGENTS = 128
N_PAST, N_CUR, N_FUT = 10, 1, 80  # 10Hz => 91 frames total

# Roadgraph types (cover lane centerlines / road lines / road edges)
LANE_TYPES      = {0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16}
STOP_SIGN_TYPES = {17}            # confirmed type definition for stop signs

STOP_AS_SINGLE_POINT = True
STOP_POINT_REDUCER   = "mean"
TIME_STEP_SECONDS    = 0.1
STOP_SPEED_RADIUS_M  = 5.0
# ====================================

os.makedirs(OUT_DIR, exist_ok=True)
for g in tf.config.list_physical_devices('GPU'):
    # Enable dynamic GPU memory growth where supported (helps avoid OOM on small GPUs)
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass

_TFREC_PAT = re.compile(r"\.tfrecords?($|-)", re.IGNORECASE)
# Helper to test whether a filename looks like a TFRecord file
is_tfrec = lambda n: _TFREC_PAT.search(os.path.basename(n)) is not None


def _build_tfrec_index(root_dir):
    """Recursively scan `root_dir` to build a map from TFRecord basename -> full path.

    This allows the JSON to reference files by basename regardless of subdirectory layout.
    Returns
    -------
    dict
        Mapping of {basename: full_path} for all '*.tfrecord' / '*.tfrecords' found.
    """
    mapping = {}
    if not os.path.isdir(root_dir):
        return mapping
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if not is_tfrec(name):
                continue
            full_path = os.path.join(dirpath, name)
            mapping.setdefault(name, full_path)
    return mapping


def load_jobs(p):
    """Load extraction jobs from a JSON manifest.

    Parameters
    ----------
    p : str
        Path to the JSON manifest. The manifest should be a list of items with
        'filename' and optional 'segment_indices'.

    Returns
    -------
    list[tuple[str, list[int]]]
        A list of (resolved_tfrec_path, segment_indices) pairs. If a filename
        is relative, it is resolved under TFREC_DIR. If still not found,
        a directory scan is used to resolve by basename.
    """
    with open(p, "r") as f:
        items = json.load(f)
    jobs = []
    tfrec_index = None
    for it in items:
        fn = it["filename"]
        if not is_tfrec(fn):
            continue
        candidate = fn if os.path.isabs(fn) else os.path.join(TFREC_DIR, fn)
        if not os.path.exists(candidate):
            if tfrec_index is None:
                tfrec_index = _build_tfrec_index(TFREC_DIR)
            candidate = tfrec_index.get(os.path.basename(fn), candidate)
        jobs.append((candidate, [int(x) for x in it.get("segment_indices", [])]))
    return jobs


def fval(ex, key, default=None):
    """Fetch a feature from a parsed tf.train.Example by key.

    Tries to read either float_list or int64_list and returns a NumPy array.
    Returns `default` if key not found or no values present.
    """
    feat = ex.features.feature
    if key not in feat: return default
    fl = feat[key].float_list.value
    il = feat[key].int64_list.value
    if len(fl): return np.array(fl, dtype=np.float32)
    if len(il): return np.array(il, dtype=np.int64)
    return default


def reshape2(a, n, t, fill=np.nan, dtype=np.float32):
    """Reshape a flat array to (n, t), padding with `fill` if not enough values.

    This is useful for WOMD fixed-tensor fields (past/current/future) where we
    need a consistent shape regardless of per-agent validity.
    """
    if a is None: return np.full((n,t), fill, dtype=dtype)
    a = np.asarray(a); need = n*t
    if a.size < need: a = np.concatenate([a, np.full(need-a.size, fill, dtype=a.dtype)])
    return a.reshape(n,t)


def type_names_from_codes(codes):
    """Map WOMD state/type integer codes to human-readable names.

    Unknown codes map to 'OTHER'. If `codes` is None, defaults to an array of zeros.
    """
    mp = {0:"UNSET", 1:"VEHICLE", 2:"PEDESTRIAN", 3:"CYCLIST"}
    if codes is None: codes = np.zeros(N_AGENTS, np.int64)
    return np.vectorize(lambda x: mp.get(int(x), "OTHER"))(codes)


def to_av_local(xs, ys, cur_x, cur_y, vx_c=None, vy_c=None, is_av_mask=None):
    """Convert world-frame positions to an AV-centric local frame.

    The origin is at the AV's current position. The heading is aligned to the AV's
    current velocity when available; otherwise zero yaw is used.

    Parameters
    ----------
    xs, ys : ndarray (N_AGENTS, T)
        World-frame positions for all agents across time.
    cur_x, cur_y : ndarray (N_AGENTS, 1)
        Current-frame positions, used to locate the AV.
    vx_c, vy_c : ndarray (N_AGENTS, 1) or None
        Current-frame velocities to estimate yaw.
    is_av_mask : ndarray (N_AGENTS,) of bool
        Boolean mask indicating which agent is the AV.

    Returns
    -------
    (xs_local, ys_local, x0, y0, yaw0)
        Local-frame positions and the reference transform parameters.
    """
    if not USE_LOCAL_FRAME or is_av_mask is None or not np.any(is_av_mask):
        return xs, ys, 0.0, 0.0, 0.0
    av = np.where(is_av_mask)[0][0]
    x0 = float(cur_x[av,0]); y0 = float(cur_y[av,0])
    if vx_c is not None and vy_c is not None and (abs(vx_c[av,0])+abs(vy_c[av,0])>1e-4):
        yaw0 = np.arctan2(float(vy_c[av,0]), float(vx_c[av,0]))
    else:
        yaw0 = 0.0
    X = xs - x0; Y = ys - y0
    c,s = np.cos(-yaw0), np.sin(-yaw0)
    return c*X - s*Y, s*X + c*Y, x0, y0, yaw0


def world_to_local_xy(x, y, x0, y0, yaw0):
    """Transform a single world-frame (x, y) into the local frame defined by (x0, y0, yaw0)."""
    X,Y = x-x0, y-y0
    c,s = np.cos(-yaw0), np.sin(-yaw0)
    return c*X - s*Y, s*X + c*Y


def vec_world_to_local(vx, vy, yaw0):
    """Rotate a world-frame 2D vector (vx, vy) into the local frame with heading yaw0."""
    c,s = np.cos(-yaw0), np.sin(-yaw0)
    return c*vx - s*vy, s*vx + c*vy


def group_roadgraph_points(xyz, ids):
    """Group roadgraph sample points into polylines by their roadgraph id.

    Parameters
    ----------
    xyz : ndarray (N, 3)
        Roadgraph points (x,y,z) in world coordinates.
    ids : ndarray (N,)
        Roadgraph identifiers. Consecutive points with the same id belong to the same polyline.

    Returns
    -------
    dict[int, ndarray]
        {roadgraph_id: polyline_points( M, 2 )} for polylines with at least 2 points.
    """
    out = {}
    if xyz is None or ids is None or len(xyz)==0: return out
    xy = xyz.reshape(-1,3)[:, :2]; ids = ids.reshape(-1)
    for rg_id in np.unique(ids):
        pts = xy[ids == rg_id]
        if pts.shape[0] >= 2: out[int(rg_id)] = pts
    return out


def reduce_points_to_single(xy, reducer="mean"):
    """Reduce a set of 2D points to a single representative (mean or median)."""
    if xy is None or len(xy)==0: return None
    return np.median(xy, axis=0) if reducer=="median" else xy.mean(axis=0)

# Global accumulators for aggregate plots (e.g., per-scenario distributions)
AV_DIST_TO_STOP, HV_DIST_TO_STOP = [], []
AV_LANE_OFFSET, HV_LANE_OFFSET = [], []
AV_LANE_OFFSET_BY_SCENARIO = {}


def closest_point_on_polyline(polyline, px, py):
    """Find the closest point on a polyline to a query point and its tangent.

    Parameters
    ----------
    polyline : ndarray (M, 2)
        Sequence of 2D points.
    px, py : float
        Query point coordinates.

    Returns
    -------
    (point, distance, tangent)
        point : ndarray (2,) the closest point on the polyline (or first point if degenerate).
        distance : float the Euclidean distance from query to the closest point.
        tangent : ndarray (2,) approximate unit tangent at the closest point (segment direction).
    """
    if polyline is None or len(polyline) == 0:
        return None, None, None
    if len(polyline) == 1:
        point = polyline[0]
        dist = float(np.hypot(point[0] - px, point[1] - py))
        return point, dist, np.array([1.0, 0.0], dtype=np.float32)
    P = np.array([px, py], dtype=np.float32)
    best_point = None
    best_dist = None
    best_tangent = None
    for i in range(len(polyline) - 1):
        A = polyline[i]
        B = polyline[i + 1]
        v = B - A
        seg_len2 = float(np.dot(v, v))
        if seg_len2 < 1e-6:
            continue
        t = float(np.clip(np.dot(P - A, v) / seg_len2, 0.0, 1.0))
        closest = A + t * v
        dist = float(np.linalg.norm(P - closest))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_point = closest
            tangent_vec = v / math.sqrt(seg_len2)
            best_tangent = tangent_vec.astype(np.float32)
    if best_point is None:
        best_point = polyline[0]
        best_dist = float(np.linalg.norm(P - best_point))
        best_tangent = np.array([1.0, 0.0], dtype=np.float32)
    return best_point, best_dist, best_tangent


def _normalize_vector(vec, eps=1e-3):
    """Return `vec / ||vec||` if above threshold; otherwise return None."""
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return None
    return vec / norm


def estimate_heading_at_index(xs_row, ys_row, vidx, idx_in_vidx, default_heading=None):
    """Estimate a unit heading vector for a trajectory at a given valid index.

    Strategy: forward difference; if not possible, backward difference; otherwise
    fall back to `default_heading`; if still unavailable, default to +x.

    Returns
    -------
    ndarray (2,)
        Unit heading vector in the same frame as `xs_row`/`ys_row`.
    """
    t = vidx[idx_in_vidx]
    # Forward difference
    if idx_in_vidx + 1 < len(vidx):
        nxt = vidx[idx_in_vidx + 1]
        vec = np.array([xs_row[nxt] - xs_row[t], ys_row[nxt] - ys_row[t]], dtype=np.float32)
        normed = _normalize_vector(vec)
        if normed is not None:
            return normed
    # Backward difference
    if idx_in_vidx > 0:
        prev = vidx[idx_in_vidx - 1]
        vec = np.array([xs_row[t] - xs_row[prev], ys_row[t] - ys_row[prev]], dtype=np.float32)
        normed = _normalize_vector(vec)
        if normed is not None:
            return normed
    # Fall back to provided default heading
    if default_heading is not None:
        normed = _normalize_vector(default_heading)
        if normed is not None:
            return normed
    # Final fallback: face positive x-axis
    return np.array([1.0, 0.0], dtype=np.float32)


def _speed_at_index(xs_row, ys_row, vidx, idx_in_vidx, dt=TIME_STEP_SECONDS):
    """Estimate scalar speed (m/s) at vidx[idx_in_vidx] using central/forward/backward differences."""
    if len(vidx) == 0:
        return None
    t_cur = vidx[idx_in_vidx]
    if not (np.isfinite(xs_row[t_cur]) and np.isfinite(ys_row[t_cur])):
        return None
    prev_pt = next_pt = None
    if idx_in_vidx > 0:
        t_prev = vidx[idx_in_vidx - 1]
        prev_pt = np.array([xs_row[t_prev], ys_row[t_prev]], dtype=np.float32)
        if not np.all(np.isfinite(prev_pt)):
            prev_pt = None
    if idx_in_vidx + 1 < len(vidx):
        t_next = vidx[idx_in_vidx + 1]
        next_pt = np.array([xs_row[t_next], ys_row[t_next]], dtype=np.float32)
        if not np.all(np.isfinite(next_pt)):
            next_pt = None
    cur_pt = np.array([xs_row[t_cur], ys_row[t_cur]], dtype=np.float32)
    if prev_pt is not None and next_pt is not None:
        dist = float(np.linalg.norm(next_pt - prev_pt))
        return dist / (2.0 * dt)
    if next_pt is not None:
        dist = float(np.linalg.norm(next_pt - cur_pt))
        return dist / dt
    if prev_pt is not None:
        dist = float(np.linalg.norm(cur_pt - prev_pt))
        return dist / dt
    return 0.0


def compute_stop_distance_before_stop(xs_row, ys_row, vidx, stop_xy, radius=STOP_SPEED_RADIUS_M):
    """Within a given radius of a stop point, find the frame with minimal speed and its distance.

    Returns
    -------
    (distance, frame_index)
        If no valid point is found within the radius, returns (None, None).
    """
    if stop_xy is None or len(vidx) == 0:
        return None, None
    stop_point = np.array(stop_xy, dtype=np.float32)
    best_speed = None
    best_dist = None
    best_frame = None
    for idx_in_vidx, t_idx in enumerate(vidx):
        px = xs_row[t_idx]
        py = ys_row[t_idx]
        if not (np.isfinite(px) and np.isfinite(py)):
            continue
        point = np.array([px, py], dtype=np.float32)
        dist = float(np.linalg.norm(point - stop_point))
        if dist > radius:
            continue
        speed = _speed_at_index(xs_row, ys_row, vidx, idx_in_vidx)
        if speed is None:
            continue
        # Prefer the slowest speed; if tie, prefer the closer distance
        if best_speed is None or speed < best_speed - 1e-3 or (abs(speed - best_speed) <= 1e-3 and dist < best_dist):
            best_speed = speed
            best_dist = dist
            best_frame = int(t_idx)
    if best_dist is None:
        return None, None
    return best_dist, best_frame


def compute_lane_offset_for_point(P, heading_vec, lane_segments_local):
    """Compute signed lateral distance from point P to the nearest lane centerline.

    Positive means left of the lane center relative to the heading; negative means right.
    The lane centerline is approximated as the midpoint between the closest left/right
    polylines (e.g., lane boundaries/marks). A maximum lateral threshold (3.7 m) filters
    unlikely matches.
    """
    if not lane_segments_local:
        return None
    left_candidate = None
    right_candidate = None
    for polyline in lane_segments_local.values():
        point, dist, tangent = closest_point_on_polyline(polyline, P[0], P[1])
        if point is None or dist is None:
            continue
        rel = point - P
        # Signed side test using cross product z-component: heading x rel
        side = heading_vec[0] * rel[1] - heading_vec[1] * rel[0]
        candidate = (point, dist, tangent)
        if side >= 0:
            if left_candidate is None or dist < left_candidate[1]:
                left_candidate = candidate
        else:
            if right_candidate is None or dist < right_candidate[1]:
                right_candidate = candidate
    if not left_candidate or not right_candidate:
        return None
    # Filter out obviously wrong pairs (too far from a typical lane boundary)
    if left_candidate[1] > 3.7 or right_candidate[1] > 3.7:
        return None
    center_point = 0.5 * (left_candidate[0] + right_candidate[0])
    tangent_vec = left_candidate[2] + right_candidate[2]
    normed_tangent = _normalize_vector(tangent_vec)
    if normed_tangent is None:
        normed_tangent = heading_vec
    normal_vec = np.array([-normed_tangent[1], normed_tangent[0]], dtype=np.float32)
    return float(np.dot(P - center_point, normal_vec))


def draw_one_sample(ex, base_name, ex_idx, hv_writer=None, av_writer=None):
    """Process one tf.train.Example to generate outputs.

    1) Parse agent trajectories, filter to vehicles (if enabled), and transform
       to AV-local frame if requested.
    2) Parse roadgraph and extract:
       - lane polylines (grouped by id)
       - stop sign points (optionally reduced to a single representative per sign)
    3) Optionally plot:
       - lane polylines, AV/HV trajectories, stop signs, and velocity arrows
    4) Optionally compute per-agent metrics (stop distance, lane center offset)
       and write them to AV/HV CSV rows.

    Returns
    -------
    str or None
        Path to the saved PNG if plotting is enabled; otherwise None.
    """
    # 1) Agents
    obj_type = fval(ex,"state/type")
    is_sdc   = fval(ex,"state/is_sdc", default=np.zeros(N_AGENTS, np.int64))
    track_id = fval(ex,"state/id",     default=np.arange(N_AGENTS, dtype=np.int64))
    past_x = reshape2(fval(ex,"state/past/x"), N_AGENTS,N_PAST)
    past_y = reshape2(fval(ex,"state/past/y"), N_AGENTS,N_PAST)
    cur_x  = reshape2(fval(ex,"state/current/x"), N_AGENTS,N_CUR)
    cur_y  = reshape2(fval(ex,"state/current/y"), N_AGENTS,N_CUR)
    fut_x  = reshape2(fval(ex,"state/future/x"), N_AGENTS,N_FUT)
    fut_y  = reshape2(fval(ex,"state/future/y"), N_AGENTS,N_FUT)
    past_v = reshape2(fval(ex,"state/past/valid"), N_AGENTS,N_PAST, fill=0,dtype=np.int32)
    cur_v  = reshape2(fval(ex,"state/current/valid"), N_AGENTS,N_CUR, fill=0,dtype=np.int32)
    fut_v  = reshape2(fval(ex,"state/future/valid"), N_AGENTS,N_FUT, fill=0,dtype=np.int32)
    vx_c   = reshape2(fval(ex,"state/current/velocity_x", default=None), N_AGENTS,N_CUR)
    vy_c   = reshape2(fval(ex,"state/current/velocity_y", default=None), N_AGENTS,N_CUR)

    xs = np.concatenate([past_x,cur_x,fut_x],axis=1)
    ys = np.concatenate([past_y,cur_y,fut_y],axis=1)
    valids = np.concatenate([past_v,cur_v,fut_v],axis=1)
    types_txt = type_names_from_codes(obj_type)
    is_av = (is_sdc.astype(int)==1)

    xs_loc, ys_loc, x0, y0, yaw0 = to_av_local(xs,ys,cur_x,cur_y,vx_c,vy_c,is_av)
    CUR = N_PAST
    cur_x_loc, cur_y_loc = xs_loc[:,CUR], ys_loc[:,CUR]
    vx_cur = vx_c[:,0] if vx_c is not None else np.zeros((N_AGENTS,),np.float32)
    vy_cur = vy_c[:,0] if vy_c is not None else np.zeros((N_AGENTS,),np.float32)
    vx_loc, vy_loc = vec_world_to_local(vx_cur, vy_cur, yaw0)

    # 2) Roadgraph
    rg_xyz = fval(ex,"roadgraph_samples/xyz")
    rg_id  = fval(ex,"roadgraph_samples/id")
    rg_t   = fval(ex,"roadgraph_samples/type")
    rg_valid = fval(ex,"roadgraph_samples/valid")
    if rg_xyz is not None: rg_xyz = rg_xyz.reshape(-1,3)
    if rg_valid is not None:
        # Respect the roadgraph valid mask when provided
        m = rg_valid.astype(bool)
        if rg_xyz is not None: rg_xyz = rg_xyz[m]
        if rg_id  is not None: rg_id  = rg_id[m]
        if rg_t   is not None: rg_t   = rg_t[m]

    lane_segments = {}
    lane_type_map = {}
    stop_pts_world = np.empty((0,2),dtype=np.float32)
    if rg_t is not None and rg_xyz is not None:
        t = rg_t.reshape(-1)
        # Collect lane-like polylines
        lane_mask = np.isin(t, list(LANE_TYPES))
        if np.any(lane_mask):
            lane_xyz = rg_xyz[lane_mask]
            lane_ids = rg_id[lane_mask] if rg_id is not None else None
            lane_types = rg_t[lane_mask] if rg_t is not None else None
            lane_segments = group_roadgraph_points(lane_xyz, lane_ids)
            if lane_ids is not None and lane_types is not None:
                flat_ids = lane_ids.reshape(-1)
                flat_types = lane_types.reshape(-1)
                for lid in np.unique(flat_ids):
                    mask_lid = (flat_ids == lid)
                    if np.any(mask_lid):
                        lane_type_map[int(lid)] = int(np.round(np.mean(flat_types[mask_lid])))
        # Collect stop sign sample points
        stop_mask = np.isin(t, list(STOP_SIGN_TYPES))
        if np.any(stop_mask):
            stop_pts_world = rg_xyz[stop_mask][:,:2]

    # Reduce stop samples to a single point per sign and convert to local frame,
    # then select the stop point closest to the AV as the representative.
    stop_local = None
    if STOP_AS_SINGLE_POINT and stop_pts_world.shape[0]>0:
        if rg_id is not None and 'stop_mask' in locals() and np.any(stop_mask):
            stop_ids = rg_id[stop_mask]
            cents = []
            for gid in np.unique(stop_ids):
                c = reduce_points_to_single(stop_pts_world[stop_ids==gid], STOP_POINT_REDUCER)
                if c is not None: cents.append(c)
            candidates = np.vstack(cents) if len(cents)>0 else stop_pts_world
        else:
            c = reduce_points_to_single(stop_pts_world, STOP_POINT_REDUCER)
            candidates = c.reshape(1,2) if c is not None else stop_pts_world
        cand_local = []
        for sx,sy in candidates:
            lx,ly = world_to_local_xy(sx,sy,x0,y0,yaw0)
            cand_local.append([lx,ly])
        cand_local = np.asarray(cand_local,np.float32)
        stop_local = cand_local[int(np.argmin(np.sum(cand_local**2,axis=1)))]

    # Build lane polylines in local frame for plotting and lane-offset metrics
    lane_segments_local = {}
    if len(lane_segments)>0:
        for lid, pts in lane_segments.items():
            Xw,Yw = pts[:,0], pts[:,1]
            X,Y = world_to_local_xy(Xw,Yw,x0,y0,yaw0) if np.any(is_av) else (Xw,Yw)
            stack = np.stack([X,Y],axis=1)
            lane_segments_local[lid] = stack

    # 3) Plotting of the scene (optional)
    fig = ax = None
    if EXPORT_SCENE_PNG:
        fig, ax = plt.subplots(figsize=(6,6))
        for pts in lane_segments.values():
            Xw,Yw = pts[:,0], pts[:,1]
            X,Y = world_to_local_xy(Xw,Yw,x0,y0,yaw0) if np.any(is_av) else (Xw,Yw)
            ax.plot(X,Y,color=COLOR_MAP["lanes"],linewidth=0.6,alpha=0.8,zorder=0)
        if stop_local is not None:
            ax.scatter([stop_local[0]],[stop_local[1]], s=36, c=COLOR_MAP["stops"], marker="o", zorder=6)

    # 4) Per-agent CSV writing and optional metrics/visuals
    scenario_key = f"{base_name}_ex{ex_idx:05d}"

    def write_rows(writer, is_av_flag, i, vidx, scenario_key, stop_metric):
        """Write one trajectory (agent i) into CSV, logging per-frame values.

        Additionally, accumulate lane-center offsets into global arrays for
        aggregate visualization, separated by AV/HV and by scenario.
        """
        do_write = writer is not None
        default_heading = np.array([vx_loc[i], vy_loc[i]], dtype=np.float32)
        if _normalize_vector(default_heading) is None:
            default_heading = None
        stop_metric_value = None
        stop_metric_frame = None
        stop_coords = ("", "")
        if stop_metric is not None:
            stop_metric_value, stop_metric_frame = stop_metric
        if stop_metric_value is not None and stop_local is not None:
            stop_coords = (float(stop_local[0]), float(stop_local[1]))

        for idx_pos, t_idx in enumerate(vidx):
            t_idx = int(t_idx)
            px = float(xs_loc[i, t_idx]); py = float(ys_loc[i, t_idx])

            d_stop = ""
            sx = sy = ""
            if stop_metric_value is not None and stop_metric_frame is not None and t_idx == stop_metric_frame:
                d_stop = float(stop_metric_value)
                sx, sy = stop_coords

            lane_offset_val = None
            if CALC_LANE_CENTER_OFFSET and lane_segments_local:
                heading_vec = estimate_heading_at_index(xs_loc[i], ys_loc[i], vidx, idx_pos, default_heading)
                P = np.array([px, py], dtype=np.float32)
                lane_offset_val = compute_lane_offset_for_point(P, heading_vec, lane_segments_local)
                if lane_offset_val is not None:
                    if is_av_flag:
                        AV_LANE_OFFSET.append(lane_offset_val)
                        AV_LANE_OFFSET_BY_SCENARIO.setdefault(scenario_key, []).append(lane_offset_val)
                    else:
                        HV_LANE_OFFSET.append(lane_offset_val)

            if do_write:
                writer.writerow({
                    "file": base_name, "ex_idx": ex_idx, "track_id": int(track_id[i]),
                    "t": t_idx,
                    "x_local": px, "y_local": py, "valid": int(valids[i, t_idx]),
                    "stop_x_local": sx, "stop_y_local": sy, "dist_to_stop": d_stop,
                    "lane_center_offset": ("" if lane_offset_val is None else lane_offset_val),
                })

    for i in range(N_AGENTS):
        if ONLY_VEHICLES and type_names_from_codes([obj_type[i]])[0] != "VEHICLE":
            continue
        vidx = np.where(valids[i]==1)[0]
        if vidx.size==0: continue
        is_av_i = bool(is_av[i])
        X = xs_loc[i,vidx]; Y = ys_loc[i,vidx]
        if EXPORT_SCENE_PNG:
            ax.plot(X,Y, linewidth=(2.4 if is_av_i else 1.2),
                    color=(COLOR_MAP["av"] if is_av_i else COLOR_MAP["hv"]),
                    alpha=1.0 if is_av_i else 0.85, zorder=(10 if is_av_i else 6))
            if DRAW_DIR_ARROWS:
                cx=float(cur_x_loc[i]); cy=float(cur_y_loc[i])
                vx=float(vx_loc[i]);    vy=float(vy_loc[i])
                if np.isfinite(cx) and np.isfinite(cy) and np.hypot(vx,vy)>1e-3:
                    n=(vx*vx+vy*vy)**0.5; dx=(vx/n)*ARROW_LEN; dy=(vy/n)*ARROW_LEN
                    ax.arrow(cx,cy,dx,dy, length_includes_head=True,
                             head_width=ARROW_LEN*0.25, head_length=ARROW_LEN*0.35,
                             linewidth=(2.0 if is_av_i else 1.2),
                             color=(COLOR_MAP["av"] if is_av_i else COLOR_MAP["hv"]),
                             alpha=1.0 if is_av_i else 0.9, width=ARROW_WIDTH,
                             zorder=(12 if is_av_i else 8))
        stop_metric = None
        if CALC_STOP_DISTANCE and stop_local is not None:
            value, frame = compute_stop_distance_before_stop(xs_loc[i], ys_loc[i], vidx, stop_local)
            if value is not None:
                stop_metric = (value, frame)
                if is_av_i:
                    AV_DIST_TO_STOP.append(value)
                else:
                    HV_DIST_TO_STOP.append(value)
        if is_av_i:
            writer_obj = av_writer if (EXPORT_AV_CSV and av_writer is not None) else None
            write_rows(writer_obj, True, i, vidx, scenario_key, stop_metric)
        else:
            writer_obj = hv_writer if (EXPORT_HV_CSV and hv_writer is not None) else None
            write_rows(writer_obj, False, i, vidx, scenario_key, stop_metric)

    if EXPORT_SCENE_PNG:
        ax.set_aspect("equal",adjustable="box")
        ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{base_name} ex{ex_idx} (AV/HV + stop sign + directions)", fontsize=10)
        legend = [
            Line2D([0],[0], color=COLOR_MAP["lanes"], lw=1.2, label="lane/roadgraph"),
            Line2D([0],[0], color=COLOR_MAP["av"], lw=2.4, label="AV trajectory"),
            Line2D([0],[0], color=COLOR_MAP["hv"], lw=1.2, label="HV trajectories"),
            Line2D([0],[0], marker='o', color='w', markerfacecolor=COLOR_MAP["stops"], markersize=6, label="stop sign"),
            Line2D([0],[0], color='k', lw=0, marker=r'$\\rightarrow$', label="direction"),
        ]
        ax.legend(handles=legend, loc="upper right", fontsize=8, frameon=True)
        out_png = os.path.join(OUT_DIR, f"{base_name}_ex{ex_idx:05d}.png")
        fig.tight_layout(); fig.savefig(out_png, dpi=200)
        plt.close(fig)
        return out_png

    return None


def make_stop_distance_boxplot():
    """Create a boxplot for AV/HV distance to stop sign if enabled and data exist."""
    if not PLOT_STOP_DISTANCE:
        return
    if len(AV_DIST_TO_STOP)+len(HV_DIST_TO_STOP) == 0:
        return
    data, labels, colors = [], [], []
    if AV_DIST_TO_STOP:
        data.append(AV_DIST_TO_STOP)
        labels.append("AV dist_to_stop")
        colors.append(COLOR_MAP["av"])
    if HV_DIST_TO_STOP:
        data.append(HV_DIST_TO_STOP)
        labels.append("HV dist_to_stop")
        colors.append(COLOR_MAP["hv"])
    fig, ax = plt.subplots(figsize=(6,5))
    positions = np.arange(1, len(data)+1, dtype=np.float32)
    ax.boxplot(data, positions=positions, labels=labels, showfliers=False)
    rng = np.random.default_rng(42)
    for vals, xpos, color in zip(data, positions, colors):
        if not vals:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            np.full(len(vals), xpos, dtype=np.float32) + jitter,
            vals,
            color=color,
            alpha=0.65,
            s=16,
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )
    ax.set_ylabel("distance (m)")
    ax.set_title("Boxplot: Distance to stop sign")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dist_to_stop_boxplot.png"), dpi=200)
    plt.close(fig)


def make_lane_offset_boxplot():
    """Create a boxplot for signed lane-center offset (scenario-wise, HV all, AV all).

    Positive offset => left of centerline; negative => right of centerline.
    """
    if not PLOT_LANE_CENTER_OFFSET:
        return
    if len(AV_LANE_OFFSET)+len(HV_LANE_OFFSET) == 0:
        return
    scenario_keys = sorted(AV_LANE_OFFSET_BY_SCENARIO.keys())

    plot_data = []
    labels = []
    colors = []
    positions = []
    pos = 1

    for key in scenario_keys:
        av_vals = AV_LANE_OFFSET_BY_SCENARIO.get(key, [])
        if av_vals:
            plot_data.append(av_vals)
            labels.append(f"AV {key}")
            colors.append(COLOR_MAP["av"])
            positions.append(pos)
            pos += 1

    if HV_LANE_OFFSET:
        plot_data.append(HV_LANE_OFFSET)
        labels.append("HV (all)")
        colors.append(COLOR_MAP["hv"])
        positions.append(pos)
        pos += 1

    if AV_LANE_OFFSET:
        plot_data.append(AV_LANE_OFFSET)
        labels.append("AV (all)")
        colors.append(COLOR_MAP["av"])
        positions.append(pos)

    if not plot_data:
        return

    width = max(6.0, 1.1 * len(plot_data))
    fig, ax = plt.subplots(figsize=(width, 5))
    ax.boxplot(plot_data, positions=positions, labels=labels, showfliers=False)

    rng = np.random.default_rng(42)
    for vals, xpos, color in zip(plot_data, positions, colors):
        if not vals:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            np.full(len(vals), xpos, dtype=np.float32) + jitter,
            vals,
            color=color,
            alpha=0.6,
            s=14,
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )

    ax.set_ylabel("signed distance (m)")
    ax.set_title("Boxplot: Signed distance to nearest lane centerline")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "lane_center_offset_boxplot.png"), dpi=200)
    plt.close(fig)


def make_boxplots():
    """Dispatch function: create enabled aggregate plots after processing all samples."""
    if PLOT_STOP_DISTANCE:
        make_stop_distance_boxplot()
    if PLOT_LANE_CENTER_OFFSET:
        make_lane_offset_boxplot()


def main():
    """Program entry point: load jobs, iterate TFRecords, export CSVs and plots.

    The loop processes each requested segment index from each TFRecord, renders a
    scene PNG (if enabled), and writes AV/HV CSV rows per frame per agent.
    After all scenes, it generates aggregate boxplots if requested.
    """
    jobs = load_jobs(JSON_PATH)
    if not jobs:
        print("JSON 为空或未匹配到 tfrecord；请检查 JSON_PATH/TFREC_DIR。"); return

    hv_writer = av_writer = None
    hv_f = av_f = None
    if EXPORT_HV_CSV:
        hv_f = open(os.path.join(OUT_DIR, "hv_trajectories.csv"), "w", newline="")
        import csv
        hv_writer = csv.DictWriter(hv_f, fieldnames=[
            "file","ex_idx","track_id","t","x_local","y_local","valid",
            "stop_x_local","stop_y_local","dist_to_stop","lane_center_offset"
        ]); hv_writer.writeheader()
    if EXPORT_AV_CSV:
        av_f = open(os.path.join(OUT_DIR, "av_trajectories.csv"), "w", newline="")
        import csv
        av_writer = csv.DictWriter(av_f, fieldnames=[
            "file","ex_idx","track_id","t","x_local","y_local","valid",
            "stop_x_local","stop_y_local","dist_to_stop","lane_center_offset"
        ]); av_writer.writeheader()

    for tfrec_path, indices in jobs:
        if not os.path.exists(tfrec_path):
            print("找不到文件：", tfrec_path); continue
        print("Reading:", os.path.basename(tfrec_path), "segments:", len(indices))
        ds = tf.data.TFRecordDataset(tfrec_path)
        recs = list(ds.as_numpy_iterator())
        for ex_idx in indices:
            if ex_idx < 0 or ex_idx >= len(recs):
                print(f"  - 跳过 idx={ex_idx} (越界，总 {len(recs)})"); continue
            ex = example_pb2.Example(); ex.ParseFromString(recs[ex_idx])
            base = os.path.basename(tfrec_path)
            png = draw_one_sample(ex, base, ex_idx, hv_writer=hv_writer, av_writer=av_writer)
            if png:
                print("  + saved:", os.path.basename(png))
            else:
                print("  + processed (PNG export disabled)")

    for f in (hv_f, av_f):
        if f: f.close()

    make_boxplots()
    print("✅ Done. CSV & plots saved in:", OUT_DIR)


if __name__ == "__main__":
    main()
