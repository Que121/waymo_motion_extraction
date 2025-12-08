# -*- coding: utf-8 -*-
import os, json, re, math, csv
from collections import defaultdict
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tensorflow.core.example import example_pb2

# ========== 配置 ==========
JSON_PATH = "./stop_radius_direction_consensus.json"
TFREC_DIR = "/mnt/home/Files/Lab/waymo_motion/dataset/"
OUT_DIR   = "./extract_out_stop_signs/"
EXPORT_STOP_SIGN_INTERACTIONS = False
STOP_SIGN_INTERACTION_JSON = os.path.join(OUT_DIR, "stop_sign_interactions.json")
EXPORT_STOP_RADIUS_DIRECTION_JSON = False
STOP_RADIUS_DIRECTION_JSON = os.path.join(OUT_DIR, "stop_radius_direction_consensus.json")
STOP_RADIUS_LANE_DIRECTION_TOL_DEG = 10.0
STOP_RADIUS_LANE_DIRECTION_TOL_RAD = math.radians(STOP_RADIUS_LANE_DIRECTION_TOL_DEG)
EXPORT_AV_MIN_SPEED_IN_RADIUS = True
AV_MIN_SPEED_CSV = os.path.join(OUT_DIR, "av_min_speed_in_radius.csv")
PLOT_AV_STOP_SPEED_DISTRIBUTION = True
STOP_SPEED_THRESHOLD_MPS = 0.5
AV_STOP_SPEED_PNG = os.path.join(OUT_DIR, "av_stop_speed_distribution.png")

USE_LOCAL_FRAME = True
ONLY_VEHICLES   = True
XLIM = (-120, 120)
YLIM = (-120, 120)

EXPORT_HV_CSV = True
EXPORT_AV_CSV = True
EXPORT_SCENE_PNG = True
EXPORT_STOP_RADIUS_TABLE = True
STOP_RADIUS_TABLE_CSV = os.path.join(OUT_DIR, "stop_radius_distances.csv")

# Metrics toggles
CALC_STOP_DISTANCE = True
PLOT_STOP_DISTANCE = True

# Lane center offset metrics
CALC_LANE_CENTER_OFFSET = False
PLOT_LANE_CENTER_OFFSET = False
PLOT_LANE_CENTER_OFFSET_OVERALL = False
PLOT_SCATTER_POINTS = False
PLOT_LANE_OFFSET_MEAN = False
PLOT_LANE_OFFSET_STD = False
PLOT_LANE_OFFSET_CASE_STATS = False
PLOT_AV_SCENARIO_OFFSETS = False

# 过滤配置
FILTER_TURNING = True
MAX_YAW_RATE = 0.08  # rad/frame (approx 4.5 degrees)
FILTER_LANE_CHANGE = True
MAX_LANE_ALIGNMENT_DIFF = 0.25 # rad (approx 14 degrees)
TURNING_PAD_FRAMES = 1
LANE_CHANGE_PAD_FRAMES = 1

# 方向箭头
DRAW_DIR_ARROWS = True
ARROW_LEN   = 6.0
ARROW_WIDTH = 0.006

COLOR_MAP = {
    "lanes": "#666666",
    "stops": "#d62728",
    "av":    "#ff7f0e",
    "hv":    "#1f77b4",
}

# WOMD TFExample shape
N_AGENTS = 128
N_PAST, N_CUR, N_FUT = 10, 1, 80  # 10Hz => 91 帧

# roadgraph types（涵盖 lane center / roadline / road edge）
LANE_TYPES      = {0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16}
STOP_SIGN_TYPES = {17}            # 你确认的定义

STOP_AS_SINGLE_POINT = True
STOP_POINT_REDUCER   = "mean"
STOP_DISTANCE_RADIUS = 5.0  # meters; None to disable radius filter
# ====================================

os.makedirs(OUT_DIR, exist_ok=True)
for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass

_TFREC_PAT = re.compile(r"\.tfrecords?($|-)", re.IGNORECASE)
is_tfrec = lambda n: _TFREC_PAT.search(os.path.basename(n)) is not None

def load_jobs(p):
    with open(p, "r") as f:
        items = json.load(f)
    jobs = []
    for it in items:
        fn = it["filename"]
        if not is_tfrec(fn): continue

        # 尝试在 TFREC_DIR 及其子目录中查找文件
        found_path = os.path.join(TFREC_DIR, fn)
        if not os.path.exists(found_path):
            for sub in ["training", "validation", "testing"]:
                candidate = os.path.join(TFREC_DIR, sub, fn)
                if os.path.exists(candidate):
                    found_path = candidate
                    break
        
        case_label = str(it.get("case", it.get("cases", it.get("scenario", "unspecified"))))
        jobs.append({
            "path": found_path,
            "indices": [int(x) for x in it.get("segment_indices", [])],
            "case": case_label
        })
    return jobs

def fval(ex, key, default=None):
    feat = ex.features.feature
    if key not in feat: return default
    fl = feat[key].float_list.value
    il = feat[key].int64_list.value
    if len(fl): return np.array(fl, dtype=np.float32)
    if len(il): return np.array(il, dtype=np.int64)
    return default

def reshape2(a, n, t, fill=np.nan, dtype=np.float32):
    if a is None: return np.full((n,t), fill, dtype=dtype)
    a = np.asarray(a); need = n*t
    if a.size < need: a = np.concatenate([a, np.full(need-a.size, fill, dtype=a.dtype)])
    return a.reshape(n,t)

def type_names_from_codes(codes):
    mp = {0:"UNSET", 1:"VEHICLE", 2:"PEDESTRIAN", 3:"CYCLIST"}
    if codes is None: codes = np.zeros(N_AGENTS, np.int64)
    return np.vectorize(lambda x: mp.get(int(x), "OTHER"))(codes)

def to_av_local(xs, ys, cur_x, cur_y, vx_c=None, vy_c=None, is_av_mask=None):
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
    X,Y = x-x0, y-y0
    c,s = np.cos(-yaw0), np.sin(-yaw0)
    return c*X - s*Y, s*X + c*Y

def vec_world_to_local(vx, vy, yaw0):
    c,s = np.cos(-yaw0), np.sin(-yaw0)
    return c*vx - s*vy, s*vx + c*vy

def group_roadgraph_points(xyz, ids):
    out = {}
    if xyz is None or ids is None or len(xyz)==0: return out
    xy = xyz.reshape(-1,3)[:, :2]; ids = ids.reshape(-1)
    for rg_id in np.unique(ids):
        pts = xy[ids == rg_id]
        if pts.shape[0] >= 2: out[int(rg_id)] = pts
    return out

def reduce_points_to_single(xy, reducer="mean"):
    if xy is None or len(xy)==0: return None
    return np.median(xy, axis=0) if reducer=="median" else xy.mean(axis=0)

# 全局累积（箱线图用）
AV_DIST_TO_STOP, HV_DIST_TO_STOP = [], []
AV_LANE_OFFSET, HV_LANE_OFFSET = [], []
AV_LANE_OFFSET_BY_SCENARIO = {}
AV_OFFSET_MEANS, HV_OFFSET_MEANS = [], []
AV_OFFSET_STDS, HV_OFFSET_STDS = [], []
AV_OFFSET_MEAN_BY_CASE = defaultdict(list)
HV_OFFSET_MEAN_BY_CASE = defaultdict(list)
AV_OFFSET_STD_BY_CASE = defaultdict(list)
HV_OFFSET_STD_BY_CASE = defaultdict(list)
STOP_SIGN_INTERACTION_ENTRIES = []
STOP_RADIUS_TABLE_ROWS = []
STOP_RADIUS_DIRECTION_SEGMENTS = defaultdict(set)
AV_MIN_SPEED_ROWS = []
AV_STOP_SPEED_SAMPLES = []
HV_STOP_SPEED_SAMPLES = []


def _closest_event(events):
    if not events:
        return None
    return min(events, key=lambda e: float(e.get("distance", math.inf)))


def _record_stop_speed_sample(is_av_flag, stop_evt):
    if not (PLOT_AV_STOP_SPEED_DISTRIBUTION and stop_evt):
        return
    speed = stop_evt.get("speed")
    if speed is None:
        return
    if STOP_SPEED_THRESHOLD_MPS is not None and speed > STOP_SPEED_THRESHOLD_MPS:
        return
    target = AV_STOP_SPEED_SAMPLES if is_av_flag else HV_STOP_SPEED_SAMPLES
    target.append(float(speed))



def update_stop_radius_table(filename, segment_index, case_label, av_stop_events, hv_stop_events):
    if not EXPORT_STOP_RADIUS_TABLE or STOP_DISTANCE_RADIUS is None:
        return
    for role, events in (("AV", av_stop_events), ("HV", hv_stop_events)):
        best = _closest_event(events)
        if best is None:
            continue
        STOP_RADIUS_TABLE_ROWS.append({
            "filename": filename,
            "segment_index": int(segment_index),
            "case": case_label,
            "role": role,
            "frame": int(best["frame"]),
            "distance": float(best["distance"]),
        })


def collect_stop_radius_direction_entries(
    filename,
    segment_index,
    case_label,
    stop_local,
    av_events,
    hv_events,
):
    if (
        not EXPORT_STOP_RADIUS_DIRECTION_JSON
        or STOP_DISTANCE_RADIUS is None
        or stop_local is None
    ):
        return

    frame_events = defaultdict(list)
    for event in av_events + hv_events:
        frame_events[int(event["frame"])].append(event)

    for frame, events in frame_events.items():
        if len(events) < 2:
            continue
        role_counts = defaultdict(int)
        ref_direction = None
        valid = True
        for evt in events:
            role_counts[evt["role"]] += 1
            direction_flag = evt.get("direction_flag")
            if direction_flag is None:
                valid = False
                break
            if ref_direction is None:
                ref_direction = direction_flag
                continue
            if not lane_directions_aligned(ref_direction, direction_flag):
                valid = False
                break
        if not valid:
            continue
        if role_counts["AV"] < 1 or role_counts["HV"] < 1:
            continue
        STOP_RADIUS_DIRECTION_SEGMENTS[filename].add(int(segment_index))
        break


def record_av_min_speed_event(filename, segment_index, case_label, event):
    if not EXPORT_AV_MIN_SPEED_IN_RADIUS or event is None:
        return
    AV_MIN_SPEED_ROWS.append({
        "filename": filename,
        "segment_index": int(segment_index),
        "case": case_label,
        "frame": int(event.get("frame", -1)),
        "speed_mps": float(event.get("speed", 0.0)),
        "distance_m": float(event.get("distance", math.nan)),
        "x_local": float(event.get("px", math.nan)),
        "y_local": float(event.get("py", math.nan)),
    })

def closest_point_on_polyline(polyline, px, py):
    """返回折线上离给定点最近的点、距离和单位切向量。"""
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


def lane_direction_flag(px, py, lane_segments_local):
    """返回包含 lane id 和单位切向的方向描述。"""
    if not lane_segments_local:
        return None
    best = None
    for lane_id, polyline in lane_segments_local.items():
        _, dist, tangent = closest_point_on_polyline(polyline, px, py)
        if dist is None or tangent is None:
            continue
        if best is None or dist < best[0]:
            best = (dist, tangent, lane_id)
    if best is None:
        return None
    tangent = _normalize_vector(best[1])
    if tangent is None:
        return None
    return {
        "lane_id": int(best[2]),
        "tangent": [float(tangent[0]), float(tangent[1])],
    }


def lane_directions_aligned(dir_a, dir_b, tolerance=STOP_RADIUS_LANE_DIRECTION_TOL_RAD):
    if dir_a is None or dir_b is None:
        return False
    if dir_a.get("lane_id") != dir_b.get("lane_id"):
        return False
    ta = np.asarray(dir_a.get("tangent", []), dtype=np.float32)
    tb = np.asarray(dir_b.get("tangent", []), dtype=np.float32)
    if ta.size != 2 or tb.size != 2:
        return False
    norm_a = float(np.linalg.norm(ta))
    norm_b = float(np.linalg.norm(tb))
    if norm_a < 1e-6 or norm_b < 1e-6:
        return False
    dot_val = np.clip(float(np.dot(ta, tb) / (norm_a * norm_b)), -1.0, 1.0)
    angle = math.acos(dot_val)
    return angle <= tolerance

def _normalize_vector(vec, eps=1e-3):
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return None
    return vec / norm

def _mark_with_padding(mask, center_idx, pad):
    if mask.size == 0:
        return
    start = max(0, center_idx - pad)
    end = min(mask.size - 1, center_idx + pad)
    mask[start:end+1] = True

def compute_signed_stop_distance(px, py, stop_point, forward_dir):
    """Return Euclidean distance to stop sign, signed by forward direction."""
    if stop_point is None:
        return None
    diff = np.array([px - float(stop_point[0]), py - float(stop_point[1])], dtype=np.float32)
    dist = float(np.hypot(diff[0], diff[1]))
    if forward_dir is None:
        return dist
    dot = float(np.dot(diff, forward_dir))
    return dist if dot >= 0.0 else -dist

def estimate_heading_at_index(xs_row, ys_row, vidx, idx_in_vidx, default_heading=None):
    """返回在 vidx[idx_in_vidx] 时刻的单位朝向向量。"""
    t = vidx[idx_in_vidx]
    # forward difference
    if idx_in_vidx + 1 < len(vidx):
        nxt = vidx[idx_in_vidx + 1]
        vec = np.array([xs_row[nxt] - xs_row[t], ys_row[nxt] - ys_row[t]], dtype=np.float32)
        normed = _normalize_vector(vec)
        if normed is not None:
            return normed
    # backward difference
    if idx_in_vidx > 0:
        prev = vidx[idx_in_vidx - 1]
        vec = np.array([xs_row[t] - xs_row[prev], ys_row[t] - ys_row[prev]], dtype=np.float32)
        normed = _normalize_vector(vec)
        if normed is not None:
            return normed
    # fall back to default heading if provided
    if default_heading is not None:
        normed = _normalize_vector(default_heading)
        if normed is not None:
            return normed
    # default facing positive x axis
    return np.array([1.0, 0.0], dtype=np.float32)


def estimate_speed_at_index(xs_row, ys_row, vidx, idx_in_vidx, dt_base=0.1):
    """基于相邻帧位置估算速度（m/s）。"""
    t = vidx[idx_in_vidx]
    def _speed_between(a_idx, b_idx):
        if a_idx == b_idx:
            return 0.0
        dx = float(xs_row[b_idx] - xs_row[a_idx])
        dy = float(ys_row[b_idx] - ys_row[a_idx])
        frame_gap = abs(b_idx - a_idx)
        dt = max(frame_gap, 1) * dt_base
        return math.hypot(dx, dy) / dt if dt > 1e-4 else 0.0

    if idx_in_vidx + 1 < len(vidx):
        nxt = vidx[idx_in_vidx + 1]
        return _speed_between(t, nxt)
    if idx_in_vidx > 0:
        prev = vidx[idx_in_vidx - 1]
        return _speed_between(prev, t)
    return 0.0

def compute_lane_offset_for_point(P, heading_vec, lane_segments_local):
    """返回 (offset, alignment_violation)。alignment_violation=True 表示应视为变道。"""
    if not lane_segments_local:
        return None, False
    left_candidate = None
    right_candidate = None
    for polyline in lane_segments_local.values():
        point, dist, tangent = closest_point_on_polyline(polyline, P[0], P[1])
        if point is None or dist is None:
            continue
        rel = point - P
        side = heading_vec[0] * rel[1] - heading_vec[1] * rel[0]
        candidate = (point, dist, tangent)
        if side >= 0:
            if left_candidate is None or dist < left_candidate[1]:
                left_candidate = candidate
        else:
            if right_candidate is None or dist < right_candidate[1]:
                right_candidate = candidate
    if not left_candidate or not right_candidate:
        return None, False
    if left_candidate[1] > 3.7 or right_candidate[1] > 3.7:
        return None, False
    center_point = 0.5 * (left_candidate[0] + right_candidate[0])
    tangent_vec = left_candidate[2] + right_candidate[2]
    normed_tangent = _normalize_vector(tangent_vec)
    if normed_tangent is None:
        normed_tangent = heading_vec
    
    # 过滤：检查车辆朝向与车道切向的夹角（筛除变道/错误匹配）
    if FILTER_LANE_CHANGE:
        dot_val = np.dot(heading_vec, normed_tangent)
        # dot_val 应该接近 1。计算夹角绝对值。
        angle_diff = np.arccos(np.clip(dot_val, -1.0, 1.0))
        if abs(angle_diff) > MAX_LANE_ALIGNMENT_DIFF:
            return None, True

    normal_vec = np.array([-normed_tangent[1], normed_tangent[0]], dtype=np.float32)
    return float(np.dot(P - center_point, normal_vec)), False

def draw_one_sample(ex, base_name, ex_idx, hv_writer=None, av_writer=None, case_label="unspecified"):
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

    forward_dir = None
    if np.any(is_av):
        if USE_LOCAL_FRAME:
            forward_dir = np.array([1.0, 0.0], dtype=np.float32)
        else:
            av_idx = np.where(is_av)[0][0]
            heading_vec = np.array([vx_cur[av_idx], vy_cur[av_idx]], dtype=np.float32)
            normed = _normalize_vector(heading_vec)
            if normed is not None:
                forward_dir = normed.astype(np.float32)

    # 2) Roadgraph
    rg_xyz = fval(ex,"roadgraph_samples/xyz")
    rg_id  = fval(ex,"roadgraph_samples/id")
    rg_t   = fval(ex,"roadgraph_samples/type")
    rg_valid = fval(ex,"roadgraph_samples/valid")
    if rg_xyz is not None: rg_xyz = rg_xyz.reshape(-1,3)
    if rg_valid is not None:
        m = rg_valid.astype(bool)
        if rg_xyz is not None: rg_xyz = rg_xyz[m]
        if rg_id  is not None: rg_id  = rg_id[m]
        if rg_t   is not None: rg_t   = rg_t[m]

    lane_segments = {}
    lane_type_map = {}
    stop_pts_world = np.empty((0,2),dtype=np.float32)
    if rg_t is not None and rg_xyz is not None:
        t = rg_t.reshape(-1)
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
        stop_mask = np.isin(t, list(STOP_SIGN_TYPES))
        if np.any(stop_mask):
            stop_pts_world = rg_xyz[stop_mask][:,:2]

    # stop 单点（局部，离 AV 最近）
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

    # lane 点云（局部）
    lane_segments_local = {}
    if len(lane_segments)>0:
        for lid, pts in lane_segments.items():
            Xw,Yw = pts[:,0], pts[:,1]
            X,Y = world_to_local_xy(Xw,Yw,x0,y0,yaw0) if np.any(is_av) else (Xw,Yw)
            stack = np.stack([X,Y],axis=1)
            lane_segments_local[lid] = stack

    # 3) 绘图
    fig = ax = None
    if EXPORT_SCENE_PNG:
        fig, ax = plt.subplots(figsize=(6,6))
        for pts in lane_segments.values():
            Xw,Yw = pts[:,0], pts[:,1]
            X,Y = world_to_local_xy(Xw,Yw,x0,y0,yaw0) if np.any(is_av) else (Xw,Yw)
            ax.plot(X,Y,color=COLOR_MAP["lanes"],linewidth=0.6,alpha=0.8,zorder=0)
        if stop_local is not None:
            ax.scatter([stop_local[0]],[stop_local[1]], s=36, c=COLOR_MAP["stops"], marker="o", zorder=6)
            if STOP_DISTANCE_RADIUS is not None:
                circle = plt.Circle(
                    (stop_local[0], stop_local[1]),
                    STOP_DISTANCE_RADIUS,
                    edgecolor=COLOR_MAP["stops"],
                    facecolor="none",
                    linewidth=1.2,
                    linestyle="--",
                    alpha=0.7,
                    zorder=5,
                )
                ax.add_patch(circle)

    # 4) 逐 agent 写 CSV（仅记录最终停下时的 dist_to_stop & lane_center_offset），并画轨迹/箭头
    scenario_key = f"{base_name}_ex{ex_idx:05d}"

    av_events = []
    hv_events = []
    av_min_speed_event = None
    av_stop_events = []
    hv_stop_events = []

    def process_agent(writer, is_av_flag, i, vidx, scenario_key, forward_dir):
        default_heading = np.array([vx_loc[i], vy_loc[i]], dtype=np.float32)
        if _normalize_vector(default_heading) is None:
            default_heading = None
        stop_coords = (float(stop_local[0]), float(stop_local[1])) if stop_local is not None else ("", "")

        needs_stop_metrics = (
            CALC_STOP_DISTANCE
            or EXPORT_STOP_SIGN_INTERACTIONS
            or EXPORT_STOP_RADIUS_TABLE
            or EXPORT_STOP_RADIUS_DIRECTION_JSON
        )

        agent_offsets = []
        inside_radius_events = []
        stop_event = None

        heading_cache = [
            estimate_heading_at_index(xs_loc[i], ys_loc[i], vidx, idx_pos, default_heading)
            for idx_pos in range(len(vidx))
        ]
        speed_cache = [
            estimate_speed_at_index(xs_loc[i], ys_loc[i], vidx, idx_pos)
            for idx_pos in range(len(vidx))
        ]
        turning_mask = np.zeros(len(vidx), dtype=bool)
        if CALC_LANE_CENTER_OFFSET and FILTER_TURNING and len(vidx) >= 2:
            for idx_pos in range(1, len(vidx)):
                h_prev = heading_cache[idx_pos - 1]
                h_cur = heading_cache[idx_pos]
                if h_prev is None or h_cur is None:
                    continue
                dot_yaw = np.dot(h_cur, h_prev)
                yaw_change = np.arccos(np.clip(dot_yaw, -1.0, 1.0))
                if yaw_change > MAX_YAW_RATE:
                    _mark_with_padding(turning_mask, idx_pos - 1, TURNING_PAD_FRAMES)
                    _mark_with_padding(turning_mask, idx_pos, TURNING_PAD_FRAMES)
        lane_change_mask = np.zeros(len(vidx), dtype=bool)

        for idx_pos, t_idx in enumerate(vidx):
            t_idx = int(t_idx)
            px = float(xs_loc[i, t_idx]); py = float(ys_loc[i, t_idx])
            heading_vec = heading_cache[idx_pos] if idx_pos < len(heading_cache) else None

            d_stop = ""
            sx = sy = ""
            signed_dist = None
            within_radius = False
            if needs_stop_metrics and stop_local is not None:
                signed_dist = compute_signed_stop_distance(px, py, stop_local, forward_dir)
                if signed_dist is not None and (
                    STOP_DISTANCE_RADIUS is None or abs(signed_dist) <= STOP_DISTANCE_RADIUS
                ):
                    within_radius = True
                    if CALC_STOP_DISTANCE:
                        d_stop = signed_dist
                        sx, sy = stop_coords
                    direction_flag = lane_direction_flag(px, py, lane_segments_local)
                    inside_radius_events.append({
                        "frame": t_idx,
                        "signed_dist": float(signed_dist),
                        "distance": float(abs(signed_dist)),
                        "px": px,
                        "py": py,
                        "track_id": int(track_id[i]),
                        "role": "AV" if is_av_flag else "HV",
                        "heading": (heading_vec.tolist() if heading_vec is not None else None),
                        "direction_flag": direction_flag,
                        "speed": float(speed_cache[idx_pos]) if idx_pos < len(speed_cache) else None,
                    })

            lane_offset_val = None

            if CALC_LANE_CENTER_OFFSET and lane_segments_local:
                lane_heading = heading_vec
                if FILTER_TURNING and turning_mask[idx_pos]:
                    lane_heading = None
                else:
                    if lane_heading is None:
                        continue
                    P = np.array([px, py], dtype=np.float32)
                    lane_offset_val, alignment_violation = compute_lane_offset_for_point(
                        P, lane_heading, lane_segments_local
                    )
                    if alignment_violation:
                        _mark_with_padding(lane_change_mask, idx_pos, LANE_CHANGE_PAD_FRAMES)
                        continue
                    if lane_change_mask[idx_pos]:
                        continue
                    if lane_offset_val is not None:
                        agent_offsets.append(lane_offset_val)
                        if is_av_flag:
                            AV_LANE_OFFSET.append(lane_offset_val)
                            AV_LANE_OFFSET_BY_SCENARIO.setdefault(scenario_key, []).append(lane_offset_val)
                        else:
                            HV_LANE_OFFSET.append(lane_offset_val)
            
            if writer is not None:
                writer.writerow({
                    "file": base_name, "ex_idx": ex_idx, "track_id": int(track_id[i]),
                    "t": t_idx,
                    "x_local": px, "y_local": py, "valid": int(valids[i, t_idx]),
                    "stop_x_local": sx, "stop_y_local": sy, "dist_to_stop": d_stop,
                    "lane_center_offset": ("" if lane_offset_val is None else lane_offset_val),
                })

        # 只要该车辆还有剩余的有效帧（未被过滤），就会计算均值/标准差并计入统计
        # 并不是因为某一帧被过滤就丢弃整个车辆数据
        if agent_offsets:
            arr = np.array(agent_offsets, dtype=np.float32)
            m_val = float(np.mean(arr))
            std_val = float(np.std(arr))
            label = case_label or "unspecified"
            if is_av_flag:
                AV_OFFSET_MEANS.append(m_val)
                AV_OFFSET_STDS.append(std_val)
                AV_OFFSET_MEAN_BY_CASE[label].append(m_val)
                AV_OFFSET_STD_BY_CASE[label].append(std_val)
            else:
                HV_OFFSET_MEANS.append(m_val)
                HV_OFFSET_STDS.append(std_val)
                HV_OFFSET_MEAN_BY_CASE[label].append(m_val)
                HV_OFFSET_STD_BY_CASE[label].append(std_val)
        if CALC_STOP_DISTANCE and inside_radius_events:
            stop_candidates = [evt for evt in inside_radius_events if evt.get("speed") is not None]
            if stop_candidates:
                min_speed = min(evt["speed"] for evt in stop_candidates)
                for evt in stop_candidates:
                    if abs(evt["speed"] - min_speed) < 1e-6:
                        stop_event = evt
                        break
            else:
                stop_event = inside_radius_events[0]
            if stop_event is not None:
                signed_dist = float(stop_event.get("signed_dist", 0.0))
                dist_val = abs(signed_dist)
                if is_av_flag:
                    AV_DIST_TO_STOP.append(dist_val)
                else:
                    HV_DIST_TO_STOP.append(dist_val)

        return inside_radius_events, stop_event

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
        
        if is_av_i:
            events, stop_evt = process_agent(av_writer, True, i, vidx, scenario_key, forward_dir)
            av_events.extend(events)
            if stop_evt is not None:
                av_stop_events.append(stop_evt)
                _record_stop_speed_sample(True, stop_evt)
        else:
            events, stop_evt = process_agent(hv_writer, False, i, vidx, scenario_key, forward_dir)
            hv_events.extend(events)
            if stop_evt is not None:
                hv_stop_events.append(stop_evt)
                _record_stop_speed_sample(False, stop_evt)

    if EXPORT_AV_MIN_SPEED_IN_RADIUS and av_stop_events:
        speed_candidates = [evt for evt in av_stop_events if evt.get("speed") is not None]
        target_list = speed_candidates if speed_candidates else av_stop_events
        av_min_speed_event = min(target_list, key=lambda e: e.get("speed", math.inf)) if target_list else None
        record_av_min_speed_event(base_name, ex_idx, case_label, av_min_speed_event)

    if EXPORT_SCENE_PNG:
        ax.set_aspect("equal",adjustable="box")
        if EXPORT_AV_MIN_SPEED_IN_RADIUS and av_min_speed_event is not None:
            ax.scatter(
                [av_min_speed_event["px"]],
                [av_min_speed_event["py"]],
                s=42,
                c="#2ca02c",
                edgecolors="black",
                linewidths=0.6,
                marker="o",
                zorder=14,
                label="AV min speed",
            )
            ax.annotate(
                f"{av_min_speed_event['speed']:.2f} m/s",
                (av_min_speed_event["px"], av_min_speed_event["py"]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,
                color="#2ca02c",
                weight="bold",
            )
        ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{base_name} ex{ex_idx} (AV/HV + stop sign + directions)", fontsize=10)
        legend = [
            Line2D([0],[0], color=COLOR_MAP["lanes"], lw=1.2, label="lane/roadgraph"),
            Line2D([0],[0], color=COLOR_MAP["av"], lw=2.4, label="AV trajectory"),
            Line2D([0],[0], color=COLOR_MAP["hv"], lw=1.2, label="HV trajectories"),
            Line2D([0],[0], marker='o', color='w', markerfacecolor=COLOR_MAP["stops"], markersize=6, label="stop sign"),
            Line2D([0],[0], color='k', lw=0, marker=r'$\rightarrow$', label="direction"),
        ]
        if EXPORT_AV_MIN_SPEED_IN_RADIUS and av_min_speed_event is not None:
            legend.append(
                Line2D(
                    [0],
                    [0],
                    marker='o',
                    color='w',
                    markerfacecolor="#2ca02c",
                    markeredgecolor="black",
                    markersize=6,
                    label="AV min speed",
                )
            )
        ax.legend(handles=legend, loc="upper right", fontsize=8, frameon=True)
        out_png = os.path.join(OUT_DIR, f"{base_name}_ex{ex_idx:05d}.png")
        fig.tight_layout(); fig.savefig(out_png, dpi=200)
        plt.close(fig)
        return out_png

    if EXPORT_STOP_SIGN_INTERACTIONS and stop_local is not None and av_events and hv_events:
        for av_event in av_events:
            for hv_event in hv_events:
                if av_event["frame"] != hv_event["frame"]:
                    continue
                diff = av_event["signed_dist"] - hv_event["signed_dist"]
                if abs(diff) < 1e-3:
                    continue
                front_role = "AV" if diff > 0 else "HV"
                back_role = "HV" if front_role == "AV" else "AV"
                entry = {
                    "filename": base_name,
                    "segment_index": int(ex_idx),
                    "case": case_label,
                    "frame": int(av_event["frame"]),
                    "order": f"{front_role}_front_{back_role}_back",
                    "stop_radius_m": STOP_DISTANCE_RADIUS,
                    "stop_local_xy": [float(stop_local[0]), float(stop_local[1])],
                    "av": {
                        "track_id": av_event["track_id"],
                        "signed_distance": av_event["signed_dist"],
                        "distance": av_event["distance"],
                        "position_xy": [av_event["px"], av_event["py"]],
                    },
                    "hv": {
                        "track_id": hv_event["track_id"],
                        "signed_distance": hv_event["signed_dist"],
                        "distance": hv_event["distance"],
                        "position_xy": [hv_event["px"], hv_event["py"]],
                    },
                }
                STOP_SIGN_INTERACTION_ENTRIES.append(entry)
                break
            else:
                continue
            break

    if STOP_DISTANCE_RADIUS is not None and stop_local is not None:
        update_stop_radius_table(base_name, ex_idx, case_label, av_stop_events, hv_stop_events)
        collect_stop_radius_direction_entries(
            base_name,
            ex_idx,
            case_label,
            stop_local,
            av_events,
            hv_events,
        )

    return None

def hv_speed():
    pass

def hv_acceleration():
    pass

def _plot_av_hv_box(av_vals, hv_vals, title, filename, scatter_seed=42, draw_zero_line=False):
    if (not av_vals) and (not hv_vals):
        return
    plt.figure(figsize=(5,5))
    data = []
    labels = []
    colors = []
    if av_vals:
        data.append(av_vals)
        labels.append("AV")
        colors.append(COLOR_MAP["av"])
    if hv_vals:
        data.append(hv_vals)
        labels.append("HV")
        colors.append(COLOR_MAP["hv"])
    if not data:
        plt.close()
        return
    bp = plt.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.35)
    for element in ("medians", "caps", "whiskers"):
        for artist in bp[element]:
            artist.set_color("#444444")
            artist.set_linewidth(1.1)

    if PLOT_SCATTER_POINTS:
        rng = np.random.default_rng(scatter_seed)
        for idx, vals in enumerate(data):
            if not vals:
                continue
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            plt.scatter(
                np.full(len(vals), idx + 1, dtype=np.float32) + jitter,
                vals,
                color=colors[idx],
                alpha=0.55,
                s=12,
            )

    if draw_zero_line:
        plt.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    plt.xlabel("AV / HV")
    plt.ylabel("distance")
    plt.title(title)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, filename), dpi=200)
    plt.close()

def make_stop_distance_boxplot():
    if not PLOT_STOP_DISTANCE:
        return
    if len(AV_DIST_TO_STOP)+len(HV_DIST_TO_STOP) == 0:
        return
    _plot_av_hv_box(
        AV_DIST_TO_STOP,
        HV_DIST_TO_STOP,
        title="Distance to stop sign (final frame)",
        filename="dist_to_stop_boxplot.png",
    )

def make_lane_offset_boxplot():
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
        labels.append("HV all")
        colors.append(COLOR_MAP["hv"])
        positions.append(pos)
        pos += 1

    if AV_LANE_OFFSET:
        plot_data.append(AV_LANE_OFFSET)
        labels.append("AV all")
        colors.append(COLOR_MAP["av"])
        positions.append(pos)

    if not plot_data:
        return

    width = max(6.0, 1.1 * len(plot_data))
    fig, ax = plt.subplots(figsize=(width, 5))
    ax.boxplot(plot_data, positions=positions, labels=labels, showfliers=False)

    if PLOT_SCATTER_POINTS:
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

    ax.set_ylabel("distance")
    ax.set_title("Signed distance to nearest lane centerline")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "lane_center_offset_boxplot.png"), dpi=200)
    plt.close(fig)

def make_av_scenario_lane_offset_boxplot():
    if not PLOT_AV_SCENARIO_OFFSETS:
        return
    scenario_items = sorted(
        ((key, vals) for key, vals in AV_LANE_OFFSET_BY_SCENARIO.items() if vals),
        key=lambda x: x[0],
    )
    if not scenario_items:
        return

    labels = [key for key, _ in scenario_items]
    data = [vals for _, vals in scenario_items]
    positions = list(range(1, len(data)+1))
    fig_width = max(8.0, 0.6 * len(data))
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    bp = ax.boxplot(data, positions=positions, labels=labels, showfliers=False, patch_artist=True)

    for patch in bp["boxes"]:
        patch.set_facecolor(COLOR_MAP["av"])
        patch.set_edgecolor(COLOR_MAP["av"])
        patch.set_alpha(0.3)
    for element in ("medians", "caps", "whiskers"):
        for artist in bp[element]:
            artist.set_color("#444444")
            artist.set_linewidth(1.0)

    if PLOT_SCATTER_POINTS:
        rng = np.random.default_rng(101)
        for vals, xpos in zip(data, positions):
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(
                np.full(len(vals), xpos, dtype=np.float32) + jitter,
                vals,
                color=COLOR_MAP["av"],
                alpha=0.45,
                s=10,
            )

    ax.set_ylabel("distance")
    ax.set_xlabel("Scenario (tfrecord)")
    ax.set_title("AV lane center offset by scenario")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "av_lane_offset_per_scenario.png"), dpi=200)
    plt.close(fig)

def make_lane_offset_overall_boxplot():
    if not PLOT_LANE_CENTER_OFFSET_OVERALL:
        return
    if len(AV_LANE_OFFSET)+len(HV_LANE_OFFSET) == 0:
        return
    _plot_av_hv_box(
        AV_LANE_OFFSET,
        HV_LANE_OFFSET,
        title="Aggregate lane center offset (per-frame)",
        filename="lane_center_offset_overall_boxplot.png",
        scatter_seed=7,
        draw_zero_line=True,
    )

def make_lane_offset_mean_boxplot():
    if not PLOT_LANE_OFFSET_MEAN:
        return
    if len(AV_OFFSET_MEANS)+len(HV_OFFSET_MEANS) == 0:
        return
    _plot_av_hv_box(
        AV_OFFSET_MEANS,
        HV_OFFSET_MEANS,
        title="Mean of Lane Center Offset per Vehicle",
        filename="lane_offset_mean_boxplot.png",
        scatter_seed=42,
        draw_zero_line=True,
    )

def make_lane_offset_std_boxplot():
    if not PLOT_LANE_OFFSET_STD:
        return
    if len(AV_OFFSET_STDS)+len(HV_OFFSET_STDS) == 0:
        return
    _plot_av_hv_box(
        AV_OFFSET_STDS,
        HV_OFFSET_STDS,
        title="Std. of Lane Center Offset per Vehicle",
        filename="lane_offset_std_boxplot.png",
        scatter_seed=52,
        draw_zero_line=False,
    )

def _prepare_case_boxplot_payload(cases, av_dict, hv_dict):
    data = []
    positions = []
    colors = []
    xticks = []
    xticklabels = []
    pos = 1
    for case in cases:
        start_pos = pos
        added = False
        av_vals = av_dict.get(case, [])
        hv_vals = hv_dict.get(case, [])
        if av_vals:
            data.append(av_vals)
            positions.append(pos)
            colors.append(COLOR_MAP["av"])
            pos += 1
            added = True
        if hv_vals:
            data.append(hv_vals)
            positions.append(pos)
            colors.append(COLOR_MAP["hv"])
            pos += 1
            added = True
        if added:
            end_pos = pos - 1
            xticks.append(0.5 * (start_pos + end_pos))
            xticklabels.append(case)
    return data, positions, colors, xticks, xticklabels

def _plot_case_box(ax, cases, av_dict, hv_dict, ylabel, title, draw_zero_line=False):
    data, positions, colors, xticks, xticklabels = _prepare_case_boxplot_payload(cases, av_dict, hv_dict)
    if not data:
        ax.set_visible(False)
        return False
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        showfliers=False,
        patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.35)
    for element in ("medians", "caps", "whiskers"):
        for artist in bp[element]:
            artist.set_color("#444444")
            artist.set_linewidth(1.1)

    if PLOT_SCATTER_POINTS:
        rng = np.random.default_rng(123)
        for idx, vals in enumerate(data):
            if not vals:
                continue
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(
                np.full(len(vals), positions[idx]) + jitter,
                vals,
                s=10,
                alpha=0.45,
                color=colors[idx],
            )

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if draw_zero_line:
        ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(True, axis="y", linestyle=":", alpha=0.3)
    return True

def make_lane_offset_case_stats_plot():
    if not PLOT_LANE_OFFSET_CASE_STATS:
        return
    case_keys = set(AV_OFFSET_MEAN_BY_CASE.keys()) | set(HV_OFFSET_MEAN_BY_CASE.keys())
    case_keys |= set(AV_OFFSET_STD_BY_CASE.keys()) | set(HV_OFFSET_STD_BY_CASE.keys())
    cases = sorted(case_keys)
    if not cases:
        return

    fig_width = max(10.0, 2.5 * len(cases))
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 5), sharey=False)
    mean_ax, std_ax = axes

    mean_plotted = _plot_case_box(
        mean_ax,
        cases,
        AV_OFFSET_MEAN_BY_CASE,
        HV_OFFSET_MEAN_BY_CASE,
        ylabel="distance",
        title="Lane-center mean offset by case",
        draw_zero_line=True,
    )
    std_plotted = _plot_case_box(
        std_ax,
        cases,
        AV_OFFSET_STD_BY_CASE,
        HV_OFFSET_STD_BY_CASE,
        ylabel="distance",
        title="Lane-center std dev by case",
        draw_zero_line=False,
    )

    handles = [
        Line2D([0], [0], color=COLOR_MAP["av"], lw=6, alpha=0.35, label="AV"),
        Line2D([0], [0], color=COLOR_MAP["hv"], lw=6, alpha=0.35, label="HV"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)

    if not (mean_plotted or std_plotted):
        plt.close(fig)
        return

    fig.tight_layout(rect=(0.02, 0.0, 0.98, 0.95))
    fig.savefig(os.path.join(OUT_DIR, "lane_offset_case_stats.png"), dpi=200)
    plt.close(fig)

def export_stop_sign_interaction_json():
    if not EXPORT_STOP_SIGN_INTERACTIONS:
        return
    payload = {
        "radius_m": STOP_DISTANCE_RADIUS,
        "items": STOP_SIGN_INTERACTION_ENTRIES,
    }
    os.makedirs(os.path.dirname(STOP_SIGN_INTERACTION_JSON) or ".", exist_ok=True)
    with open(STOP_SIGN_INTERACTION_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("  ↳ stop-sign interactions written:", STOP_SIGN_INTERACTION_JSON)


def export_stop_radius_table():
    if not EXPORT_STOP_RADIUS_TABLE:
        return
    os.makedirs(os.path.dirname(STOP_RADIUS_TABLE_CSV) or ".", exist_ok=True)
    with open(STOP_RADIUS_TABLE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "segment_index", "case", "role", "frame", "distance"],
        )
        writer.writeheader()
        writer.writerows(STOP_RADIUS_TABLE_ROWS)
    if STOP_RADIUS_TABLE_ROWS:
        print("  ↳ stop-radius distance table written:", STOP_RADIUS_TABLE_CSV)
    else:
        print("  ↳ stop-radius table written (empty):", STOP_RADIUS_TABLE_CSV)


def export_av_min_speed_rows():
    if not EXPORT_AV_MIN_SPEED_IN_RADIUS:
        return
    os.makedirs(os.path.dirname(AV_MIN_SPEED_CSV) or ".", exist_ok=True)
    with open(AV_MIN_SPEED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "segment_index",
                "case",
                "frame",
                "speed_mps",
                "distance_m",
                "x_local",
                "y_local",
            ],
        )
        writer.writeheader()
        writer.writerows(AV_MIN_SPEED_ROWS)
    msg = "  ↳ AV min-speed rows written" if AV_MIN_SPEED_ROWS else "  ↳ AV min-speed rows written (empty)"
    print(msg + ":", AV_MIN_SPEED_CSV)


def export_stop_radius_direction_json():
    if not EXPORT_STOP_RADIUS_DIRECTION_JSON:
        return
    items = [
        {"filename": filename, "segment_indices": sorted(indices)}
        for filename, indices in sorted(STOP_RADIUS_DIRECTION_SEGMENTS.items())
        if indices
    ]
    payload = {
        "radius_m": STOP_DISTANCE_RADIUS,
        "items": items,
    }
    os.makedirs(os.path.dirname(STOP_RADIUS_DIRECTION_JSON) or ".", exist_ok=True)
    with open(STOP_RADIUS_DIRECTION_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(
        "  ↳ stop-radius consensus cases written:",
        STOP_RADIUS_DIRECTION_JSON,
        f"(items={len(items)})",
    )


def make_boxplots():
    if PLOT_STOP_DISTANCE:
        make_stop_distance_boxplot()
    if PLOT_LANE_CENTER_OFFSET:
        make_lane_offset_boxplot()
        make_av_scenario_lane_offset_boxplot()
    if PLOT_LANE_CENTER_OFFSET_OVERALL:
        make_lane_offset_overall_boxplot()
    if PLOT_LANE_OFFSET_MEAN:
        make_lane_offset_mean_boxplot()
    if PLOT_LANE_OFFSET_STD:
        make_lane_offset_std_boxplot()
    if PLOT_LANE_OFFSET_CASE_STATS:
        make_lane_offset_case_stats_plot()


def plot_av_stop_speed_distribution():
    if not PLOT_AV_STOP_SPEED_DISTRIBUTION:
        return
    if not AV_STOP_SPEED_SAMPLES and not HV_STOP_SPEED_SAMPLES:
        return
    threshold = STOP_SPEED_THRESHOLD_MPS if STOP_SPEED_THRESHOLD_MPS is not None else 0.0
    max_speed = 0.0
    if AV_STOP_SPEED_SAMPLES:
        max_speed = max(max_speed, max(AV_STOP_SPEED_SAMPLES))
    if HV_STOP_SPEED_SAMPLES:
        max_speed = max(max_speed, max(HV_STOP_SPEED_SAMPLES))
    upper = max(threshold, max_speed + 0.1)
    bins = np.linspace(0.0, max(upper, 0.1), 20)
    plt.figure(figsize=(5, 4))
    plotted = False
    if AV_STOP_SPEED_SAMPLES:
        plt.hist(
            AV_STOP_SPEED_SAMPLES,
            bins=bins,
            color="#2ca02c",
            alpha=0.65,
            edgecolor="black",
            label="AV",
        )
        plotted = True
    if HV_STOP_SPEED_SAMPLES:
        plt.hist(
            HV_STOP_SPEED_SAMPLES,
            bins=bins,
            color="#1f77b4",
            alpha=0.45,
            edgecolor="black",
            label="HV",
        )
        plotted = True
    if STOP_SPEED_THRESHOLD_MPS is not None:
        plt.axvline(
            STOP_SPEED_THRESHOLD_MPS,
            color="#d62728",
            linestyle="--",
            linewidth=1.2,
            label="threshold",
        )
    plt.xlabel("speed (m/s)")
    plt.ylabel("count")
    plt.title("Stop speeds per vehicle (first min)")
    if plotted:
        plt.legend()
    plt.tight_layout()
    plt.savefig(AV_STOP_SPEED_PNG, dpi=200)
    plt.close()

def main():
    jobs = load_jobs(JSON_PATH)
    if not jobs:
        print("JSON 为空或未匹配到 tfrecord；请检查 JSON_PATH/TFREC_DIR。"); return

    hv_writer = av_writer = None
    hv_f = av_f = None
    if EXPORT_HV_CSV:
        hv_f = open(os.path.join(OUT_DIR, "hv_trajectories.csv"), "w", newline="")
        hv_writer = csv.DictWriter(hv_f, fieldnames=[
            "file","ex_idx","track_id","t","x_local","y_local","valid",
            "stop_x_local","stop_y_local","dist_to_stop","lane_center_offset"
        ]); hv_writer.writeheader()
    if EXPORT_AV_CSV:
        av_f = open(os.path.join(OUT_DIR, "av_trajectories.csv"), "w", newline="")
        av_writer = csv.DictWriter(av_f, fieldnames=[
            "file","ex_idx","track_id","t","x_local","y_local","valid",
            "stop_x_local","stop_y_local","dist_to_stop","lane_center_offset"
        ]); av_writer.writeheader()

    for job in jobs:
        tfrec_path = job["path"]
        indices = job.get("indices", [])
        case_label = job.get("case", "unspecified")
        if not os.path.exists(tfrec_path):
            print("找不到文件：", tfrec_path); continue
        print("Reading:", os.path.basename(tfrec_path), "segments:", len(indices), "indices:", indices, "case:", case_label)
        ds = tf.data.TFRecordDataset(tfrec_path)
        recs = list(ds.as_numpy_iterator())
        for ex_idx in indices:
            if ex_idx < 0 or ex_idx >= len(recs):
                print(f"  - 跳过 idx={ex_idx} (越界，总 {len(recs)})"); continue
            ex = example_pb2.Example(); ex.ParseFromString(recs[ex_idx])
            base = os.path.basename(tfrec_path)
            png = draw_one_sample(
                ex,
                base,
                ex_idx,
                hv_writer=hv_writer,
                av_writer=av_writer,
                case_label=case_label,
            )
            if png:
                print("  + saved:", os.path.basename(png))
            else:
                print("  + processed (PNG export disabled)")

    for f in (hv_f, av_f):
        if f: f.close()

    make_boxplots()
    plot_av_stop_speed_distribution()
    export_stop_sign_interaction_json()
    export_stop_radius_table()
    export_av_min_speed_rows()
    export_stop_radius_direction_json()
    print("✅ Done. CSV & plots saved in:", OUT_DIR)

if __name__ == "__main__":
    main()
