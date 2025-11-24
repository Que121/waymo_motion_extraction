# -*- coding: utf-8 -*-
import os, json, re, math
from collections import defaultdict
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tensorflow.core.example import example_pb2

# ========== Configuration ==========
# Path to the JSON file containing the list of scenarios to process
JSON_PATH = "./summary.json"
# Directory containing the Waymo Open Motion Dataset (WOMD) TFRecord files
TFREC_DIR = "/mnt/home/Files/Lab/waymo_motion/dataset/"
# Directory where results (CSV, PNGs) will be saved
OUT_DIR   = "./extract_out"

# Coordinate System Settings
USE_LOCAL_FRAME = True  # If True, transforms all coordinates to be relative to the AV's starting position
ONLY_VEHICLES   = True  # If True, only processes agents classified as 'VEHICLE'
XLIM = (-120, 120)      # X-axis limits for the scene visualization (meters)
YLIM = (-120, 120)      # Y-axis limits for the scene visualization (meters)

# Export Toggles
EXPORT_HV_CSV = False      # Export CSV data for Human-Driven Vehicles (HV)
EXPORT_AV_CSV = False      # Export CSV data for the Autonomous Vehicle (AV)
EXPORT_SCENE_PNG = False   # Generate a top-down PNG image for every processed scenario

# Metrics Toggles (Enable/Disable specific calculations and plots)
CALC_STOP_DISTANCE = False          # Calculate distance to the nearest stop sign
PLOT_STOP_DISTANCE = False          # Plot stop sign distance statistics
CALC_LANE_CENTER_OFFSET = True      # Calculate lateral deviation from lane center
PLOT_LANE_CENTER_OFFSET = True      # Plot basic lane offset boxplots
PLOT_LANE_CENTER_OFFSET_OVERALL = True # Plot aggregate lane offset for all agents combined
PLOT_SCATTER_POINTS = False         # Overlay scatter points on boxplots (jittered)
PLOT_LANE_OFFSET_MEAN = True        # Plot mean offset per agent
PLOT_LANE_OFFSET_STD = True         # Plot standard deviation of offset per agent
PLOT_LANE_OFFSET_CASE_STATS = True  # Plot stats grouped by specific cases (from JSON)
PLOT_AV_SCENARIO_OFFSETS = True     # Plot AV specific offsets per scenario file

# Filtering Configuration (Rejection logic for valid "lane keeping" behavior)
FILTER_TURNING = True               # Exclude frames where the vehicle is turning
MAX_YAW_RATE = 0.08                 # Threshold: Rad/frame (approx 4.5 deg). Higher = turning
FILTER_LANE_CHANGE = True           # Exclude frames where the vehicle is changing lanes
MAX_LANE_ALIGNMENT_DIFF = 0.25      # Threshold: Radians (approx 14 deg). Angle diff between heading and lane
TURNING_PAD_FRAMES = 1              # Number of buffer frames to exclude before/after a turn
LANE_CHANGE_PAD_FRAMES = 1          # Number of buffer frames to exclude before/after a lane change

# Visualization settings for per-scene images
DRAW_DIR_ARROWS = True              # Draw velocity direction arrows on agents
ARROW_LEN   = 6.0                   # Visual length of the arrow
ARROW_WIDTH = 0.006                 # Visual width of the arrow

# Color palette for plotting
COLOR_MAP = {
    "lanes": "#666666",  # Grey for road lines
    "stops": "#d62728",  # Red for stop signs
    "av":    "#ff7f0e",  # Orange for Autonomous Vehicle
    "hv":    "#1f77b4",  # Blue for Human Vehicles
}

# WOMD Tensor Layout Constants
N_AGENTS = 128                       # Max agents per scene in WOMD
N_PAST, N_CUR, N_FUT = 10, 1, 80     # History (1s), Current, Future (8s). Total 9.1s @ 10Hz

# Road Graph ID definitions
# These IDs correspond to road boundaries, lane centers, crosswalks, etc.
LANE_TYPES      = {0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16}
STOP_SIGN_TYPES = {17}               # ID 17 is specifically for Stop Signs

# Stop Sign Processing
STOP_AS_SINGLE_POINT = True          # Reduce a stop line (polyline) to a single coordinate
STOP_POINT_REDUCER   = "mean"        # Method to reduce points: 'mean' or 'median'
# ====================================

# Create output directory if it doesn't exist
os.makedirs(OUT_DIR, exist_ok=True)

# Enable GPU memory growth to prevent TensorFlow from hogging all VRAM
for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass

# Regex to identify TFRecord files
_TFREC_PAT = re.compile(r"\.tfrecords?($|-)", re.IGNORECASE)
is_tfrec = lambda n: _TFREC_PAT.search(os.path.basename(n)) is not None

def load_jobs(p):
    """
    Parses the summary.json file to create a list of processing jobs.
    Resolves the full path of the TFRecord files.
    """
    with open(p, "r") as f:
        items = json.load(f)
    jobs = []
    for it in items:
        fn = it["filename"]
        if not is_tfrec(fn): continue

        # Check in root dir, then check subdirectories (training/validation/testing)
        found_path = os.path.join(TFREC_DIR, fn)
        if not os.path.exists(found_path):
            for sub in ["training", "validation", "testing"]:
                candidate = os.path.join(TFREC_DIR, sub, fn)
                if os.path.exists(candidate):
                    found_path = candidate
                    break
        
        # Get the case label (e.g., "Left Turn", "Lane Change")
        case_label = str(it.get("case", it.get("cases", it.get("scenario", "unspecified"))))
        jobs.append({
            "path": found_path,
            "indices": [int(x) for x in it.get("segment_indices", [])], # Which specific examples in the file to run
            "case": case_label
        })
    return jobs

def fval(ex, key, default=None):
    """Helper to extract float_list or int64_list features from a TFExample."""
    feat = ex.features.feature
    if key not in feat: return default
    fl = feat[key].float_list.value
    il = feat[key].int64_list.value
    if len(fl): return np.array(fl, dtype=np.float32)
    if len(il): return np.array(il, dtype=np.int64)
    return default

def reshape2(a, n, t, fill=np.nan, dtype=np.float32):
    """
    Reshapes a flat array into (N_AGENTS, TIMESTEPS).
    Handles padding if the input array is shorter than expected.
    """
    if a is None: return np.full((n,t), fill, dtype=dtype)
    a = np.asarray(a); need = n*t
    if a.size < need: a = np.concatenate([a, np.full(need-a.size, fill, dtype=a.dtype)])
    return a.reshape(n,t)

def type_names_from_codes(codes):
    """Maps integer type codes to human-readable strings (Vehicle, Pedestrian, etc.)."""
    mp = {0:"UNSET", 1:"VEHICLE", 2:"PEDESTRIAN", 3:"CYCLIST"}
    if codes is None: codes = np.zeros(N_AGENTS, np.int64)
    return np.vectorize(lambda x: mp.get(int(x), "OTHER"))(codes)

def to_av_local(xs, ys, cur_x, cur_y, vx_c=None, vy_c=None, is_av_mask=None):
    """
    Transforms global coordinates (xs, ys) into the AV's local coordinate system.
    The AV at the current timestep (t=0) becomes (0,0) with heading 0 (facing +x).
    """
    if not USE_LOCAL_FRAME or is_av_mask is None or not np.any(is_av_mask):
        return xs, ys, 0.0, 0.0, 0.0
    
    # Find the index of the AV (SDC - Self Driving Car)
    av = np.where(is_av_mask)[0][0]
    x0 = float(cur_x[av,0]); y0 = float(cur_y[av,0])
    
    # Calculate AV heading based on velocity vector
    if vx_c is not None and vy_c is not None and (abs(vx_c[av,0])+abs(vy_c[av,0])>1e-4):
        yaw0 = np.arctan2(float(vy_c[av,0]), float(vx_c[av,0]))
    else:
        yaw0 = 0.0 # Default if stationary

    # Translation
    X = xs - x0; Y = ys - y0
    
    # Rotation (2D rotation matrix)
    c,s = np.cos(-yaw0), np.sin(-yaw0)
    return c*X - s*Y, s*X + c*Y, x0, y0, yaw0

def world_to_local_xy(x, y, x0, y0, yaw0):
    """Transforms a single point (or array of points) from World -> Local."""
    X,Y = x-x0, y-y0
    c,s = np.cos(-yaw0), np.sin(-yaw0)
    return c*X - s*Y, s*X + c*Y

def vec_world_to_local(vx, vy, yaw0):
    """Transforms a vector (e.g., velocity) from World -> Local (Rotation only)."""
    c,s = np.cos(-yaw0), np.sin(-yaw0)
    return c*vx - s*vy, s*vx + c*vy

def group_roadgraph_points(xyz, ids):
    """Groups flat roadgraph point arrays into a dictionary of polylines keyed by ID."""
    out = {}
    if xyz is None or ids is None or len(xyz)==0: return out
    xy = xyz.reshape(-1,3)[:, :2]; ids = ids.reshape(-1)
    for rg_id in np.unique(ids):
        pts = xy[ids == rg_id]
        if pts.shape[0] >= 2: out[int(rg_id)] = pts
    return out

def reduce_points_to_single(xy, reducer="mean"):
    """Reduces a cluster of points (e.g., a stop line) to a single (x,y) coordinate."""
    if xy is None or len(xy)==0: return None
    return np.median(xy, axis=0) if reducer=="median" else xy.mean(axis=0)

# Global accumulators used to store results across all scenarios for final boxplots
AV_DIST_TO_STOP, HV_DIST_TO_STOP = [], []
AV_LANE_OFFSET, HV_LANE_OFFSET = [], []
AV_LANE_OFFSET_BY_SCENARIO = {}
AV_OFFSET_MEANS, HV_OFFSET_MEANS = [], []
AV_OFFSET_STDS, HV_OFFSET_STDS = [], []
AV_OFFSET_MEAN_BY_CASE = defaultdict(list)
HV_OFFSET_MEAN_BY_CASE = defaultdict(list)
AV_OFFSET_STD_BY_CASE = defaultdict(list)
HV_OFFSET_STD_BY_CASE = defaultdict(list)

def closest_point_on_polyline(polyline, px, py):
    """
    Finds the closest point on a polyline to a given point (px, py).
    Returns:
      - best_point: (x,y) on the line
      - best_dist: Euclidean distance
      - best_tangent: Unit vector describing the direction of the line segment
    """
    if polyline is None or len(polyline) == 0:
        return None, None, None
    
    # If polyline is just a single point
    if len(polyline) == 1:
        point = polyline[0]
        dist = float(np.hypot(point[0] - px, point[1] - py))
        return point, dist, np.array([1.0, 0.0], dtype=np.float32)
    
    P = np.array([px, py], dtype=np.float32)
    best_point = None
    best_dist = None
    best_tangent = None

    # Iterate over every segment of the polyline
    for i in range(len(polyline) - 1):
        A = polyline[i]
        B = polyline[i + 1]
        v = B - A
        seg_len2 = float(np.dot(v, v))
        if seg_len2 < 1e-6: continue # Skip zero-length segments

        # Project point P onto vector AB (clamped between 0 and 1)
        t = float(np.clip(np.dot(P - A, v) / seg_len2, 0.0, 1.0))
        closest = A + t * v
        dist = float(np.linalg.norm(P - closest))

        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_point = closest
            # Calculate unit tangent vector
            tangent_vec = v / math.sqrt(seg_len2)
            best_tangent = tangent_vec.astype(np.float32)

    # Fallback if calculation failed
    if best_point is None:
        best_point = polyline[0]
        best_dist = float(np.linalg.norm(P - best_point))
        best_tangent = np.array([1.0, 0.0], dtype=np.float32)
    
    return best_point, best_dist, best_tangent

def _normalize_vector(vec, eps=1e-3):
    """Normalizes a 2D vector. Returns None if length is too small."""
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return None
    return vec / norm

def _mark_with_padding(mask, center_idx, pad):
    """Sets a boolean mask to True at center_idx and 'pad' neighbors."""
    if mask.size == 0: return
    start = max(0, center_idx - pad)
    end = min(mask.size - 1, center_idx + pad)
    mask[start:end+1] = True

def estimate_heading_at_index(xs_row, ys_row, vidx, idx_in_vidx, default_heading=None):
    """
    Estimates the heading of an agent at a specific timestep using Finite Differences.
    1. Tries Forward Difference (Next - Current)
    2. Tries Backward Difference (Current - Prev)
    3. Falls back to default_heading or +X axis.
    """
    t = vidx[idx_in_vidx]
    # Forward difference
    if idx_in_vidx + 1 < len(vidx):
        nxt = vidx[idx_in_vidx + 1]
        vec = np.array([xs_row[nxt] - xs_row[t], ys_row[nxt] - ys_row[t]], dtype=np.float32)
        normed = _normalize_vector(vec)
        if normed is not None: return normed
    # Backward difference
    if idx_in_vidx > 0:
        prev = vidx[idx_in_vidx - 1]
        vec = np.array([xs_row[t] - xs_row[prev], ys_row[t] - ys_row[prev]], dtype=np.float32)
        normed = _normalize_vector(vec)
        if normed is not None: return normed
    # Fallback
    if default_heading is not None:
        normed = _normalize_vector(default_heading)
        if normed is not None: return normed
    return np.array([1.0, 0.0], dtype=np.float32)

def compute_lane_offset_for_point(P, heading_vec, lane_segments_local):
    """
    Core Logic: Computes the signed offset from the nearest lane center.
    Returns (offset, alignment_violation).
    alignment_violation is True if the vehicle is not parallel to the lane (likely changing lanes).
    """
    if not lane_segments_local:
        return None, False
    
    left_candidate = None
    right_candidate = None

    # Find the nearest lane boundaries (or centerlines) to the left and right
    for polyline in lane_segments_local.values():
        point, dist, tangent = closest_point_on_polyline(polyline, P[0], P[1])
        if point is None or dist is None: continue

        # Use Cross Product (2D determinant) to determine if line is Left or Right of agent
        # rel vector from agent (P) to point on line
        rel = point - P
        side = heading_vec[0] * rel[1] - heading_vec[1] * rel[0]
        
        candidate = (point, dist, tangent)
        if side >= 0: # Left side
            if left_candidate is None or dist < left_candidate[1]:
                left_candidate = candidate
        else: # Right side
            if right_candidate is None or dist < right_candidate[1]:
                right_candidate = candidate

    # We need both a left and right candidate to define a "lane" context
    if not left_candidate or not right_candidate:
        return None, False

    # Heuristic: If nearest lines are > 3.7m away, we probably aren't in a standard lane
    if left_candidate[1] > 3.7 or right_candidate[1] > 3.7:
        return None, False

    # Calculate virtual center point between the two found lines
    center_point = 0.5 * (left_candidate[0] + right_candidate[0])
    tangent_vec = left_candidate[2] + right_candidate[2]
    normed_tangent = _normalize_vector(tangent_vec)
    if normed_tangent is None: normed_tangent = heading_vec
    
    # --- Lane Change Filtering ---
    if FILTER_LANE_CHANGE:
        # Check alignment: dot product of Heading vs Lane Tangent
        dot_val = np.dot(heading_vec, normed_tangent)
        angle_diff = np.arccos(np.clip(dot_val, -1.0, 1.0))
        # If angle is too large, assume lane change and reject this sample
        if abs(angle_diff) > MAX_LANE_ALIGNMENT_DIFF:
            return None, True

    # Calculate signed distance to the computed center
    # Rotate tangent 90 degrees to get normal vector
    normal_vec = np.array([-normed_tangent[1], normed_tangent[0]], dtype=np.float32)
    return float(np.dot(P - center_point, normal_vec)), False

def draw_one_sample(ex, base_name, ex_idx, hv_writer=None, av_writer=None, case_label="unspecified"):
    """
    Processes a single scenario (Example) from the TFRecord.
    1. Extracts Agent & Map data.
    2. Transforms to Local Coordinates.
    3. Calculates Metrics (Stop Dist, Lane Offset).
    4. Filters out Turns/Lane Changes.
    5. Optionally plots the scene.
    """
    # 1) Extract Agent Data
    obj_type = fval(ex,"state/type")
    is_sdc   = fval(ex,"state/is_sdc", default=np.zeros(N_AGENTS, np.int64))
    track_id = fval(ex,"state/id",     default=np.arange(N_AGENTS, dtype=np.int64))
    
    # Reshape flattened arrays into (Agents, Time)
    past_x = reshape2(fval(ex,"state/past/x"), N_AGENTS,N_PAST)
    past_y = reshape2(fval(ex,"state/past/y"), N_AGENTS,N_PAST)
    cur_x  = reshape2(fval(ex,"state/current/x"), N_AGENTS,N_CUR)
    cur_y  = reshape2(fval(ex,"state/current/y"), N_AGENTS,N_CUR)
    fut_x  = reshape2(fval(ex,"state/future/x"), N_AGENTS,N_FUT)
    fut_y  = reshape2(fval(ex,"state/future/y"), N_AGENTS,N_FUT)
    
    # Validity masks (1 = valid data exists for this frame)
    past_v = reshape2(fval(ex,"state/past/valid"), N_AGENTS,N_PAST, fill=0,dtype=np.int32)
    cur_v  = reshape2(fval(ex,"state/current/valid"), N_AGENTS,N_CUR, fill=0,dtype=np.int32)
    fut_v  = reshape2(fval(ex,"state/future/valid"), N_AGENTS,N_FUT, fill=0,dtype=np.int32)
    vx_c   = reshape2(fval(ex,"state/current/velocity_x", default=None), N_AGENTS,N_CUR)
    vy_c   = reshape2(fval(ex,"state/current/velocity_y", default=None), N_AGENTS,N_CUR)

    # Concatenate Past + Current + Future
    xs = np.concatenate([past_x,cur_x,fut_x],axis=1)
    ys = np.concatenate([past_y,cur_y,fut_y],axis=1)
    valids = np.concatenate([past_v,cur_v,fut_v],axis=1)
    types_txt = type_names_from_codes(obj_type)
    is_av = (is_sdc.astype(int)==1)

    # Transform coordinates to AV-Local Frame
    xs_loc, ys_loc, x0, y0, yaw0 = to_av_local(xs,ys,cur_x,cur_y,vx_c,vy_c,is_av)
    
    # Get current velocity in local frame for heading calculation
    CUR = N_PAST
    cur_x_loc, cur_y_loc = xs_loc[:,CUR], ys_loc[:,CUR]
    vx_cur = vx_c[:,0] if vx_c is not None else np.zeros((N_AGENTS,),np.float32)
    vy_cur = vy_c[:,0] if vy_c is not None else np.zeros((N_AGENTS,),np.float32)
    vx_loc, vy_loc = vec_world_to_local(vx_cur, vy_cur, yaw0)

    # 2) Extract Roadgraph Data
    rg_xyz = fval(ex,"roadgraph_samples/xyz")
    rg_id  = fval(ex,"roadgraph_samples/id")
    rg_t   = fval(ex,"roadgraph_samples/type")
    rg_valid = fval(ex,"roadgraph_samples/valid")
    
    # Filter only valid map points
    if rg_xyz is not None: rg_xyz = rg_xyz.reshape(-1,3)
    if rg_valid is not None:
        m = rg_valid.astype(bool)
        if rg_xyz is not None: rg_xyz = rg_xyz[m]
        if rg_id  is not None: rg_id  = rg_id[m]
        if rg_t   is not None: rg_t   = rg_t[m]

    lane_segments = {}
    stop_pts_world = np.empty((0,2),dtype=np.float32)
    
    # Process Map: Extract Lanes and Stop Signs
    if rg_t is not None and rg_xyz is not None:
        t = rg_t.reshape(-1)
        # Extract Lanes
        lane_mask = np.isin(t, list(LANE_TYPES))
        if np.any(lane_mask):
            lane_xyz = rg_xyz[lane_mask]
            lane_ids = rg_id[lane_mask] if rg_id is not None else None
            lane_segments = group_roadgraph_points(lane_xyz, lane_ids)
        # Extract Stop Signs
        stop_mask = np.isin(t, list(STOP_SIGN_TYPES))
        if np.any(stop_mask):
            stop_pts_world = rg_xyz[stop_mask][:,:2]

    # Find nearest Stop Sign location (Local Frame)
    stop_local = None
    if STOP_AS_SINGLE_POINT and stop_pts_world.shape[0]>0:
        # Group stop sign points by ID, reduce to centroid
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
        
        # Convert candidates to local frame and pick the closest one to (0,0)
        cand_local = []
        for sx,sy in candidates:
            lx,ly = world_to_local_xy(sx,sy,x0,y0,yaw0)
            cand_local.append([lx,ly])
        cand_local = np.asarray(cand_local,np.float32)
        stop_local = cand_local[int(np.argmin(np.sum(cand_local**2,axis=1)))]

    # Transform Lane Polylines to Local Frame (Pre-computation for efficiency)
    lane_segments_local = {}
    if len(lane_segments)>0:
        for lid, pts in lane_segments.items():
            Xw,Yw = pts[:,0], pts[:,1]
            X,Y = world_to_local_xy(Xw,Yw,x0,y0,yaw0) if np.any(is_av) else (Xw,Yw)
            stack = np.stack([X,Y],axis=1)
            lane_segments_local[lid] = stack

    # 3) Initialize Visualization (if enabled)
    fig = ax = None
    if EXPORT_SCENE_PNG:
        fig, ax = plt.subplots(figsize=(6,6))
        # Plot road lines
        for pts in lane_segments.values():
            Xw,Yw = pts[:,0], pts[:,1]
            X,Y = world_to_local_xy(Xw,Yw,x0,y0,yaw0) if np.any(is_av) else (Xw,Yw)
            ax.plot(X,Y,color=COLOR_MAP["lanes"],linewidth=0.6,alpha=0.8,zorder=0)
        # Plot stop sign
        if stop_local is not None:
            ax.scatter([stop_local[0]],[stop_local[1]], s=36, c=COLOR_MAP["stops"], marker="o", zorder=6)

    # 4) Process Every Agent
    scenario_key = f"{base_name}_ex{ex_idx:05d}"

    def process_agent(writer, is_av_flag, i, vidx, scenario_key):
        """Inner helper to process a single agent's trajectory."""
        default_heading = np.array([vx_loc[i], vy_loc[i]], dtype=np.float32)
        if _normalize_vector(default_heading) is None:
            default_heading = None
        stop_coords = (float(stop_local[0]), float(stop_local[1])) if (CALC_STOP_DISTANCE and stop_local is not None) else ("", "")

        agent_offsets = []

        # -- Step A: Calculate Headings and Turning Mask --
        heading_cache = []
        if CALC_LANE_CENTER_OFFSET and lane_segments_local:
            for idx_pos in range(len(vidx)):
                heading_cache.append(
                    estimate_heading_at_index(xs_loc[i], ys_loc[i], vidx, idx_pos, default_heading)
                )
        
        # Detect Turning: Calculate Yaw Rate (change in heading per frame)
        turning_mask = np.zeros(len(vidx), dtype=bool)
        if CALC_LANE_CENTER_OFFSET and FILTER_TURNING and len(vidx) >= 2:
            for idx_pos in range(1, len(vidx)):
                h_prev = heading_cache[idx_pos - 1]
                h_cur = heading_cache[idx_pos]
                if h_prev is None or h_cur is None: continue
                
                # Dot product gives Cos(theta)
                dot_yaw = np.dot(h_cur, h_prev)
                yaw_change = np.arccos(np.clip(dot_yaw, -1.0, 1.0))
                
                # If yaw change exceeds threshold, mark frame (and padding) as turning
                if yaw_change > MAX_YAW_RATE:
                    _mark_with_padding(turning_mask, idx_pos - 1, TURNING_PAD_FRAMES)
                    _mark_with_padding(turning_mask, idx_pos, TURNING_PAD_FRAMES)
        
        lane_change_mask = np.zeros(len(vidx), dtype=bool)

        # -- Step B: Iterate Time Steps --
        for idx_pos, t_idx in enumerate(vidx):
            t_idx = int(t_idx)
            px = float(xs_loc[i, t_idx]); py = float(ys_loc[i, t_idx])

            # Stop Distance Calculation
            d_stop = ""
            sx = sy = ""
            if CALC_STOP_DISTANCE and stop_local is not None:
                d_stop = float(np.hypot(px - stop_local[0], py - stop_local[1]))
                sx, sy = stop_coords
                # Save stop distance only for the LAST frame (closest point usually)
                if idx_pos == len(vidx) - 1:
                    if is_av_flag: AV_DIST_TO_STOP.append(d_stop)
                    else:          HV_DIST_TO_STOP.append(d_stop)

            # Lane Offset Calculation
            lane_offset_val = None
            heading_vec = None

            if CALC_LANE_CENTER_OFFSET and lane_segments_local:
                heading_vec = heading_cache[idx_pos] if heading_cache else None
                # Skip if Turning
                if FILTER_TURNING and turning_mask[idx_pos]:
                    heading_vec = None
                else:
                    P = np.array([px, py], dtype=np.float32)
                    # Compute Offset
                    lane_offset_val, alignment_violation = compute_lane_offset_for_point(
                        P, heading_vec, lane_segments_local
                    )
                    # Handle Lane Change detection
                    if alignment_violation:
                        _mark_with_padding(lane_change_mask, idx_pos, LANE_CHANGE_PAD_FRAMES)
                        continue
                    if lane_change_mask[idx_pos]:
                        continue
                    
                    # Store valid offset
                    if lane_offset_val is not None:
                        agent_offsets.append(lane_offset_val)
                        if is_av_flag:
                            AV_LANE_OFFSET.append(lane_offset_val)
                            AV_LANE_OFFSET_BY_SCENARIO.setdefault(scenario_key, []).append(lane_offset_val)
                        else:
                            HV_LANE_OFFSET.append(lane_offset_val)
            
            # Write to CSV
            if writer is not None:
                writer.writerow({
                    "file": base_name, "ex_idx": ex_idx, "track_id": int(track_id[i]),
                    "t": t_idx,
                    "x_local": px, "y_local": py, "valid": int(valids[i, t_idx]),
                    "stop_x_local": sx, "stop_y_local": sy, "dist_to_stop": d_stop,
                    "lane_center_offset": ("" if lane_offset_val is None else lane_offset_val),
                })

        # -- Step C: Aggregate Agent Stats --
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

    # Main Agent Loop for Visualization and Processing
    for i in range(N_AGENTS):
        # Filter by vehicle type
        if ONLY_VEHICLES and type_names_from_codes([obj_type[i]])[0] != "VEHICLE":
            continue
        vidx = np.where(valids[i]==1)[0]
        if vidx.size==0: continue
        is_av_i = bool(is_av[i])
        
        # Plot Trajectory line
        X = xs_loc[i,vidx]; Y = ys_loc[i,vidx]
        if EXPORT_SCENE_PNG:
            ax.plot(X,Y, linewidth=(2.4 if is_av_i else 1.2),
                    color=(COLOR_MAP["av"] if is_av_i else COLOR_MAP["hv"]),
                    alpha=1.0 if is_av_i else 0.85, zorder=(10 if is_av_i else 6))
            
            # Plot Direction Arrow (Current Timestep)
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
        
        # Call the processing helper
        if is_av_i:
            process_agent(av_writer, True, i, vidx, scenario_key)
        else:
            process_agent(hv_writer, False, i, vidx, scenario_key)

    # Finalize and Save Plot
    if EXPORT_SCENE_PNG:
        ax.set_aspect("equal",adjustable="box")
        ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{base_name} ex{ex_idx} (AV/HV + stop sign + directions)", fontsize=10)
        # Custom Legend
        legend = [
            Line2D([0],[0], color=COLOR_MAP["lanes"], lw=1.2, label="lane/roadgraph"),
            Line2D([0],[0], color=COLOR_MAP["av"], lw=2.4, label="AV trajectory"),
            Line2D([0],[0], color=COLOR_MAP["hv"], lw=1.2, label="HV trajectories"),
            Line2D([0],[0], marker='o', color='w', markerfacecolor=COLOR_MAP["stops"], markersize=6, label="stop sign"),
            Line2D([0],[0], color='k', lw=0, marker=r'$\rightarrow$', label="direction"),
        ]
        ax.legend(handles=legend, loc="upper right", fontsize=8, frameon=True)
        out_png = os.path.join(OUT_DIR, f"{base_name}_ex{ex_idx:05d}.png")
        fig.tight_layout(); fig.savefig(out_png, dpi=200)
        plt.close(fig)
        return out_png

    return None

def hv_speed(): pass
def hv_acceleration(): pass

def _plot_av_hv_box(av_vals, hv_vals, title, filename, scatter_seed=42, draw_zero_line=False):
    """Generic helper to draw a Boxplot comparing AV vs HV data."""
    if (not av_vals) and (not hv_vals): return
    plt.figure(figsize=(5,5))
    data = []
    labels = []
    colors = []
    # Add AV data
    if av_vals:
        data.append(av_vals); labels.append("AV"); colors.append(COLOR_MAP["av"])
    # Add HV data
    if hv_vals:
        data.append(hv_vals); labels.append("HV"); colors.append(COLOR_MAP["hv"])
    if not data:
        plt.close(); return
    
    # Draw boxplot
    bp = plt.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_edgecolor(color); patch.set_alpha(0.35)
    for element in ("medians", "caps", "whiskers"):
        for artist in bp[element]: artist.set_color("#444444"); artist.set_linewidth(1.1)

    # Overlay jittered scatter points
    if PLOT_SCATTER_POINTS:
        rng = np.random.default_rng(scatter_seed)
        for idx, vals in enumerate(data):
            if not vals: continue
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            plt.scatter(
                np.full(len(vals), idx + 1, dtype=np.float32) + jitter,
                vals, color=colors[idx], alpha=0.55, s=12,
            )

    if draw_zero_line:
        plt.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    plt.xlabel("AV / HV")
    plt.ylabel("distance")
    plt.title(title)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, filename), dpi=200)
    plt.close()

# --- Plotting Wrapper Functions ---

def make_stop_distance_boxplot():
    if not PLOT_STOP_DISTANCE: return
    if len(AV_DIST_TO_STOP)+len(HV_DIST_TO_STOP) == 0: return
    _plot_av_hv_box(AV_DIST_TO_STOP, HV_DIST_TO_STOP, "Distance to stop sign (final frame)", "dist_to_stop_boxplot.png")

def make_lane_offset_boxplot():
    """Generates a complex boxplot grouping AV results by scenario vs All HVs."""
    if not PLOT_LANE_CENTER_OFFSET: return
    if len(AV_LANE_OFFSET)+len(HV_LANE_OFFSET) == 0: return
    scenario_keys = sorted(AV_LANE_OFFSET_BY_SCENARIO.keys())

    plot_data = []; labels = []; colors = []; positions = []
    pos = 1

    # Add each AV Scenario individually
    for key in scenario_keys:
        av_vals = AV_LANE_OFFSET_BY_SCENARIO.get(key, [])
        if av_vals:
            plot_data.append(av_vals); labels.append(f"AV {key}"); colors.append(COLOR_MAP["av"])
            positions.append(pos); pos += 1

    # Add Aggregate HV
    if HV_LANE_OFFSET:
        plot_data.append(HV_LANE_OFFSET); labels.append("HV all"); colors.append(COLOR_MAP["hv"])
        positions.append(pos); pos += 1

    # Add Aggregate AV
    if AV_LANE_OFFSET:
        plot_data.append(AV_LANE_OFFSET); labels.append("AV all"); colors.append(COLOR_MAP["av"])
        positions.append(pos)

    if not plot_data: return

    width = max(6.0, 1.1 * len(plot_data))
    fig, ax = plt.subplots(figsize=(width, 5))
    ax.boxplot(plot_data, positions=positions, labels=labels, showfliers=False)

    if PLOT_SCATTER_POINTS:
        rng = np.random.default_rng(42)
        for vals, xpos, color in zip(plot_data, positions, colors):
            if not vals: continue
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(np.full(len(vals), xpos, dtype=np.float32) + jitter, vals, color=color, alpha=0.6, s=14, edgecolors="white", linewidths=0.3, zorder=3)

    ax.set_ylabel("distance")
    ax.set_title("Signed distance to nearest lane centerline")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "lane_center_offset_boxplot.png"), dpi=200)
    plt.close(fig)

def make_av_scenario_lane_offset_boxplot():
    """Generates boxplot specifically for AV stats per scenario."""
    if not PLOT_AV_SCENARIO_OFFSETS: return
    scenario_items = sorted(((key, vals) for key, vals in AV_LANE_OFFSET_BY_SCENARIO.items() if vals), key=lambda x: x[0])
    if not scenario_items: return

    labels = [key for key, _ in scenario_items]
    data = [vals for _, vals in scenario_items]
    positions = list(range(1, len(data)+1))
    fig_width = max(8.0, 0.6 * len(data))
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    bp = ax.boxplot(data, positions=positions, labels=labels, showfliers=False, patch_artist=True)

    for patch in bp["boxes"]:
        patch.set_facecolor(COLOR_MAP["av"]); patch.set_edgecolor(COLOR_MAP["av"]); patch.set_alpha(0.3)
    for element in ("medians", "caps", "whiskers"):
        for artist in bp[element]: artist.set_color("#444444"); artist.set_linewidth(1.0)

    if PLOT_SCATTER_POINTS:
        rng = np.random.default_rng(101)
        for vals, xpos in zip(data, positions):
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(np.full(len(vals), xpos, dtype=np.float32) + jitter, vals, color=COLOR_MAP["av"], alpha=0.45, s=10)

    ax.set_ylabel("distance"); ax.set_xlabel("Scenario (tfrecord)"); ax.set_title("AV lane center offset by scenario")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.tick_params(axis='x', rotation=20)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "av_lane_offset_per_scenario.png"), dpi=200)
    plt.close(fig)

def make_lane_offset_overall_boxplot():
    if not PLOT_LANE_CENTER_OFFSET_OVERALL: return
    if len(AV_LANE_OFFSET)+len(HV_LANE_OFFSET) == 0: return
    _plot_av_hv_box(AV_LANE_OFFSET, HV_LANE_OFFSET, "Aggregate lane center offset (per-frame)", "lane_center_offset_overall_boxplot.png", scatter_seed=7, draw_zero_line=True)

def make_lane_offset_mean_boxplot():
    if not PLOT_LANE_OFFSET_MEAN: return
    if len(AV_OFFSET_MEANS)+len(HV_OFFSET_MEANS) == 0: return
    _plot_av_hv_box(AV_OFFSET_MEANS, HV_OFFSET_MEANS, "Mean of Lane Center Offset per Vehicle", "lane_offset_mean_boxplot.png", scatter_seed=42, draw_zero_line=True)

def make_lane_offset_std_boxplot():
    if not PLOT_LANE_OFFSET_STD: return
    if len(AV_OFFSET_STDS)+len(HV_OFFSET_STDS) == 0: return
    _plot_av_hv_box(AV_OFFSET_STDS, HV_OFFSET_STDS, "Std. of Lane Center Offset per Vehicle", "lane_offset_std_boxplot.png", scatter_seed=52, draw_zero_line=False)

def _prepare_case_boxplot_payload(cases, av_dict, hv_dict):
    """Helper to organize data for side-by-side AV/HV comparisons."""
    data = []; positions = []; colors = []; xticks = []; xticklabels = []
    pos = 1
    for case in cases:
        start_pos = pos; added = False
        av_vals = av_dict.get(case, [])
        hv_vals = hv_dict.get(case, [])
        if av_vals:
            data.append(av_vals); positions.append(pos); colors.append(COLOR_MAP["av"]); pos += 1; added = True
        if hv_vals:
            data.append(hv_vals); positions.append(pos); colors.append(COLOR_MAP["hv"]); pos += 1; added = True
        if added:
            end_pos = pos - 1
            xticks.append(0.5 * (start_pos + end_pos)); xticklabels.append(case)
    return data, positions, colors, xticks, xticklabels

def _plot_case_box(ax, cases, av_dict, hv_dict, ylabel, title, draw_zero_line=False):
    data, positions, colors, xticks, xticklabels = _prepare_case_boxplot_payload(cases, av_dict, hv_dict)
    if not data: ax.set_visible(False); return False
    bp = ax.boxplot(data, positions=positions, widths=0.55, showfliers=False, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_edgecolor(color); patch.set_alpha(0.35)
    for element in ("medians", "caps", "whiskers"):
        for artist in bp[element]: artist.set_color("#444444"); artist.set_linewidth(1.1)

    if PLOT_SCATTER_POINTS:
        rng = np.random.default_rng(123)
        for idx, vals in enumerate(data):
            if not vals: continue
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.full(len(vals), positions[idx]) + jitter, vals, s=10, alpha=0.45, color=colors[idx])

    ax.set_xticks(xticks); ax.set_xticklabels(xticklabels, rotation=20, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title)
    if draw_zero_line: ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(True, axis="y", linestyle=":", alpha=0.3)
    return True

def make_lane_offset_case_stats_plot():
    """Generates side-by-side plots for Mean and Std Dev of lane offsets."""
    if not PLOT_LANE_OFFSET_CASE_STATS: return
    case_keys = set(AV_OFFSET_MEAN_BY_CASE.keys()) | set(HV_OFFSET_MEAN_BY_CASE.keys())
    case_keys |= set(AV_OFFSET_STD_BY_CASE.keys()) | set(HV_OFFSET_STD_BY_CASE.keys())
    cases = sorted(case_keys)
    if not cases: return

    fig_width = max(10.0, 2.5 * len(cases))
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 5), sharey=False)
    mean_ax, std_ax = axes

    mean_plotted = _plot_case_box(mean_ax, cases, AV_OFFSET_MEAN_BY_CASE, HV_OFFSET_MEAN_BY_CASE, "distance", "Lane-center mean offset by case", True)
    std_plotted = _plot_case_box(std_ax, cases, AV_OFFSET_STD_BY_CASE, HV_OFFSET_STD_BY_CASE, "distance", "Lane-center std dev by case", False)

    handles = [Line2D([0], [0], color=COLOR_MAP["av"], lw=6, alpha=0.35, label="AV"),
               Line2D([0], [0], color=COLOR_MAP["hv"], lw=6, alpha=0.35, label="HV")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)

    if not (mean_plotted or std_plotted): plt.close(fig); return
    fig.tight_layout(rect=(0.02, 0.0, 0.98, 0.95))
    fig.savefig(os.path.join(OUT_DIR, "lane_offset_case_stats.png"), dpi=200)
    plt.close(fig)

def make_boxplots():
    """Master function to trigger all configured plots."""
    if PLOT_STOP_DISTANCE: make_stop_distance_boxplot()
    if PLOT_LANE_CENTER_OFFSET: make_lane_offset_boxplot(); make_av_scenario_lane_offset_boxplot()
    if PLOT_LANE_CENTER_OFFSET_OVERALL: make_lane_offset_overall_boxplot()
    if PLOT_LANE_OFFSET_MEAN: make_lane_offset_mean_boxplot()
    if PLOT_LANE_OFFSET_STD: make_lane_offset_std_boxplot()
    if PLOT_LANE_OFFSET_CASE_STATS: make_lane_offset_case_stats_plot()

def main():
    jobs = load_jobs(JSON_PATH)
    if not jobs:
        print("JSON is empty or no matching tfrecords found; check JSON_PATH/TFREC_DIR."); return

    # Setup CSV Writers
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

    # Main Processing Loop
    for job in jobs:
        tfrec_path = job["path"]
        indices = job.get("indices", [])
        case_label = job.get("case", "unspecified")
        if not os.path.exists(tfrec_path):
            print("File not found:", tfrec_path); continue
        
        print("Reading:", os.path.basename(tfrec_path), "segments:", len(indices), "indices:", indices, "case:", case_label)
        
        # Load TFRecord Dataset
        ds = tf.data.TFRecordDataset(tfrec_path)
        recs = list(ds.as_numpy_iterator())
        
        # Process specified indices
        for ex_idx in indices:
            if ex_idx < 0 or ex_idx >= len(recs):
                print(f"  - Skip idx={ex_idx} (OutOfBounds, total {len(recs)})"); continue
            
            ex = example_pb2.Example(); ex.ParseFromString(recs[ex_idx])
            base = os.path.basename(tfrec_path)
            
            # Execute logic for one sample
            png = draw_one_sample(
                ex, base, ex_idx,
                hv_writer=hv_writer,
                av_writer=av_writer,
                case_label=case_label,
            )
            if png: print("  + saved:", os.path.basename(png))
            else:   print("  + processed (PNG export disabled)")

    if hv_f: hv_f.close()
    if av_f: av_f.close()

    # Generate Summary Plots
    make_boxplots()
    print("✅ Done. CSV & plots saved in:", OUT_DIR)

if __name__ == "__main__":
    main()