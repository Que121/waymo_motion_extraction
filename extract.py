# -*- coding: utf-8 -*-
"""
This script extracts and visualizes specific scenarios from the Waymo Open Motion Dataset (WOMD).

It processes TFRecord files based on a JSON configuration, transforms coordinates
to the Autonomous Vehicle's (AV) local frame, calculates metrics like distance to
stop signs and lane center offset, and exports trajectories to CSV files and
visualizations to PNG files.
"""

import os, json, re, math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tensorflow.core.example import example_pb2

# ========== Configuration ==========
JSON_PATH = "./test_one.json"  # Path to the JSON file defining which scenarios to extract
TFREC_DIR = "/mnt/home/Files/Lab/waymo_motion/dataset/validation" # Directory containing the .tfrecord files
OUT_DIR   = "./extract_out"  # Directory to save CSV files and PNG plots

USE_LOCAL_FRAME = True  # Whether to transform all coordinates to the AV's local frame
ONLY_VEHICLES   = True  # If True, only process agents of type VEHICLE
XLIM = (-120, 120)      # X-axis limits for plots
YLIM = (-120, 120)      # Y-axis limits for plots

EXPORT_HV_CSV = True    # Toggle to save "Human Vehicle" (non-AV) trajectories
EXPORT_AV_CSV = True    # Toggle to save "Autonomous Vehicle" (SDC) trajectories
EXPORT_SCENE_PNG = False # Toggle to save a PNG visualization of each scene

# Metrics toggles
CALC_STOP_DISTANCE = False  # If True, calculate distance to the nearest stop sign
PLOT_STOP_DISTANCE = False  # If True, generate a boxplot for stop sign distances
CALC_LANE_CENTER_OFFSET = True # If True, calculate signed distance to the lane centerline
PLOT_LANE_CENTER_OFFSET = True # If True, generate a boxplot for lane offsets

# Directional arrow plotting parameters
DRAW_DIR_ARROWS = True  # If True, draw an arrow indicating the current velocity vector
ARROW_LEN   = 6.0       # Length of the direction arrow in meters
ARROW_WIDTH = 0.006     # Width of the direction arrow

# Color mapping for plots
COLOR_MAP = {
    "lanes": "#666666",  # Lane centerlines
    "stops": "#d62728",  # Stop signs
    "av":    "#ff7f0e",  # Autonomous Vehicle (AV) / Self-Driving Car (SDC)
    "hv":    "#1f77b4",  # Human-driven Vehicles (HV)
}

# WOMD TFExample shape constants
N_AGENTS = 128          # Maximum number of agents per scenario
N_PAST, N_CUR, N_FUT = 10, 1, 80  # 10Hz data => 1.0s past, 0.1s current, 8.0s future = 91 total frames

# Roadgraph types (covers lane center / roadline / road edge)
LANE_TYPES      = {0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16}
STOP_SIGN_TYPES = {17}          # Confirmed definition for stop signs

STOP_AS_SINGLE_POINT = True  # If True, reduce stop sign polygons to a single point
STOP_POINT_REDUCER   = "mean" # Method to reduce points ("mean" or "median")
# ====================================

# Create output directory if it doesn't exist
os.makedirs(OUT_DIR, exist_ok=True)
# Configure TensorFlow to grow GPU memory as needed
for g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass

# Regex to identify TFRecord files
_TFREC_PAT = re.compile(r"\.tfrecords?($|-)", re.IGNORECASE)
is_tfrec = lambda n: _TFREC_PAT.search(os.path.basename(n)) is not None

def load_jobs(p):
    """Loads scenario extraction jobs from the JSON_PATH file."""
    with open(p, "r") as f:
        items = json.load(f)
    jobs = []
    for it in items:
        fn = it["filename"]
        if not is_tfrec(fn): continue
        # Append a tuple: (full_tfrecord_path, [list_of_segment_indices])
        jobs.append((os.path.join(TFREC_DIR, fn), [int(x) for x in it.get("segment_indices", [])]))
    return jobs

def fval(ex, key, default=None):
    """Extracts a feature list from a tf.Example proto by key."""
    feat = ex.features.feature
    if key not in feat: return default
    fl = feat[key].float_list.value
    il = feat[key].int64_list.value
    if len(fl): return np.array(fl, dtype=np.float32)
    if len(il): return np.array(il, dtype=np.int64)
    return default

def reshape2(a, n, t, fill=np.nan, dtype=np.float32):
    """Reshapes a flat array 'a' to (n, t), padding with 'fill' if necessary."""
    if a is None: return np.full((n,t), fill, dtype=dtype)
    a = np.asarray(a); need = n*t
    if a.size < need: a = np.concatenate([a, np.full(need-a.size, fill, dtype=a.dtype)])
    return a.reshape(n,t)

def type_names_from_codes(codes):
    """Maps agent type integer codes to human-readable names."""
    mp = {0:"UNSET", 1:"VEHICLE", 2:"PEDESTRIAN", 3:"CYCLIST"}
    if codes is None: codes = np.zeros(N_AGENTS, np.int64)
    return np.vectorize(lambda x: mp.get(int(x), "OTHER"))(codes)

def to_av_local(xs, ys, cur_x, cur_y, vx_c=None, vy_c=None, is_av_mask=None):
    """
    Transforms world coordinates (xs, ys) to the AV's local frame.
    The local frame is centered at the AV's current position (x0, y0) and
    rotated by its current yaw (yaw0).
    """
    if not USE_LOCAL_FRAME or is_av_mask is None or not np.any(is_av_mask):
        # Return world frame if not using local or no AV found
        return xs, ys, 0.0, 0.0, 0.0
    
    # Find the first agent marked as the AV (is_sdc=1)
    av_idx = np.where(is_av_mask)[0][0]
    x0 = float(cur_x[av_idx, 0])
    y0 = float(cur_y[av_idx, 0])
    
    # Calculate yaw from current velocity vector
    if vx_c is not None and vy_c is not None and (abs(vx_c[av_idx,0])+abs(vy_c[av_idx,0]) > 1e-4):
        yaw0 = np.arctan2(float(vy_c[av_idx,0]), float(vx_c[av_idx,0]))
    else:
        # Default to 0 yaw if velocity is negligible
        yaw0 = 0.0
        
    # Apply transformation (translate then rotate)
    X = xs - x0
    Y = ys - y0
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    return c*X - s*Y, s*X + c*Y, x0, y0, yaw0

def world_to_local_xy(x, y, x0, y0, yaw0):
    """Helper to transform a single (x, y) point to the local frame."""
    X, Y = x-x0, y-y0
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    return c*X - s*Y, s*X + c*Y

def vec_world_to_local(vx, vy, yaw0):
    """Helper to rotate a velocity vector (vx, vy) to the local frame."""
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    return c*vx - s*vy, s*vx + c*vy

def group_roadgraph_points(xyz, ids):
    """Groups roadgraph (x,y,z) points by their unique element ID."""
    out = {}
    if xyz is None or ids is None or len(xyz)==0: return out
    xy = xyz.reshape(-1,3)[:, :2] # Discard Z coordinate
    ids = ids.reshape(-1)
    for rg_id in np.unique(ids):
        pts = xy[ids == rg_id]
        if pts.shape[0] >= 2: # Only keep polylines with at least 2 points
            out[int(rg_id)] = pts
    return out

def reduce_points_to_single(xy, reducer="mean"):
    """Reduces a set of points (e.g., a stop sign) to a single centroid."""
    if xy is None or len(xy)==0: return None
    if reducer == "median":
        return np.median(xy, axis=0)
    else:
        return xy.mean(axis=0)

# Global accumulators (for boxplots)
AV_DIST_TO_STOP, HV_DIST_TO_STOP = [], []
AV_LANE_OFFSET, HV_LANE_OFFSET = [], []
AV_LANE_OFFSET_BY_SCENARIO = {} # For per-scenario plotting

def closest_point_on_polyline(polyline, px, py):
    """Finds the closest point, distance, and unit tangent vector on a polyline to a given point (px, py)."""
    if polyline is None or len(polyline) == 0:
        return None, None, None
    if len(polyline) == 1:
        # Handle polyline with a single point
        point = polyline[0]
        dist = float(np.hypot(point[0] - px, point[1] - py))
        return point, dist, np.array([1.0, 0.0], dtype=np.float32) # Default tangent
    
    P = np.array([px, py], dtype=np.float32)
    best_point = None
    best_dist = None
    best_tangent = None

    # Iterate over all segments (A, B) in the polyline
    for i in range(len(polyline) - 1):
        A = polyline[i]
        B = polyline[i + 1]
        v = B - A # Segment vector
        seg_len2 = float(np.dot(v, v))
        if seg_len2 < 1e-6:
            continue # Skip zero-length segments
        
        # Project P onto the line defined by the segment
        # t is the normalized projection parameter
        t = float(np.clip(np.dot(P - A, v) / seg_len2, 0.0, 1.0))
        
        closest = A + t * v # Closest point *on the segment*
        dist = float(np.linalg.norm(P - closest))
        
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_point = closest
            tangent_vec = v / math.sqrt(seg_len2) # Unit tangent of the segment
            best_tangent = tangent_vec.astype(np.float32)
            
    if best_point is None:
        # Fallback if all segments were zero-length
        best_point = polyline[0]
        best_dist = float(np.linalg.norm(P - best_point))
        best_tangent = np.array([1.0, 0.0], dtype=np.float32)
        
    return best_point, best_dist, best_tangent

def _normalize_vector(vec, eps=1e-3):
    """Safely normalizes a 2D vector, returning None if norm is near zero."""
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return None
    return vec / norm

def estimate_heading_at_index(xs_row, ys_row, vidx, idx_in_vidx, default_heading=None):
    """
    Returns the unit heading vector at the timestamp vidx[idx_in_vidx].
    Tries forward difference, then backward difference, then default_heading.
    """
    t = vidx[idx_in_vidx]
    
    # Try forward difference
    if idx_in_vidx + 1 < len(vidx):
        nxt = vidx[idx_in_vidx + 1]
        vec = np.array([xs_row[nxt] - xs_row[t], ys_row[nxt] - ys_row[t]], dtype=np.float32)
        normed = _normalize_vector(vec)
        if normed is not None:
            return normed
            
    # Try backward difference
    if idx_in_vidx > 0:
        prev = vidx[idx_in_vidx - 1]
        vec = np.array([xs_row[t] - xs_row[prev], ys_row[t] - ys_row[prev]], dtype=np.float32)
        normed = _normalize_vector(vec)
        if normed is not None:
            return normed
            
    # Fall back to default heading (usually current velocity)
    if default_heading is not None:
        normed = _normalize_vector(default_heading)
        if normed is not None:
            return normed
            
    # Default to facing positive x-axis
    return np.array([1.0, 0.0], dtype=np.float32)

def compute_lane_offset_for_point(P, heading_vec, lane_segments_local):
    """
    Calculates the signed distance from a vehicle point (P) to the lane centerline.
    The centerline is estimated from the nearest left and right lane boundaries.
    Signed distance: Left is positive, Right is negative.
    """
    if not lane_segments_local:
        return None
        
    left_candidate = None
    right_candidate = None

    # Find the closest lane boundary on the left and right
    for polyline in lane_segments_local.values():
        point, dist, tangent = closest_point_on_polyline(polyline, P[0], P[1])
        if point is None or dist is None:
            continue
        
        rel = point - P # Vector from vehicle to point on lane
        # Use 2D cross product to determine side (left/right)
        side = heading_vec[0] * rel[1] - heading_vec[1] * rel[0]
        
        candidate = (point, dist, tangent)
        if side >= 0: # Point is to the "left" of the vehicle's heading
            if left_candidate is None or dist < left_candidate[1]:
                left_candidate = candidate
        else: # Point is to the "right"
            if right_candidate is None or dist < right_candidate[1]:
                right_candidate = candidate

    # We need both a left and right boundary to estimate a centerline
    if not left_candidate or not right_candidate:
        return None

    # Filter out boundaries that are too far away (e.g., adjacent lanes)
    if left_candidate[1] > 3.7 or right_candidate[1] > 3.7: # 3.7m is a typical lane width
        return None

    # Estimate centerline and orientation
    center_point = 0.5 * (left_candidate[0] + right_candidate[0])
    tangent_vec = left_candidate[2] + right_candidate[2] # Average tangents
    normed_tangent = _normalize_vector(tangent_vec)
    if normed_tangent is None:
        normed_tangent = heading_vec # Fallback to vehicle heading
        
    # Get the normal vector (points "left" of the lane tangent)
    normal_vec = np.array([-normed_tangent[1], normed_tangent[0]], dtype=np.float32)
    
    # Project the vector from centerline-to-vehicle onto the normal vector
    return float(np.dot(P - center_point, normal_vec))

def draw_one_sample(ex, base_name, ex_idx, hv_writer=None, av_writer=None):
    """
    Main processing function for a single scenario (tf.Example).
    Extracts data, calculates metrics, writes to CSV, and (optionally) plots.
    """
    
    # 1) Load and Reshape Agent Data
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

    # Concatenate all time steps
    xs = np.concatenate([past_x,cur_x,fut_x],axis=1)
    ys = np.concatenate([past_y,cur_y,fut_y],axis=1)
    valids = np.concatenate([past_v,cur_v,fut_v],axis=1)
    types_txt = type_names_from_codes(obj_type)
    is_av = (is_sdc.astype(int)==1)

    # Transform all agent coordinates to AV's local frame
    xs_loc, ys_loc, x0, y0, yaw0 = to_av_local(xs,ys,cur_x,cur_y,vx_c,vy_c,is_av)
    CUR = N_PAST # Index of the "current" time step
    cur_x_loc, cur_y_loc = xs_loc[:,CUR], ys_loc[:,CUR]
    
    # Get current velocity vectors
    vx_cur = vx_c[:,0] if vx_c is not None else np.zeros((N_AGENTS,),np.float32)
    vy_cur = vy_c[:,0] if vy_c is not None else np.zeros((N_AGENTS,),np.float32)
    # Rotate velocity vectors to local frame
    vx_loc, vy_loc = vec_world_to_local(vx_cur, vy_cur, yaw0)

    # 2) Load and Filter Roadgraph Data
    rg_xyz = fval(ex,"roadgraph_samples/xyz")
    rg_id  = fval(ex,"roadgraph_samples/id")
    rg_t   = fval(ex,"roadgraph_samples/type")
    rg_valid = fval(ex,"roadgraph_samples/valid")
    
    if rg_xyz is not None: rg_xyz = rg_xyz.reshape(-1,3)
    # Apply validity mask to all roadgraph features
    if rg_valid is not None:
        m = rg_valid.astype(bool)
        if rg_xyz is not None: rg_xyz = rg_xyz[m]
        if rg_id  is not None: rg_id  = rg_id[m]
        if rg_t   is not None: rg_t   = rg_t[m]

    lane_segments = {} # Dict to store {id: [N,2] points}
    lane_type_map = {} # Dict to store {id: type_code}
    stop_pts_world = np.empty((0,2),dtype=np.float32) # Array for all stop sign points

    if rg_t is not None and rg_xyz is not None:
        t = rg_t.reshape(-1)
        
        # Filter for lane points
        lane_mask = np.isin(t, list(LANE_TYPES))
        if np.any(lane_mask):
            lane_xyz = rg_xyz[lane_mask]
            lane_ids = rg_id[lane_mask] if rg_id is not None else None
            lane_types = rg_t[lane_mask] if rg_t is not None else None
            # Group points into polylines by ID
            lane_segments = group_roadgraph_points(lane_xyz, lane_ids)
            # Create a map from lane ID to lane type
            if lane_ids is not None and lane_types is not None:
                flat_ids = lane_ids.reshape(-1)
                flat_types = lane_types.reshape(-1)
                for lid in np.unique(flat_ids):
                    mask_lid = (flat_ids == lid)
                    if np.any(mask_lid):
                        lane_type_map[int(lid)] = int(np.round(np.mean(flat_types[mask_lid])))
        
        # Filter for stop sign points
        stop_mask = np.isin(t, list(STOP_SIGN_TYPES))
        if np.any(stop_mask):
            stop_pts_world = rg_xyz[stop_mask][:,:2]

    # 3) Process Stop Signs (if enabled)
    stop_local = None # The single, closest stop sign point in the local frame
    if STOP_AS_SINGLE_POINT and stop_pts_world.shape[0]>0:
        if rg_id is not None and 'stop_mask' in locals() and np.any(stop_mask):
            # Group stop sign points by ID and find centroid of each
            stop_ids = rg_id[stop_mask]
            cents = []
            for gid in np.unique(stop_ids):
                c = reduce_points_to_single(stop_pts_world[stop_ids==gid], STOP_POINT_REDUCER)
                if c is not None: cents.append(c)
            candidates = np.vstack(cents) if len(cents)>0 else stop_pts_world
        else:
            # If no IDs, just find centroid of all stop points
            c = reduce_points_to_single(stop_pts_world, STOP_POINT_REDUCER)
            candidates = c.reshape(1,2) if c is not None else stop_pts_world
        
        # Transform all candidate stop points to local frame
        cand_local = []
        for sx,sy in candidates:
            lx,ly = world_to_local_xy(sx,sy,x0,y0,yaw0)
            cand_local.append([lx,ly])
        cand_local = np.asarray(cand_local,np.float32)
        
        # Find the one closest to the AV (origin)
        stop_local = cand_local[int(np.argmin(np.sum(cand_local**2,axis=1)))]

    # 4) Process Lane Segments (Local Frame)
    lane_segments_local = {} # Dict to store {id: [N,2] local points}
    if len(lane_segments)>0:
        for lid, pts in lane_segments.items():
            Xw,Yw = pts[:,0], pts[:,1]
            # Transform to local frame
            X,Y = world_to_local_xy(Xw,Yw,x0,y0,yaw0) if np.any(is_av) else (Xw,Yw)
            stack = np.stack([X,Y],axis=1)
            lane_segments_local[lid] = stack

    # 5) Initialize Plot (if enabled)
    fig = ax = None
    if EXPORT_SCENE_PNG:
        fig, ax = plt.subplots(figsize=(6,6))
        # Plot all lane segments
        for pts in lane_segments_local.values():
            X,Y = pts[:,0], pts[:,1]
            ax.plot(X,Y,color=COLOR_MAP["lanes"],linewidth=0.6,alpha=0.8,zorder=0)
        # Plot the single stop sign point
        if stop_local is not None:
            ax.scatter([stop_local[0]],[stop_local[1]], s=36, c=COLOR_MAP["stops"], marker="o", zorder=6)

    # 6) Process Each Agent (Write CSV & Plot Trajectories)
    scenario_key = f"{base_name}_ex{ex_idx:05d}"

    def write_rows(writer, is_av_flag, i, vidx, scenario_key):
        """Inner function to write all timesteps for agent 'i' to the CSV writer."""
        if writer is None: return
        
        # Get the default heading (current velocity) for metric calculation fallbacks
        default_heading = np.array([vx_loc[i], vy_loc[i]], dtype=np.float32)
        if _normalize_vector(default_heading) is None:
            default_heading = None # Current velocity is zero
            
        stop_coords = (float(stop_local[0]), float(stop_local[1])) if (CALC_STOP_DISTANCE and stop_local is not None) else ("", "")

        for idx_pos, t_idx in enumerate(vidx):
            t_idx = int(t_idx)
            px = float(xs_loc[i, t_idx]); py = float(ys_loc[i, t_idx])

            # --- Calculate Stop Distance Metric ---
            d_stop = ""
            sx = sy = ""
            if CALC_STOP_DISTANCE and stop_local is not None:
                d_stop = float(np.hypot(px - stop_local[0], py - stop_local[1]))
                sx, sy = stop_coords
                if idx_pos == len(vidx) - 1:  # Only accumulate stats for the final frame
                    if is_av_flag:
                        AV_DIST_TO_STOP.append(d_stop)
                    else:
                        HV_DIST_TO_STOP.append(d_stop)

            # --- Calculate Lane Center Offset Metric ---
            lane_offset_val = None
            if CALC_LANE_CENTER_OFFSET and lane_segments_local:
                # Estimate heading at this specific timestep
                heading_vec = estimate_heading_at_index(xs_loc[i], ys_loc[i], vidx, idx_pos, default_heading)
                P = np.array([px, py], dtype=np.float32)
                lane_offset_val = compute_lane_offset_for_point(P, heading_vec, lane_segments_local)
                
                if lane_offset_val is not None:
                    # Accumulate stats for all valid timesteps
                    if is_av_flag:
                        AV_LANE_OFFSET.append(lane_offset_val)
                        AV_LANE_OFFSET_BY_SCENARIO.setdefault(scenario_key, []).append(lane_offset_val)
                    else:
                        HV_LANE_OFFSET.append(lane_offset_val)
            
            # Write one row per timestep
            writer.writerow({
                "file": base_name, "ex_idx": ex_idx, "track_id": int(track_id[i]),
                "t": t_idx,
                "x_local": px, "y_local": py, "valid": int(valids[i, t_idx]),
                "stop_x_local": sx, "stop_y_local": sy, "dist_to_stop": d_stop,
                "lane_center_offset": ("" if lane_offset_val is None else lane_offset_val),
            })
    
    # --- Main Agent Loop ---
    for i in range(N_AGENTS):
        # Filter out non-vehicles if requested
        if ONLY_VEHICLES and type_names_from_codes([obj_type[i]])[0] != "VEHICLE":
            continue
            
        vidx = np.where(valids[i]==1)[0] # Get all valid time indices
        if vidx.size==0: continue
        
        is_av_i = bool(is_av[i])
        X = xs_loc[i,vidx]; Y = ys_loc[i,vidx] # Get local trajectory
        
        if EXPORT_SCENE_PNG:
            # Plot the trajectory
            ax.plot(X,Y, linewidth=(2.4 if is_av_i else 1.2),
                    color=(COLOR_MAP["av"] if is_av_i else COLOR_MAP["hv"]),
                    alpha=1.0 if is_av_i else 0.85, zorder=(10 if is_av_i else 6))
            
            # Plot the direction arrow
            if DRAW_DIR_ARROWS:
                cx=float(cur_x_loc[i]); cy=float(cur_y_loc[i])
                vx=float(vx_loc[i]);    vy=float(vy_loc[i])
                # Only draw if position is valid and velocity is non-zero
                if np.isfinite(cx) and np.isfinite(cy) and np.hypot(vx,vy)>1e-3:
                    n=(vx*vx+vy*vy)**0.5; dx=(vx/n)*ARROW_LEN; dy=(vy/n)*ARROW_LEN
                    ax.arrow(cx,cy,dx,dy, length_includes_head=True,
                             head_width=ARROW_LEN*0.25, head_length=ARROW_LEN*0.35,
                             linewidth=(2.0 if is_av_i else 1.2),
                             color=(COLOR_MAP["av"] if is_av_i else COLOR_MAP["hv"]),
                             alpha=1.0 if is_av_i else 0.9, width=ARROW_WIDTH,
                             zorder=(12 if is_av_i else 8))
        
        # Write data to the appropriate CSV file
        if is_av_i and av_writer is not None and EXPORT_AV_CSV:
            write_rows(av_writer, True, i, vidx, scenario_key)
        if (not is_av_i) and hv_writer is not None and EXPORT_HV_CSV:
            write_rows(hv_writer, False, i, vidx, scenario_key)

    # 7) Finalize and Save Plot (if enabled)
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
            Line2D([0],[0], color='k', lw=0, marker=r'$\rightarrow$', label="direction"),
        ]
        ax.legend(handles=legend, loc="upper right", fontsize=8, frameon=True)
        out_png = os.path.join(OUT_DIR, f"{base_name}_ex{ex_idx:05d}.png")
        fig.tight_layout(); fig.savefig(out_png, dpi=200)
        plt.close(fig)
        return out_png

    return None # Return None if PNG was not exported

def hv_speed():
    # TODO: Implement this function
    pass

def hv_acceleration():
    # TODO: Implement this function
    pass

def make_stop_distance_boxplot():
    """Generates and saves a boxplot for the 'distance to stop' metric."""
    if not PLOT_STOP_DISTANCE:
        return
    if len(AV_DIST_TO_STOP)+len(HV_DIST_TO_STOP) == 0:
        return
    plt.figure(figsize=(6,5))
    data, labels = [], []
    if AV_DIST_TO_STOP: data.append(AV_DIST_TO_STOP); labels.append("AV dist_to_stop")
    if HV_DIST_TO_STOP: data.append(HV_DIST_TO_STOP); labels.append("HV dist_to_stop")
    plt.boxplot(data, labels=labels, showfliers=False) # showfliers=False hides outliers
    plt.ylabel("distance (m)"); plt.title("Boxplot: Distance to stop sign (final frame)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "dist_to_stop_boxplot.png"), dpi=200)
    plt.close()

def make_lane_offset_boxplot():
    """Generates and saves a boxplot for the 'lane center offset' metric."""
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

    # Add per-scenario AV data
    for key in scenario_keys:
        av_vals = AV_LANE_OFFSET_BY_SCENARIO.get(key, [])
        if av_vals:
            plot_data.append(av_vals)
            labels.append(f"AV {key}")
            colors.append(COLOR_MAP["av"])
            positions.append(pos)
            pos += 1
    
    # Add aggregated HV data
    if HV_LANE_OFFSET:
        plot_data.append(HV_LANE_OFFSET)
        labels.append("HV (all)")
        colors.append(COLOR_MAP["hv"])
        positions.append(pos)
        pos += 1

    # Add aggregated AV data
    if AV_LANE_OFFSET:
        plot_data.append(AV_LANE_OFFSET)
        labels.append("AV (all)")
        colors.append(COLOR_MAP["av"])
        positions.append(pos)

    if not plot_data:
        return

    # Create boxplot
    width = max(6.0, 1.1 * len(plot_data))
    fig, ax = plt.subplots(figsize=(width, 5))
    ax.boxplot(plot_data, positions=positions, labels=labels, showfliers=False)

    # Overlay jittered scatter plot for better visibility
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
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--") # Draw y=0 line
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "lane_center_offset_boxplot.png"), dpi=200)
    plt.close(fig)

def make_boxplots():
    """Wrapper function to generate all enabled metric plots."""
    if PLOT_STOP_DISTANCE:
        make_stop_distance_boxplot()
    if PLOT_LANE_CENTER_OFFSET:
        make_lane_offset_boxplot()

def main():
    jobs = load_jobs(JSON_PATH)
    if not jobs:
        print("JSON is empty or no tfrecords matched; check JSON_PATH/TFREC_DIR."); return

    hv_writer = av_writer = None
    hv_f = av_f = None
    
    # Initialize CSV writers
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

    # --- Main Processing Loop ---
    for tfrec_path, indices in jobs:
        if not os.path.exists(tfrec_path):
            print("File not found:", tfrec_path); continue
        
        print("Reading:", os.path.basename(tfrec_path), "segments:", len(indices))
        ds = tf.data.TFRecordDataset(tfrec_path)
        # Load all records from the file into memory
        recs = list(ds.as_numpy_iterator())
        
        # Process only the specified segment indices
        for ex_idx in indices:
            if ex_idx < 0 or ex_idx >= len(recs):
                print(f"  - Skipping idx={ex_idx} (out of bounds for file with {len(recs)} segments)"); continue
            
            # Parse the specific record
            ex = example_pb2.Example(); ex.ParseFromString(recs[ex_idx])
            base = os.path.basename(tfrec_path)
            
            # Call the main processing function
            png = draw_one_sample(ex, base, ex_idx, hv_writer=hv_writer, av_writer=av_writer)
            
            if png:
                print("  + saved:", os.path.basename(png))
            else:
                print("  + processed (PNG export disabled)")

    # Clean up
    for f in (hv_f, av_f):
        if f: f.close()

    # Generate final summary plots
    make_boxplots()
    print("✅ Done. CSV & plots saved in:", OUT_DIR)

if __name__ == "__main__":
    main()