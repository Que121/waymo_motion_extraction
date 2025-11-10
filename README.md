# waymo_motion_extraction# Waymo Motion Dataset Extraction and Visualization Tool

This repository provides a comprehensive Python script for extracting, analyzing, and visualizing trajectory and roadgraph data from the **Waymo Open Motion Dataset (WOMD)**.  
It allows you to compute geometric and behavioral metrics (e.g., lane-center offset, stop distance), export per-frame trajectory CSVs for AVs/HVs, and generate scene visualizations and boxplots for downstream analysis.

---

## 📦 Overview

The main script `waymo_motion_extract_en_commented.py` performs the following operations:

1. **Load dataset configuration** from a JSON manifest (`summary.json`).
2. **Parse TFRecord files** from the Waymo Motion Dataset.
3. **Extract agent trajectories** (past, current, future) and roadgraph features (lane centers, stop signs).
4. **Transform coordinates** into AV-local reference frame.
5. **Compute metrics:**
   - Lane-center offset (signed distance from trajectory to nearest lane centerline).
   - Distance to stop sign (distance at minimal speed within a given radius).
6. **Export CSVs:** detailed frame-wise trajectory data for human-driven (HV) and autonomous (AV) vehicles.
7. **Visualize scenes:** render local map with lane geometry, AV/HV trajectories, and velocity arrows.
8. **Aggregate statistics:** boxplots summarizing lane offset and stop distance distributions.

---

## 🗂 Directory Structure

```plaintext
project_root/
│
├── waymo_motion_extract_en_commented.py   # Main extraction script (English commented)
├── summary.json                           # Job manifest listing TFRecord files and indices
├── extract_out/                           # Output folder for CSVs and visualizations
│   ├── hv_trajectories.csv                # Human-driven vehicle trajectories
│   ├── av_trajectories.csv                # Autonomous vehicle trajectories
│   ├── *_exXXXXX.png                      # Scene plots (optional)
│   ├── lane_center_offset_boxplot.png     # Boxplot for lane offset
│   └── dist_to_stop_boxplot.png           # Boxplot for stop distance
└── README.md                              # This document
```

# ⚙️ Configuration Guide

All key configuration parameters for `waymo_motion_extract_en_commented.py`  
are defined at the top of the script. Modify them as needed before running.

| Parameter | Description | Example |
|------------|--------------|----------|
| `JSON_PATH` | Path to the job manifest JSON file containing TFRecord file names and indices | `./summary.json` |
| `TFREC_DIR` | Directory containing `.tfrecord` files from the Waymo Motion Dataset | `/mnt/home/Files/Lab/waymo_motion/dataset` |
| `OUT_DIR` | Directory for CSV and plot outputs | `./extract_out` |
| `USE_LOCAL_FRAME` | Convert all coordinates to the **AV-centric local coordinate frame** (x-forward, y-left) | `True` |
| `ONLY_VEHICLES` | Filter and process **only vehicle agents** (ignore pedestrians/cyclists) | `True` |
| `CALC_LANE_CENTER_OFFSET` | Compute signed lateral distance to the **nearest lane centerline** | `True` |
| `CALC_STOP_DISTANCE` | Compute **distance to stop sign** when vehicle speed is minimal | `False` |
| `PLOT_LANE_CENTER_OFFSET` | Generate **boxplot visualization** for lane offset | `True` |
| `PLOT_STOP_DISTANCE` | Generate **boxplot visualization** for stop distance | `False` |

---

## 🧭 Additional Notes

- **Local frame transform:**  
  When `USE_LOCAL_FRAME=True`, all positions are rotated and translated so that  
  the AV’s current position is the origin (0, 0) and its heading defines the x-axis.

- **Stop sign point reduction:**  
  Each group of sampled stop points (from `roadgraph_samples/type=17`)  
  is reduced to a single representative point using mean or median.

- **TFRecord indexing:**  
  The script automatically scans `TFREC_DIR` to resolve basenames listed in `JSON_PATH`.  
  It supports both absolute and relative file references.

- **Output management:**  
  The directory `OUT_DIR` will be created automatically if it doesn’t exist.  
  All generated CSVs, PNGs, and boxplots will be saved there.

---

## 🧩 Example Setup (Lab Environment)

```bash
# Example configuration for local lab environment
JSON_PATH=./summary.json
TFREC_DIR=/mnt/home/Files/Lab/waymo_motion/dataset
OUT_DIR=./extract_out

USE_LOCAL_FRAME=True
ONLY_VEHICLES=True
CALC_LANE_CENTER_OFFSET=True
CALC_STOP_DISTANCE=False
PLOT_LANE_CENTER_OFFSET=True
PLOT_STOP_DISTANCE=False
```

# 🚀 Usage Guide

This section describes how to run the **Waymo Motion Dataset Extraction and Visualization Tool**,  
including environment setup, data preparation, and expected outputs.

---

## Prerequisites

Before running the script, ensure your environment meets the following requirements:

### 🧰 Required Dependencies
- **Python** ≥ 3.8  
- **TensorFlow** ≥ 2.8  
- **NumPy**  
- **Matplotlib**

Install dependencies using:

```bash
pip install tensorflow matplotlib numpy
```

# 📊 Metrics Description

This section details the key quantitative metrics computed by  
`waymo_motion_extract_en_commented.py` for trajectory analysis.

---

## 🛣 1. Lane Center Offset

**Definition:**  
The **signed lateral distance** between an agent’s position and the **nearest lane centerline**.

| Symbol | Description |
|---------|-------------|
| `d_lane` | Signed distance from vehicle position to lane centerline (meters) |
| `+` | Vehicle is to the **left** of the lane centerline (relative to its heading) |
| `–` | Vehicle is to the **right** of the lane centerline |

**Computation Process:**
1. The lane polylines are reconstructed from `roadgraph_samples/id` and transformed to the AV-local frame.
2. For each agent, its heading vector is estimated using forward/backward differences in position.
3. The nearest left and right lane boundaries (within 3.7 m) are located.
4. The lane centerline is estimated as the midpoint between them.
5. The signed offset `d_lane = (P - C) ⋅ n`  
   where `P` is the vehicle position, `C` is the lane-center point, and `n` is the lane normal.

**Interpretation:**
- Small |d_lane| → good lane-keeping accuracy  
- Large |d_lane| → lateral drift or off-lane behavior  
- Useful for **lane adherence evaluation**, **control stability**, and **autonomous navigation diagnostics**

---

## 🚦 2. Distance to Stop Sign

**Definition:**  
The **distance from a vehicle’s position to the nearest stop sign point**  
when its instantaneous speed is minimal within a given radius (`STOP_SPEED_RADIUS_M`).

**Computation Steps:**
1. Identify all stop sign points from `roadgraph_samples/type=17`.  
2. Reduce each group of sampled points (same ID) to a single representative point (mean/median).  
3. Convert to AV-local frame and select the one closest to the AV.
4. For each agent:
   - Within a radius (default **5 m**) of the stop sign,
   - Find the frame where speed is minimal,
   - Record the corresponding distance as `d_stop`.

**Parameters:**
| Variable | Description | Default |
|-----------|--------------|----------|
| `STOP_SPEED_RADIUS_M` | Search radius around stop point | `5.0` m |
| `STOP_POINT_REDUCER` | Aggregation method for stop samples | `"mean"` |

**Interpretation:**
- Indicates **how closely vehicles comply with stop signs**.  
- Useful for analyzing **stopping behavior**, **policy adherence**, and **driver intent prediction**.

---

## 🧮 3. Visualization Metrics

When plotting (if `EXPORT_SCENE_PNG=True`):

- **Lanes (gray)** → `roadgraph_samples/type ∈ {0,1,2,3,...,16}`  
- **Stop signs (red dots)** → `type = 17`  
- **AV trajectory (orange)** → `is_sdc=1`  
- **HV trajectories (blue)** → all others  
- **Velocity arrows** → current motion direction scaled by speed magnitude

---

> ⚙️ All metrics are computed in the **AV-local coordinate system**  
> (origin = AV current pose; x-forward, y-left) to ensure spatial consistency across scenarios.


# ✍️ Author and Acknowledgements

## 👤 Author

**Haohua Que (Quinton Que)**  
Affiliations: **University of Georgia (UGA)**  
Research Focus: **Autonomous Driving, BEV Perception, Robotics Exploration, Agricultural AI**  
Email: *[hq10606@uga.edu]*  
---

## 🧾 Citation

If you use or extend this tool in academic work, please cite it as:

```bibtex
@misc{que2025waymoextract,
  title     = {Waymo Motion Dataset Extraction and Visualization Tool},
  author    = {Haohua Que},
  year      = {2025},
  note      = {Available via Mobolity Lab internal repository},
  howpublished = {\url{https://github.com/haohuaque/waymo-motion-extract}}
}
```