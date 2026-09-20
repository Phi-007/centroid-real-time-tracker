<div align="center">

# 🎯 Centroid Real-Time Tracker

<img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/OpenCV-Video_IO-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
<img src="https://img.shields.io/badge/NumPy-Kalman_Filter-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/YOLOv10-Ultralytics-00FFFF?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Sort-Radix_(LSB)-brightgreen?style=for-the-badge"/>

### A lightweight, dependency-minimal multi-object tracker
### combining grid-indexed spatial search, 2D Kalman filtering, and a custom Radix Sort ranking step

[🎥 **Demo**](#-demo) • [🧠 **How It Works**](#-algorithm-overview) • [📂 **File Breakdown**](#-repository-structure--file-breakdown) • [🚀 **Quick Start**](#-quick-start-guide)

---

<a name="-demo"></a>




https://github.com/user-attachments/assets/ae2628c5-5e47-4372-9fa2-467159bc0a59




</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Algorithm Overview](#-algorithm-overview)
  - [Core Concept](#core-concept)
  - [Step-by-Step Execution Pipeline](#step-by-step-execution-pipeline)
- [Important Setup Prerequisites](#-important-setup-prerequisites)
- [Repository Structure & File Breakdown](#-repository-structure--file-breakdown)
  - [File Details](#file-details)
- [Quick Start Guide](#-quick-start-guide)
  - [1. Installation](#1-installation)
  - [2. Data & Model Preparation](#2-data--model-preparation)
  - [3. Running the Tracker](#3-running-the-tracker)
- [Configuration & Parameters](#-configuration--parameters)
- [Notes & Limitations](#-notes--limitations)

---

<a name="-overview"></a>
## 🔍 Overview

This project tackles **real-time multi-object tracking** on video, matching detections to
tracks frame-by-frame using **centroid distance** instead of IoU. It combines three
optimization strategies to stay fast without heavy dependencies:

<table>
<tr>
<td align="center">🗺️ <b>Grid Indexing</b></td>
<td align="center">📈 <b>Kalman Filter</b></td>
<td align="center">🔢 <b>Radix Sort</b></td>
</tr>
<tr>
<td>Localizes candidate search to 8 neighbor cells instead of scanning every track</td>
<td>Predicts motion frame-to-frame, surviving brief occlusion</td>
<td>Ranks match candidates by distance in linear time, no comparison sort</td>
</tr>
</table>

**Key highlights:**
- 🧩 **Detector-agnostic** — works with any bounding-box detector (YOLOv10 by default)
- ⚡ **O(neighborhood) association** via uniform spatial grid, not O(n·m)
- 🎥 **Kalman-predicted continuity** through short occlusions/missed detections
- 🖼️ **Built-in visualizer** — annotated grid, track IDs, and confirmed/predicted state, exported to MP4

---

<a name="-algorithm-overview"></a>
## 🧠 Algorithm Overview

### Core Concept

A centroid-based tracker is a simple yet effective multi-object tracking (MOT) technique that relies on the spatial proximity of detected object centers (centroids) between consecutive frames to maintain consistent track identities across time. It operates under the assumption that objects move gradually from frame to frame, meaning the nearest track centroid in the next frame most likely corresponds to the same physical object.

To maintain high efficiency and scalability, this algorithm enhances classic centroid association with three core optimization strategies:
1. **Grid Spatial Indexing ($S \times S$):** The image canvas is partitioned into uniform grid cells. Candidate searches are localized strictly to neighboring cells, reducing computational complexity from global $O(N \cdot M)$ checks down to fast local neighborhoods.
2. **Kalman Filter Motion Prediction:** A 2D constant velocity Kalman filter estimates object trajectories and predicts future centroids $(p_x, p_y)$, making tracking robust against linear motion and transient jitter.
3. **Radix Sort (LSD/LSB):** Candidate detection-track pairs are sorted by Euclidean distance in linear time using a precision-scaled Radix Sort rather than standard $O(K \log K)$ comparison sorts.

---

### Step-by-Step Execution Pipeline

```
[ Frame Input ]
       │
       ▼
1. Grid Partitioning & YOLOv10 Detection (cx, cy)
       │
       ▼
2. Kalman Filter Position Prediction for Active Tracks
       │
       ▼
3. 8-Neighbor Spatial Candidate Pair Extraction & Distance Gating
       │
       ▼
4. Precision Radix Sorting of Pair Distances (Ascending)
       │
       ▼
5. Greedy Detection-to-Track Assignment & State Update
       │
       ├─────────────────────────┬─────────────────────────┐
       ▼                         ▼                         ▼
6. Unassigned Detections  7. Assigned Tracks     8. Unassigned Tracks
   ➔ Create New Tracks       ➔ Update Kalman        ➔ Increment Missed Counter
                                                     ➔ Delete if > max_missed
```

1. **🗺️ Grid Partitioning:**
   - The frame of dimensions $W \times H$ is divided into a uniform grid of cell size $S \times S$.
   - Each cell receives a unique integer ID (`cell_id = row × width_cells + col`).
   - Detections from YOLOv10 are converted to centroid coordinates $(c_x, c_y)$ and mapped to their respective grid cell IDs.

2. **📈 Track State Prediction:**
   - For every active track $t$, the 2D Kalman filter predicts its state vector $[x, y, v_x, v_y]^T$.
   - The predicted centroid $(p_x, p_y)$ is saved and mapped into the grid to update the spatial index mapping tracks to cells.

3. **🔎 Neighbor Candidate Search & Gating:**
   - For each detection $d$, the algorithm identifies its cell and its **8 adjacent neighbor cells**.
   - Collect all candidate track IDs from these neighbor cells.
   - Euclidean distance $d(\text{det}, \text{track}) = \sqrt{(c_x - p_x)^2 + (c_y - p_y)^2}$ is computed.
   - If the distance is less than `max_distance` threshold, the $(\text{track} - \text{detection})$ pair is retained.

4. **🔢 Radix Sort Pair Ranking:**
   - Candidate pairs `(distance, det_idx, track_id)` are sorted in ascending order using **Radix Sort (LSB)**.
   - Floating-point distances are scaled to integer precision for stable counting-sort passes.
   - Closest detection-track matches are prioritized at the front of the list.

5. **✅ Greedy Match Assignment:**
   - Initialize empty sets `assigned_detections` and `assigned_tracks`.
   - Iterate through sorted candidate pairs `(distance, detection, track_id)`:
     - If detection is in `assigned_detections` or track ID is in `assigned_tracks` $\rightarrow$ skip.
     - Otherwise: assign detection $\rightarrow$ track, add detection to `assigned_detections`, add track to `assigned_tracks`, and update Kalman filter state.

6. **🆕 Create New Tracks for Unassigned Detections:**
   - For each detection not in `assigned_detections`:
     - Initialize Kalman filter for that detection.
     - Create new track ID and add to `assigned_tracks`.

7. **⏳ Handle Unassigned Tracks:**
   - For each active track not in `assigned_tracks`:
     - Predict next position using Kalman filter and increment `missed_counter`.
     - If `missed_counter` exceeds `max_missed` $\rightarrow$ delete/retire track.

---

<a name="-important-setup-prerequisites"></a>
## ⚠️ Important Setup Prerequisites

> ⚠️ **IMPORTANT NOTICE BEFORE RUNNING THE PROJECT**
>
> 1. **🎯 Retrain Object Detector:** The included `yolov10_detector.py` script references custom weights (`best.pt`). You **must train or supply your own YOLO model** fine-tuned on your target domain and update the weights file path in `main.py`.
> 2. **🖼️ Extract Target Frames:** The pipeline reads sequence inputs as individual image frames from a directory (e.g., `.jpg`, `.png`). If starting from a video file, you **must extract the video frames into a target folder** prior to executing `main.py`.

---

<a name="-repository-structure--file-breakdown"></a>
## 🗂️ Repository Structure & File Breakdown

All files live flat in the project root — no subfolders.

| File Name | Key Classes / Functions | Description |
| :--- | :--- | :--- |
| [`indexer.py`](#indexerpy) 🗺️ | `Indexer` | Handles spatial grid indexing, coordinate-to-cell mapping, and 8-neighbor lookup. |
| [`kalman_filter.py`](#kalman_filterpy) 📈 | `KalmanFilter` | Implements a 2D constant velocity linear Kalman Filter for motion prediction and state estimation. |
| [`radix_sort.py`](#radix_sortpy) 🔢 | `RadixSort`, `CountingSort` | Implements an LSD Radix Sort subroutine to stably rank candidate pair distances in linear time. |
| [`tracker.py`](#trackerpy) 🧭 | `Tracker`, `Track` | Encapsulates core MOT logic: spatial pairing, candidate sorting, greedy assignment, and track creation/pruning. |
| [`visualize.py`](#visualizepy) 🖼️ | `Visualize` | Renders grid lines, centroid indicators (green = updated, red = predicted), tracking IDs, and exports MP4 video. |
| [`yolov10_detector.py`](#yolov10_detectorpy) 🔍 | `YOLOv10Warpper` | Wrapper class for Ultralytics YOLOv10 object detection, bounding box extraction, and confidence filtering. |
| [`main.py`](#mainpy) 🎬 | `main()` | Main execution entry point. Loads image frames, invokes detection & tracking pipeline, and exports output video. |

---

### File Details

<a name="indexerpy"></a>
#### 🗺️ `indexer.py`
Provides spatial hashing to accelerate candidate association:
- **`point_to_cell(x, y)`**: Maps continuous coordinate points $(x, y)$ to discrete grid row/col indices and a unique scalar `cell_id` with boundary clamping.
- **`neighbors8(cell_id)`**: Calculates scalar IDs of the 8 surrounding cells (plus self), correctly handling image boundary limits.

<a name="kalman_filterpy"></a>
#### 📈 `kalman_filter.py`
Maintains kinematics for moving targets using state vector $\mathbf{x} = [x, y, v_x, v_y]^T$:
- **State Transition Matrix ($F$):** Applies constant velocity model with timestep $dt$.
- **`predict()`**: Projects track state forward into the next frame ($\mathbf{x} \leftarrow F\mathbf{x}$) and updates covariance $P$.
- **`update(meas_x, meas_y)`**: Corrects internal state vector using observed object detection coordinates $[z_x, z_y]^T$.

<a name="radix_sortpy"></a>
#### 🔢 `radix_sort.py`
Provides efficient distance sorting for association pairs:
- **`CountingSort.sort_by_digit()`**: Stable digit-wise counting sort subroutine.
- **`RadixSort.sort()`**: Scales floating-point distances by a precision factor ($10^{\text{precision}}$) into integers, executes digit passes across the base, and converts back to sorted floating-point pair tuples.

<a name="trackerpy"></a>
#### 🧭 `tracker.py`
Contains track state structures and assignment engine:
- **`Track` Data Class**: Stores track metadata including ID, assigned Kalman Filter, bounding box, missed counter, color code, and active/retired flags.
- **`Tracker.update()`**: Orchestrates spatial cell grouping, neighbor candidate extraction, gating by `max_distance`, Radix Sort ranking, greedy assignment, new track spawning, and missed track pruning.

<a name="visualizepy"></a>
#### 🖼️ `visualize.py`
Handles visual output formatting and video export:
- **`draw_grid()`**: Overlays spatial cell grid boundaries onto output image frames.
- **`draw()`**: Draws centroids (green circle for active detection update, red circle for prediction-only state), track ID, and class labels.
- **`write_frame()` & `release()`**: Manages OpenCV `VideoWriter` stream output.

<a name="yolov10_detectorpy"></a>
#### 🔍 `yolov10_detector.py`
Provides bounding box detection input:
- Loads custom weights (`best.pt`) via Ultralytics YOLO API.
- Converts raw bounding boxes $[x_1, y_1, x_2, y_2]$ to upper-left format $(x_1, y_1, w, h)$ with confidence filtering ($> 0.5$).

<a name="mainpy"></a>
#### 🎬 `main.py`
Combines all sub-modules into a unified processing loop:
- Ingests image frame sequences from a specified folder directory.
- Runs detection, tracking update, and frame visualization per timestep.
- Saves annotated results to an output MP4 file.

---

<a name="-quick-start-guide"></a>
## 🚀 Quick Start Guide

### 1. Installation

Clone the repository and install required Python packages:

```bash
git clone https://github.com/your-username/grid-centroid-tracker.git
cd grid-centroid-tracker

pip install numpy opencv-python ultralytics
```

---

### 2. Data & Model Preparation

1. **🖼️ Extract Video Frames:** Convert your input video file into a folder of sequential images:
   ```bash
   ffmpeg -i input_video.mp4 -q:v 2 input_frames/frame_%04d.png
   ```

2. **🧠 Detector Weights:** Place your fine-tuned YOLOv10 weights file (e.g., `best.pt`) in your project directory.

---

### 3. Running the Tracker

Modify the paths inside `main.py` to point to your frames directory and model weights:

```python
# Inside main.py
frames_folder = Path("path/to/input_frames")
output_video  = Path("path/to/tracker_output.mp4")
detector      = YOLOv10Warpper("path/to/best.pt")
```

Execute the tracking pipeline:

```bash
python main.py
```

---

<a name="-configuration--parameters"></a>
## 🎛️ Configuration & Parameters

Key hyperparameters can be adjusted in `main()` inside `main.py`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `cell_size` | `64` | Width/height in pixels of each grid cell ($S \times S$). |
| `max_distance` | `100.0` | Gating threshold distance (pixels) between predicted centroid and detection. |
| `max_missed` | `5` | Maximum consecutive unassigned frames before retiring a track. |
| `fps` | `25` | Output video frame rate. |

---

<a name="-notes--limitations"></a>
## ⚠️ Notes & Limitations

- 🎯 Matching is **centroid-distance based**, not IoU-based — works well for small/medium objects with steady frame-to-frame motion, less well for large overlapping boxes.
- 🔍 Detector currently filters to a single class (`class_id == 0`) with a fixed confidence threshold in `yolov10_detector.py`.
- 🆔 No re-identification — a track that retires and later reappears gets a new ID.

---

<div align="center">
Made with 🎯 for real-time tracking experiments | Give it a ⭐ if you found it useful!
</div>
