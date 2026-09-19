<div align="center">

# 🎯 Centroid Real-Time Tracker

**A lightweight, dependency-minimal multi-object tracker** — grid-indexed spatial search, per-object Kalman filtering, and a custom radix-sort greedy assignment. Built to sit on top of any bounding-box detector and track objects across video in real time.

</div>

<div align="center">

🎥 **[Watch the demo →](#demo)**

</div>

---

## 📑 Table of Contents

| Section | What's there |
|---|---|
| [🎥 Demo](#demo) | Output video preview |
| [🧠 How It Works](#how-it-works) | The predict → associate → update pipeline, step by step |
| [📂 File Organization](#file-organization) | What each file does, and how they call into each other |
| [⚙️ Requirements](#requirements) | Dependencies & install |
| [🚀 Usage](#usage) | How to run it on your own video |
| [🎛️ Tuning](#tuning) | Key parameters and what they control |
| [⚠️ Notes & Limitations](#notes--limitations) | What this tracker does and doesn't handle |

---

## 🎥 Demo

---

## 🧠 How It Works

Every frame runs through a **predict → associate → update** loop, matching by centroid distance rather than IoU:

| Step | What happens |
|---|---|
| **1. Detect** | The detector (YOLOv10 by default) returns boxes `(x, y, w, h, class_id)` for the current frame |
| **2. Predict** | Every active track's Kalman filter predicts its next position from a constant-velocity motion model |
| **3. Spatial index** | Predicted positions and new detections are dropped into a uniform grid — each detection only checks tracks in its own cell + 8 neighbors, avoiding an O(n·m) comparison |
| **4. Candidates** | For each detection, nearby tracks become match candidates, scored by Euclidean distance and filtered by `max_distance` |
| **5. Sort** | Candidate `(distance, detection, track)` pairs are ordered ascending using a **custom radix sort** (floats scaled to ints, bucketed by digit) |
| **6. Assign** | Pairs are consumed closest-first, greedily — each detection/track is used at most once |
| **7. Lifecycle** | Matched tracks update; unmatched detections spawn new tracks; unmatched tracks accumulate misses and retire after `max_missed` frames |
| **8. Visualize** | Each frame is annotated (grid + track dots: 🟢 detector-confirmed, 🔴 prediction-only) and written to the output video |

This gives a fast, detector-agnostic tracker that survives brief occlusion via Kalman prediction — no IoU matching or appearance embeddings needed.

---

## 📂 File Organization

All files live flat in the project root — no subfolders. Here's what each one is responsible for:

| File | Role |
|---|---|
| `main.py` | 🎬 Entry point — wires detector → tracker → visualizer together and runs the per-frame loop |
| `yolov10_detector.py` | 🔍 Detector wrapper — loads a YOLOv10 model, returns `(x, y, w, h, class_id)` boxes |
| `indexer.py` | 🗺️ Spatial grid — maps points to cells, finds 8-neighbor cells for fast candidate lookup |
| `tracker.py` | 🧭 Core tracking logic — owns all `Track` objects, runs predict/associate/update per frame |
| `kalman_filter.py` | 📈 Per-track motion model — predicts and corrects each object's position |
| `radix_sort.py` | 🔢 Custom sort — orders match candidates by distance without relying on built-in sort |
| `visualize.py` | 🖼️ Drawing + video writer — renders grid, track dots, and labels onto each frame |

**How they connect, per frame:**

```
frames folder
      │
      ▼
yolov10_detector.py  ──▶  detections (x, y, w, h, class_id)
      │
      ▼
tracker.py  ── update(detections, frame_idx) ──┐
      │                                        │
      ├── indexer.py       (bucket predictions + detections into grid cells)
      ├── kalman_filter.py (predict/update each Track's position — one instance per track)
      └── radix_sort.py    (order match candidates by distance)
      │
      ▼
(dets_info, tracks_vis)
      │
      ▼
visualize.py  ──▶  draw() ──▶ write_frame()  ──▶  output.mp4
```

`main.py` is the only file that touches every module — `tracker.py` never talks to the detector or video writer directly, keeping tracking logic decoupled from I/O.

---

## ⚙️ Requirements

```bash
pip install opencv-python numpy ultralytics
```

---

## 🚀 Usage

> ⚠️ Before running, you need **(1)** a folder of extracted video frames and **(2)** trained YOLOv10 weights. This repo tracks — it doesn't extract frames or train a detector.

**1. Extract frames from your video**

```bash
ffmpeg -i input_video.mp4 frames_folder/frame_%05d.jpg
```

**2. Point `main.py` at your frames and weights**

```python
frames_folder = Path("/path/to/frames_folder")
output_video  = Path("output.mp4")
detector      = YOLOv10Warpper("/path/to/yolov10_model.pt")
```

**3. Run**

```bash
python main.py
```

The annotated output video is written to `output_video`, with each track's ID, class, and confirmed/predicted state drawn on top of the grid.

---

## 🎛️ Tuning

| Parameter | Where | Effect |
|---|---|---|
| `cell_size` | `Indexer` | Grid cell size in px — smaller = finer buckets, more overhead |
| `max_distance` | `Tracker` | Max centroid distance to consider a detection/track a valid match |
| `max_missed` | `Tracker` | Frames a track can go unmatched before retiring |
| `process_var` / `meas_var` | `KalmanFilter` | Trust in motion model vs. trust in raw detections |

---

## ⚠️ Notes & Limitations

- Matching is **centroid-distance based**, not IoU-based — works well for small/medium objects with steady frame-to-frame motion, less well for large overlapping boxes.
- Detector currently filters to a single class (`class_id == 0`) with a fixed confidence threshold in `yolov10_detector.py`.
- No re-identification — a track that retires and later reappears gets a new ID.
