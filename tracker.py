import math
import random
from typing import List, Tuple, Dict
from dataclasses import dataclass, field
# import time
# from collections import deque

from indexer import Indexer
from radix_sort import RadixSort
from kalman_filter import KalmanFilter

# ---------------------------
# Track dataclass
# ---------------------------
@dataclass
class Track:
    id: int
    kf: KalmanFilter
    class_id: int
    last_box: Tuple[int, int, int, int]
    last_seen_frame: int
    missed_count: int = 0
    active: bool = True
    retired: bool = False
    age: int = 0
    color: Tuple[int, int, int] = field(default_factory=lambda: (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    ))

    #confidence: float = 1.0

    def predict(self):
        return self.kf.predict()
        #px, py = self.kf.predict()
        #return (px + random.uniform(-0.5, 0.5), py + random.uniform(-0.5, 0.5))

    def update(self, cx: float, cy: float, box: Tuple[int,int,int,int], frame_idx: int):
        self.kf.update(cx, cy)
        self.last_box = box
        self.last_seen_frame = frame_idx
        self.missed_count = 0
        self.age += 1

    def mark_missed(self):
        self.missed_count += 1
        self.age += 1


# ---------------------------
# Tracker: modified version
# ---------------------------
class Tracker:
    def __init__(self, indexer: Indexer, max_distance: float = 80.0, max_missed: int = 5):
        self.indexer = indexer
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

    def _predict_all(self): #
        preds = {}
        for tid, track in list(self.tracks.items()):
            if track.active and not track.retired:
                px, py = track.predict()
                preds[tid] = (px, py)
        return preds

    def _tracks_by_cell(self, preds: Dict[int, Tuple[float,float]]) -> Dict[int, List[int]]:
        mapping = {}
        for tid, (px, py) in preds.items():
            _, _, cid = self.indexer.point_to_cell(px, py)
            mapping.setdefault(cid, []).append(tid)
        return mapping

    def update(self, detections: List[Tuple[int,int,int,int,int]], frame_idx: int):
        """
        detections: list of (x, y, w, h, class_id)
        """
        dets = []
        for (x, y, w, h, cls) in detections:
            cx = x + w / 2.0
            cy = y + h / 2.0
            dets.append({
                'box': (x, y, w, h),
                'cx': cx,
                'cy': cy,
                'class_id': cls,
                'matched': False
            })

        preds = self._predict_all()
        preds_by_cell = self._tracks_by_cell(preds)

        candidate_pairs = []
        radix_sorter = RadixSort(base=10, precision=4)  
        for di, det in enumerate(dets):
            _, _, det_cell = self.indexer.point_to_cell(det['cx'], det['cy'])
            neighbor_cells = self.indexer.neighbors8(det_cell, include_self=True)
            
            candidates = []
            for nc in neighbor_cells:
                candidates.extend(preds_by_cell.get(nc, []))
            candidates = list(set(candidates))
            
            for tid in candidates:
                px, py = preds[tid]

                dist = math.hypot(det['cx']-px, det['cy']-py)

                if dist <= self.max_distance:
                    candidate_pairs.append((dist, di, tid))

        sorted_pairs = radix_sorter.sort(candidate_pairs)
        used_dets, used_tracks = set(), set()

        for dist, di, tid in sorted_pairs:
            if di in used_dets or tid in used_tracks:
                continue
            # Assign
            dets[di]['matched'] = True
            used_dets.add(di)
            used_tracks.add(tid)

            det = dets[di]
            self.tracks[tid].update(det['cx'], det['cy'], det['box'], frame_idx)
            self.tracks[tid].class_id = det['class_id']
        for di, det in enumerate(dets):
            if not det['matched']:
                new_kf = KalmanFilter(det['cx'], det['cy'])
                tid = self.next_id
                self.next_id += 1

                new_track = Track(
                    id=tid,
                    kf=new_kf,
                    class_id=det['class_id'],
                    last_box=det['box'],
                    last_seen_frame=frame_idx # 0 it was di
                )
                new_track.kf.update(det['cx'], det['cy'])
                self.tracks[tid] = new_track

        for tid, track in list(self.tracks.items()):
            if tid in used_tracks: # not in
                continue
            
            if not track.active or track.retired:
                continue
            
            track.mark_missed()
            if track.missed_count >= self.max_missed:
                track.active = False
                track.retired = True

        current_tracks = []
        for tid, track in self.tracks.items():
            if track.retired:
                continue
            px, py = track.kf.get_position()
            updated = (track.last_seen_frame == frame_idx and track.missed_count == 0)
            current_tracks.append({
                'id': tid,
                'cx': px,
                'cy': py,
                'updated': updated,
                'color': track.color,
                'class_id': track.class_id,
                'missed': track.missed_count,
                'box': track.last_box
            })

        return dets, current_tracks