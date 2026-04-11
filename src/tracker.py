import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class TrackState:
    track_id: int
    center: Tuple[float, float]
    last_center: Tuple[float, float]
    missed_frames: int = 0


class CentroidTracker:
    """A lightweight tracker that assigns stable IDs and speed estimates."""

    def __init__(self, max_missed_frames: int = 12, match_distance: float = 80.0):
        self.max_missed_frames = max_missed_frames
        self.match_distance = match_distance
        self._next_id = 1
        self._tracks: List[TrackState] = []

    @staticmethod
    def _center(box: List[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _build_candidates(self, centers: List[Tuple[float, float]]):
        candidates = []
        for t_idx, track in enumerate(self._tracks):
            for c_idx, center in enumerate(centers):
                dist = self._distance(track.center, center)
                if dist <= self.match_distance:
                    candidates.append((dist, t_idx, c_idx))
        candidates.sort(key=lambda x: x[0])
        return candidates

    def update(self, boxes: List[List[float]]):
        """
        Input: list of [x1, y1, x2, y2]
        Output: list of dicts with track_id and speed for each input box index.
        """
        if not boxes:
            for track in self._tracks:
                track.missed_frames += 1
            self._tracks = [t for t in self._tracks if t.missed_frames <= self.max_missed_frames]
            return []

        centers = [self._center(b) for b in boxes]
        candidates = self._build_candidates(centers)

        assigned_tracks = set()
        assigned_boxes = set()
        track_to_box = {}

        for _, t_idx, c_idx in candidates:
            if t_idx in assigned_tracks or c_idx in assigned_boxes:
                continue
            assigned_tracks.add(t_idx)
            assigned_boxes.add(c_idx)
            track_to_box[t_idx] = c_idx

        for t_idx, track in enumerate(self._tracks):
            if t_idx in track_to_box:
                c_idx = track_to_box[t_idx]
                track.last_center = track.center
                track.center = centers[c_idx]
                track.missed_frames = 0
            else:
                track.missed_frames += 1

        self._tracks = [t for t in self._tracks if t.missed_frames <= self.max_missed_frames]

        for c_idx, center in enumerate(centers):
            if c_idx in assigned_boxes:
                continue
            new_track = TrackState(
                track_id=self._next_id,
                center=center,
                last_center=center,
                missed_frames=0,
            )
            self._next_id += 1
            self._tracks.append(new_track)

        outputs = []
        for center in centers:
            best_track = min(self._tracks, key=lambda t: self._distance(t.center, center))
            speed = self._distance(best_track.center, best_track.last_center)
            outputs.append({"track_id": best_track.track_id, "speed": speed})

        return outputs
