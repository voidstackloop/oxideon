"""Multi-object tracking primitives (Hungarian, Kalman, IoU/ByteTrack) for Vision."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ...oxideon import Tensor as _Tensor, from_numpy as _from_numpy


def _to_np(t: Any) -> np.ndarray:
    if isinstance(t, _Tensor):
        return t.to_numpy()
    return np.asarray(t, dtype=np.float32)


def linear_sum_assignment(cost: Any) -> Tuple[_Tensor, _Tensor]:
    """Solves the rectangular linear assignment problem (Hungarian method).

    Args:
        cost: 2-D finite cost matrix [N, M].

    Returns:
        (row_ind, col_ind) int64 Tensors pairing each row (if N <= M) or each
        column (if N > M) with a distinct counterpart at minimum total cost.
    """
    c = np.asarray(_to_np(cost), dtype=np.float64)
    if c.ndim != 2:
        raise ValueError(f"cost must be 2-D, got shape {c.shape}")
    n, m = c.shape
    if n == 0 or m == 0:
        raise ValueError("cost matrix must be non-empty")
    if not np.all(np.isfinite(c)):
        raise ValueError("cost matrix must be finite")
    transposed = False
    if n > m:
        c = c.T.copy()
        n, m = c.shape
        transposed = True
    # E-maxx O(n^2 m) potentials method, 1-indexed, requires n <= m.
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int64)
    way = np.zeros(m + 1, dtype=np.int64)
    a = np.zeros((n + 1, m + 1))
    a[1:, 1:] = c
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = a[i0, j] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[int(p[j])] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
    assign = np.zeros(n + 1, dtype=np.int64)
    for j in range(1, m + 1):
        if p[j]:
            assign[int(p[j])] = j
    rows = np.arange(n, dtype=np.int64)
    cols = assign[1:] - 1
    if transposed:
        rows, cols = cols, rows
        order = np.argsort(rows, kind="stable")
        rows, cols = rows[order], cols[order]
    return _from_numpy(np.ascontiguousarray(rows)), _from_numpy(np.ascontiguousarray(cols))


def _xyxy_to_z(box: np.ndarray) -> np.ndarray:
    w = max(float(box[2] - box[0]), 1e-6)
    h = max(float(box[3] - box[1]), 1e-6)
    cx = (float(box[0]) + float(box[2])) / 2.0
    cy = (float(box[1]) + float(box[3])) / 2.0
    return np.array([cx, cy, w * h, w / h], dtype=np.float64)


def _x_to_xyxy(x: np.ndarray) -> np.ndarray:
    cx, cy, s, r = (float(v) for v in x[:4])
    s = max(s, 1e-6)
    r = max(r, 1e-6)
    w = np.sqrt(s * r)
    h = s / w
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float64)


class KalmanBoxTracker:
    """Constant-velocity Kalman filter on [cx, cy, area, ratio] (SORT-style).

    Args:
        box: Initial [x1, y1, x2, y2] detection.
    """

    def __init__(self, box: Any) -> None:
        flat = np.asarray(_to_np(box), dtype=np.float64).reshape(-1)
        if flat.shape != (4,):
            raise ValueError(f"box must have 4 elements, got shape {flat.shape}")
        z = _xyxy_to_z(flat)
        if not np.all(np.isfinite(z)):
            raise ValueError("box must be finite")
        self.x = np.zeros(7, dtype=np.float64)
        self.x[:4] = z
        self.P = np.diag([10.0] * 4 + [10000.0] * 3)
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.zeros((4, 7))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.01])
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])
        self.hits = 1
        self.age = 0

    def predict(self) -> np.ndarray:
        """Advances the state and returns the predicted [x1, y1, x2, y2] box."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        return _x_to_xyxy(self.x)

    def update(self, box: Any) -> np.ndarray:
        """Corrects the state with a detection, returns corrected xyxy box."""
        flat = np.asarray(_to_np(box), dtype=np.float64).reshape(-1)
        if flat.shape != (4,):
            raise ValueError(f"box must have 4 elements, got shape {flat.shape}")
        z = _xyxy_to_z(flat)
        if not np.all(np.isfinite(z)):
            raise ValueError("box must be finite")
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.hits += 1
        self.age = 0
        return _x_to_xyxy(self.x)

    def get_state(self) -> np.ndarray:
        """Returns the current [x1, y1, x2, y2] estimate."""
        return _x_to_xyxy(self.x)


def _pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-12)


def _check_dets(dets: np.ndarray) -> np.ndarray:
    d = np.asarray(dets, dtype=np.float64)
    if d.ndim != 2 or d.shape[1] != 4:
        raise ValueError(f"detections must be [N, 4], got shape {d.shape}")
    if len(d) and not np.all(np.isfinite(d)):
        raise ValueError("detections must be finite")
    return d


class IoUTracker:
    """Greedy multi-object tracker with Hungarian IoU association.

    Args:
        iou_threshold: Minimum IoU for a track-detection match.
        max_age: Drop tracks unmatched for more than this many frames.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 1) -> None:
        if not 0.0 < float(iou_threshold) < 1.0:
            raise ValueError("iou_threshold must be in (0, 1)")
        if int(max_age) < 0:
            raise ValueError("max_age must be >= 0")
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.tracks: List[Dict[str, Any]] = []
        self._next_id = 0

    def update(self, detections: Any) -> List[Tuple[int, np.ndarray]]:
        """Associates detections and returns [(track_id, box)] matched frames."""
        dets = _check_dets(_to_np(detections))
        alive = [t for t in self.tracks if t["age"] <= self.max_age]
        matched_det = set()
        if alive and len(dets):
            boxes = np.stack([t["box"] for t in alive])
            iou = _pairwise_iou(boxes, dets)
            rows, cols = (v.to_numpy() for v in linear_sum_assignment(1.0 - iou))
            for r, c in zip(rows.tolist(), cols.tolist()):
                if iou[r, c] >= self.iou_threshold:
                    alive[r]["box"] = dets[c].copy()
                    alive[r]["age"] = 0
                    alive[r]["hits"] += 1
                    matched_det.add(c)
                else:
                    alive[r]["age"] += 1
            unmatched_tracks = [t for i, t in enumerate(alive) if i not in set(rows.tolist())]
            for t in unmatched_tracks:
                t["age"] += 1
        else:
            for t in alive:
                t["age"] += 1
        for c in range(len(dets)):
            if c not in matched_det:
                alive.append({"id": self._next_id, "box": dets[c].copy(), "age": 0, "hits": 1})
                self._next_id += 1
        self.tracks = [t for t in alive if t["age"] <= self.max_age]
        return [(t["id"], t["box"].copy()) for t in self.tracks if t["age"] == 0]


class ByteTracker:
    """Simplified two-stage ByteTrack: high-score then low-score association.

    Each track carries a :class:`KalmanBoxTracker`. Unmatched low-score
    detections are discarded (never spawn tracks); unmatched high-score
    detections spawn new tracks.

    Args:
        high_threshold: Detections at/above this score are "high".
        low_threshold: Detections below this score are ignored entirely.
        iou_threshold: Minimum IoU for association in either stage.
        max_age: Drop tracks unmatched for more than this many frames.
    """

    def __init__(
        self,
        high_threshold: float = 0.6,
        low_threshold: float = 0.1,
        iou_threshold: float = 0.3,
        max_age: int = 5,
    ) -> None:
        if not 0.0 <= float(low_threshold) < float(high_threshold) <= 1.0:
            raise ValueError("need 0 <= low_threshold < high_threshold <= 1")
        if not 0.0 < float(iou_threshold) < 1.0:
            raise ValueError("iou_threshold must be in (0, 1)")
        if int(max_age) < 0:
            raise ValueError("max_age must be >= 0")
        self.high_threshold = float(high_threshold)
        self.low_threshold = float(low_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.tracks: List[Dict[str, Any]] = []
        self._next_id = 0

    def _associate(
        self, tracks: List[Dict[str, Any]], dets: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not tracks or not len(dets):
            return [], list(range(len(tracks))), list(range(len(dets)))
        boxes = np.stack([t["kf"].get_state() for t in tracks])
        iou = _pairwise_iou(boxes, dets)
        rows, cols = (v.to_numpy() for v in linear_sum_assignment(1.0 - iou))
        matches, used_t, used_d = [], set(), set()
        for r, c in zip(rows.tolist(), cols.tolist()):
            if iou[r, c] >= self.iou_threshold:
                matches.append((r, c))
                used_t.add(r)
                used_d.add(c)
        return (
            matches,
            [i for i in range(len(tracks)) if i not in used_t],
            [i for i in range(len(dets)) if i not in used_d],
        )

    def update(self, detections: Any, scores: Any) -> List[Tuple[int, np.ndarray]]:
        """Associates one frame; returns [(track_id, box)] for live tracks."""
        dets = _check_dets(_to_np(detections))
        sc = np.asarray(_to_np(scores), dtype=np.float64).reshape(-1)
        if len(sc) != len(dets):
            raise ValueError("scores length must match detections")
        if len(sc) and (not np.all(np.isfinite(sc)) or np.any(sc < 0) or np.any(sc > 1)):
            raise ValueError("scores must be finite values in [0, 1]")
        for t in self.tracks:
            t["kf"].predict()
        keep = sc >= self.low_threshold
        dets, sc = dets[keep], sc[keep]
        high = dets[sc >= self.high_threshold]
        low = dets[sc < self.high_threshold]
        matches, unmatched_t, unmatched_d = self._associate(self.tracks, high)
        for r, c in matches:
            self.tracks[r]["kf"].update(high[c])
            self.tracks[r]["age"] = 0
            self.tracks[r]["hits"] += 1
        pending = [self.tracks[i] for i in unmatched_t]
        matches2, unmatched_t2, _ = self._associate(pending, low)
        for r, c in matches2:
            pending[r]["kf"].update(low[c])
            pending[r]["age"] = 0
            pending[r]["hits"] += 1
        for i in unmatched_t2:
            pending[i]["age"] += 1
        for c in unmatched_d:
            self.tracks.append({
                "id": self._next_id,
                "kf": KalmanBoxTracker(high[c]),
                "age": 0,
                "hits": 1,
            })
            self._next_id += 1
        self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]
        return [(t["id"], t["kf"].get_state().copy()) for t in self.tracks if t["age"] == 0]


__all__ = ["linear_sum_assignment", "KalmanBoxTracker", "IoUTracker", "ByteTracker"]
