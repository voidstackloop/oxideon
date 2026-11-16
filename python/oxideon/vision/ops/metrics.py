"""Detection / segmentation / pose evaluation metrics for Oxideon Vision."""

from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from ...oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .boxes import box_iou


def _to_np(t: Any) -> np.ndarray:
    if isinstance(t, _Tensor):
        return t.to_numpy()
    return np.asarray(t, dtype=np.float32)


def _check_boxes(b: np.ndarray, name: str) -> np.ndarray:
    b = np.asarray(b, dtype=np.float32)
    if b.ndim != 2 or b.shape[1] != 4:
        raise ValueError(f"{name} must be [N, 4], got shape {b.shape}")
    if b.shape[0] and not np.all(np.isfinite(b)):
        raise ValueError(f"{name} must be finite")
    return b


def _ap_101(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float,
) -> float:
    order = np.argsort(-pred_scores, kind="stable")
    pred_boxes = pred_boxes[order]
    matched = np.zeros(len(gt_boxes), dtype=bool)
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    if len(gt_boxes):
        ious = box_iou(pred_boxes, gt_boxes).to_numpy()
        for i in range(len(pred_boxes)):
            row = ious[i].copy()
            row[matched] = -1.0
            j = int(np.argmax(row))
            if row[j] >= iou_threshold:
                tp[i] = 1.0
                matched[j] = True
            else:
                fp[i] = 1.0
    else:
        fp[:] = 1.0
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    recalls = tp_c / max(len(gt_boxes), 1)
    precisions = tp_c / np.maximum(tp_c + fp_c, 1e-12)
    # Monotone envelope from the right, then COCO 101-point interpolation.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    thresholds = np.arange(0.0, 1.001, 0.01)
    ap = 0.0
    for t in thresholds:
        mask = recalls >= t
        ap += float(np.max(precisions[mask])) if np.any(mask) else 0.0
    return ap / len(thresholds)


def mean_average_precision(
    pred_boxes: Any,
    pred_scores: Any,
    pred_classes: Any,
    gt_boxes: Any,
    gt_classes: Any,
    iou_thresholds: Sequence[float] = (0.5,),
) -> Dict[str, Any]:
    """COCO-style mean average precision over classes and IoU thresholds.

    Classes without ground truth are ignored (as in pycocotools). Each AP uses
    the 101-point interpolated precision-recall curve.

    Returns:
        Dict with "map" (mean over thresholds and GT classes), "aps"
        (per-class AP averaged over thresholds), and "thresholds" echo.
    """
    pb = _check_boxes(_to_np(pred_boxes), "pred_boxes")
    gb = _check_boxes(_to_np(gt_boxes), "gt_boxes")
    ps = np.asarray(_to_np(pred_scores), dtype=np.float32).reshape(-1)
    pc = np.asarray(_to_np(pred_classes)).astype(np.int64).reshape(-1)
    gc = np.asarray(_to_np(gt_classes)).astype(np.int64).reshape(-1)
    if len(pb) != len(ps) or len(pb) != len(pc):
        raise ValueError("pred_boxes/pred_scores/pred_classes lengths disagree")
    if len(gb) != len(gc):
        raise ValueError("gt_boxes/gt_classes lengths disagree")
    if len(ps) and not np.all(np.isfinite(ps)):
        raise ValueError("pred_scores must be finite")
    thresholds = [float(t) for t in iou_thresholds]
    if not thresholds or any(not 0.0 < t < 1.0 for t in thresholds):
        raise ValueError("iou_thresholds must be non-empty values in (0, 1)")
    if np.any(pc < 0) or np.any(gc < 0):
        raise ValueError("class ids must be non-negative")

    aps: Dict[int, float] = {}
    for cls in sorted(set(gc.tolist())):
        pm = pc == cls
        gm = gc == cls
        ap = float(np.mean([
            _ap_101(pb[pm], ps[pm], gb[gm], t) for t in thresholds
        ]))
        aps[int(cls)] = ap
    mean_ap = float(np.mean(list(aps.values()))) if aps else 0.0
    return {"map": mean_ap, "aps": aps, "thresholds": thresholds}


def mask_iou(pred_masks: Any, gt_masks: Any) -> _Tensor:
    """Pairwise mask IoU between [N, H, W] and [M, H, W] boolean masks."""
    p = np.asarray(_to_np(pred_masks)).astype(bool)
    g = np.asarray(_to_np(gt_masks)).astype(bool)
    if p.ndim != 3 or g.ndim != 3:
        raise ValueError("masks must be [N, H, W] and [M, H, W]")
    if p.shape[1:] != g.shape[1:]:
        raise ValueError(f"mask spatial shapes disagree: {p.shape} vs {g.shape}")
    inter = np.logical_and(p[:, None], g[None, :]).sum(axis=(2, 3)).astype(np.float32)
    union = np.logical_or(p[:, None], g[None, :]).sum(axis=(2, 3)).astype(np.float32)
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return _from_numpy(np.ascontiguousarray(iou, dtype=np.float32))


def panoptic_quality(
    pred_masks: Any,
    gt_masks: Any,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Single-class panoptic quality with greedy IoU > 0.5 matching.

    Returns dict with "pq", "sq" (segmentation quality), "rq" (recognition
    quality), "tp", "fp", "fn". With no predictions and no ground truth,
    PQ is defined as 1.0.
    """
    p = np.asarray(_to_np(pred_masks)).astype(bool)
    g = np.asarray(_to_np(gt_masks)).astype(bool)
    if p.ndim != 3 or g.ndim != 3:
        raise ValueError("masks must be [N, H, W] and [M, H, W]")
    if p.shape[1:] != g.shape[1:]:
        raise ValueError(f"mask spatial shapes disagree: {p.shape} vs {g.shape}")
    if not 0.0 < float(iou_threshold) < 1.0:
        raise ValueError("iou_threshold must be in (0, 1)")
    if len(p) == 0 and len(g) == 0:
        return {"pq": 1.0, "sq": 1.0, "rq": 1.0, "tp": 0.0, "fp": 0.0, "fn": 0.0}
    iou = mask_iou(p, g).to_numpy() if len(p) and len(g) else np.zeros((len(p), len(g)))
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    used_p, used_g = set(), set()
    matched_sum = 0.0
    n_tp = 0
    for i, j in order:
        i, j = int(i), int(j)
        if iou[i, j] <= iou_threshold or i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matched_sum += float(iou[i, j])
        n_tp += 1
    n_fp = len(p) - n_tp
    n_fn = len(g) - n_tp
    denom = n_tp + 0.5 * n_fp + 0.5 * n_fn
    pq = matched_sum / denom if denom else 1.0
    return {
        "pq": pq,
        "sq": matched_sum / n_tp if n_tp else 0.0,
        "rq": n_tp / denom if denom else 1.0,
        "tp": float(n_tp),
        "fp": float(n_fp),
        "fn": float(n_fn),
    }


def oks(
    pred_kpts: Any,
    gt_kpts: Any,
    sigmas: Any,
    area: float,
    visible: Optional[Any] = None,
) -> float:
    """Object Keypoint Similarity: mean exp(-d^2 / (2 s^2 k^2)) over keypoints.

    Args:
        pred_kpts / gt_kpts: [K, 2] (x, y) coordinates.
        sigmas: [K] per-keypoint falloff constants (> 0).
        area: Object segment area s^2 (> 0).
        visible: Optional [K] visibility mask; invisible keypoints are skipped.
    """
    p = np.asarray(_to_np(pred_kpts), dtype=np.float64)
    g = np.asarray(_to_np(gt_kpts), dtype=np.float64)
    s = np.asarray(_to_np(sigmas), dtype=np.float64).reshape(-1)
    if p.ndim != 2 or p.shape[1] != 2 or g.shape != p.shape:
        raise ValueError("keypoints must be [K, 2] with matching shapes")
    if s.shape != (p.shape[0],) or np.any(s <= 0) or not np.all(np.isfinite(s)):
        raise ValueError("sigmas must be finite positive with length K")
    if not np.isfinite(area) or area <= 0:
        raise ValueError("area must be finite and positive")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(g)):
        raise ValueError("keypoints must be finite")
    if visible is None:
        mask = np.ones(p.shape[0], dtype=bool)
    else:
        mask = np.asarray(_to_np(visible)).astype(bool).reshape(-1)
        if mask.shape != (p.shape[0],):
            raise ValueError("visible must have length K")
    if not np.any(mask):
        return 0.0
    d2 = np.sum((p - g) ** 2, axis=1)
    return float(np.mean(np.exp(-d2[mask] / (2.0 * area * s[mask] ** 2))))


def pck(pred_kpts: Any, gt_kpts: Any, threshold: float, scale: float = 1.0) -> float:
    """Fraction of keypoints with ||pred - gt|| / scale <= threshold."""
    p = np.asarray(_to_np(pred_kpts), dtype=np.float64)
    g = np.asarray(_to_np(gt_kpts), dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or g.shape != p.shape:
        raise ValueError("keypoints must be [K, 2] with matching shapes")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(g)):
        raise ValueError("keypoints must be finite")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if p.shape[0] == 0:
        return 0.0
    dist = np.sqrt(np.sum((p - g) ** 2, axis=1)) / scale
    return float(np.mean(dist <= threshold))


def heatmap_to_keypoints(heatmaps: Any) -> _Tensor:
    """Argmax keypoints [K, 2] (x, y) with quarter-pixel neighbor refinement.

    Each coordinate is nudged by 0.25px toward the higher of its two neighbors
    (standard sub-pixel trick from human-pose estimation).
    """
    h = np.asarray(_to_np(heatmaps), dtype=np.float64)
    if h.ndim != 3:
        raise ValueError(f"heatmaps must be [K, H, W], got shape {h.shape}")
    k, hh, ww = h.shape
    if hh < 2 or ww < 2:
        raise ValueError("heatmaps must be at least 2x2")
    if not np.all(np.isfinite(h)):
        raise ValueError("heatmaps must be finite")
    flat = h.reshape(k, -1)
    yi = (np.argmax(flat, axis=1) // ww).astype(np.int64)
    xi = (np.argmax(flat, axis=1) % ww).astype(np.int64)
    pts = np.stack([xi, yi], axis=1).astype(np.float64)
    for i in range(k):
        x, y = int(xi[i]), int(yi[i])
        if 0 < x < ww - 1:
            pts[i, 0] += 0.25 * np.sign(h[i, y, x + 1] - h[i, y, x - 1])
        if 0 < y < hh - 1:
            pts[i, 1] += 0.25 * np.sign(h[i, y + 1, x] - h[i, y - 1, x])
    return _from_numpy(np.ascontiguousarray(pts.astype(np.float32)))


__all__: List[str] = [
    "mean_average_precision",
    "mask_iou",
    "panoptic_quality",
    "oks",
    "pck",
    "heatmap_to_keypoints",
]
