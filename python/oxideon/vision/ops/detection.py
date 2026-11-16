"""Anchor-based detection decoding (YOLO-style) for Oxideon Vision."""

from typing import Any, Dict, List, Tuple
import numpy as np

from ...oxideon import Tensor as _Tensor, from_numpy as _from_numpy


def _to_np(t: Any) -> np.ndarray:
    if isinstance(t, _Tensor):
        return t.to_numpy()
    return np.asarray(t, dtype=np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64))).astype(np.float32)


def yolo_decode(
    predictions: Any,
    anchors: Any,
    stride: int,
    num_classes: int,
    conf_threshold: float = 0.25,
) -> Tuple[_Tensor, _Tensor, _Tensor]:
    """Decodes YOLOv3-style raw predictions for a single image.

    Args:
        predictions: Array/Tensor of shape [H, W, A * (5 + C)] with raw
            (tx, ty, tw, th, objectness, class logits...) per anchor.
        anchors: Array/Tensor of shape [A, 2] with (width, height) in pixels.
        stride: Pixels per grid cell (image_size = grid_size * stride).
        num_classes: Number of classes C (>= 1).
        conf_threshold: Keep detections with objectness * class score >= this.

    Returns:
        (boxes, scores, classes): boxes [N, 4] xyxy in pixels, scores [N],
        classes [N] int64, sorted by decreasing score. Empty [0, 4]/[0]/[0]
        when nothing passes the threshold.
    """
    pred = _to_np(predictions).astype(np.float32)
    anch = _to_np(anchors).astype(np.float32)
    if pred.ndim != 3:
        raise ValueError(f"predictions must be [H, W, A*(5+C)], got shape {pred.shape}")
    if anch.ndim != 2 or anch.shape[1] != 2:
        raise ValueError(f"anchors must be [A, 2], got shape {anch.shape}")
    if not np.all(np.isfinite(pred)):
        raise ValueError("predictions must be finite")
    if not np.all(np.isfinite(anch)) or np.any(anch <= 0.0):
        raise ValueError("anchors must be finite and positive")
    if not isinstance(stride, (int, np.integer)) or int(stride) < 1:
        raise ValueError(f"stride must be a positive int, got {stride}")
    if not isinstance(num_classes, (int, np.integer)) or int(num_classes) < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}")
    stride = int(stride)
    num_classes = int(num_classes)
    n_anchors = int(anch.shape[0])
    expect = n_anchors * (5 + num_classes)
    if pred.shape[2] != expect:
        raise ValueError(
            f"predictions last dim {pred.shape[2]} != A*(5+C) = {expect} "
            f"(A={n_anchors}, C={num_classes})"
        )
    if not np.isfinite(conf_threshold):
        raise ValueError("conf_threshold must be finite")

    h, w = int(pred.shape[0]), int(pred.shape[1])
    cells = pred.reshape(h, w, n_anchors, 5 + num_classes)
    tx, ty = cells[..., 0], cells[..., 1]
    tw, th = np.clip(cells[..., 2], -8.0, 8.0), np.clip(cells[..., 3], -8.0, 8.0)
    obj = _sigmoid(cells[..., 4])
    cls = _sigmoid(cells[..., 5:])

    cx = np.arange(w, dtype=np.float32)[None, :, None]
    cy = np.arange(h, dtype=np.float32)[:, None, None]
    bx = (_sigmoid(tx) + cx) * stride
    by = (_sigmoid(ty) + cy) * stride
    bw = anch[None, None, :, 0] * np.exp(tw)
    bh = anch[None, None, :, 1] * np.exp(th)

    x1 = (bx - bw / 2.0).reshape(-1)
    y1 = (by - bh / 2.0).reshape(-1)
    x2 = (bx + bw / 2.0).reshape(-1)
    y2 = (by + bh / 2.0).reshape(-1)
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    scores_all = obj.reshape(-1, 1) * cls.reshape(-1, num_classes)
    classes = np.argmax(scores_all, axis=1).astype(np.int64)
    scores = scores_all[np.arange(scores_all.shape[0]), classes]

    keep = scores >= float(conf_threshold)
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    order = np.argsort(-scores, kind="stable")
    boxes = np.ascontiguousarray(boxes[order], dtype=np.float32)
    scores = np.ascontiguousarray(scores[order], dtype=np.float32)
    classes = np.ascontiguousarray(classes[order])
    return (
        _from_numpy(boxes.reshape(-1, 4)),
        _from_numpy(scores.reshape(-1)),
        _from_numpy(classes),
    )


def yolo_decode_batch(
    predictions: Any,
    anchors: Any,
    stride: int,
    num_classes: int,
    conf_threshold: float = 0.25,
) -> List[Dict[str, _Tensor]]:
    """Decodes a batch [B, H, W, A * (5 + C)] into per-image dicts.

    Returns:
        List of B dicts with keys "boxes", "scores", "classes" (see
        :func:`yolo_decode`).
    """
    pred = _to_np(predictions).astype(np.float32)
    if pred.ndim != 4:
        raise ValueError(f"predictions must be [B, H, W, A*(5+C)], got shape {pred.shape}")
    out: List[Dict[str, _Tensor]] = []
    for b in range(int(pred.shape[0])):
        boxes, scores, classes = yolo_decode(pred[b], anchors, stride, num_classes, conf_threshold)
        out.append({"boxes": boxes, "scores": scores, "classes": classes})
    return out
