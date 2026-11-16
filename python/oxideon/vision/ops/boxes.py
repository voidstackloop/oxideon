"""Bounding box geometric operations and Non-Maximum Suppression for Oxideon Vision."""

from typing import Any
import numpy as np

from ...oxideon import Tensor as _Tensor, from_numpy as _from_numpy


def _to_np(t: Any) -> np.ndarray:
    if isinstance(t, _Tensor):
        return t.to_numpy()
    return np.asarray(t, dtype=np.float32)


def _box_iou_tensor(boxes1: _Tensor, boxes2: _Tensor) -> tuple[_Tensor, _Tensor, _Tensor, _Tensor, _Tensor, _Tensor]:
    """Differentiable pairwise IoU + union + enclosing components using Tensor ops."""
    x1_1, y1_1, x2_1, y2_1 = boxes1[:, 0:1], boxes1[:, 1:2], boxes1[:, 2:3], boxes1[:, 3:4]
    x1_2 = boxes2[:, 0].unsqueeze(0)
    y1_2 = boxes2[:, 1].unsqueeze(0)
    x2_2 = boxes2[:, 2].unsqueeze(0)
    y2_2 = boxes2[:, 3].unsqueeze(0)

    area1 = (x2_1 - x1_1).clamp_min(0.0) * (y2_1 - y1_1).clamp_min(0.0)
    area2 = (x2_2 - x1_2).clamp_min(0.0) * (y2_2 - y1_2).clamp_min(0.0)

    inter_x1 = 0.5 * (x1_1 + x1_2 + (x1_1 - x1_2).abs())
    inter_y1 = 0.5 * (y1_1 + y1_2 + (y1_1 - y1_2).abs())
    inter_x2 = 0.5 * (x2_1 + x2_2 - (x2_1 - x2_2).abs())
    inter_y2 = 0.5 * (y2_1 + y2_2 - (y2_1 - y2_2).abs())

    inter = (inter_x2 - inter_x1).clamp_min(0.0) * (inter_y2 - inter_y1).clamp_min(0.0)
    union = area1 + area2 - inter
    iou = inter / union.clamp_min(1e-12)

    enc_x1 = 0.5 * (x1_1 + x1_2 - (x1_1 - x1_2).abs())
    enc_y1 = 0.5 * (y1_1 + y1_2 - (y1_1 - y1_2).abs())
    enc_x2 = 0.5 * (x2_1 + x2_2 + (x2_1 - x2_2).abs())
    enc_y2 = 0.5 * (y2_1 + y2_2 + (y2_1 - y2_2).abs())
    enc_area = (enc_x2 - enc_x1).clamp_min(0.0) * (enc_y2 - enc_y1).clamp_min(0.0)
    return iou, union, enc_area, area1, area2, inter


def box_iou(boxes1: _Tensor, boxes2: _Tensor) -> _Tensor:
    """Computes pairwise Intersection over Union (IoU) between two sets of boxes [x1, y1, x2, y2].

    Args:
        boxes1: Tensor of shape [N, 4]
        boxes2: Tensor of shape [M, 4]

    Returns:
        Tensor of shape [N, M] with IoU values in [0, 1].
    """
    if isinstance(boxes1, _Tensor) and isinstance(boxes2, _Tensor):
        if boxes1.requires_grad or boxes2.requires_grad:
            b2 = boxes2.to_dtype(boxes1.dtype) if boxes2.dtype != boxes1.dtype else boxes2
            iou, _, _, _, _, _ = _box_iou_tensor(boxes1, b2)
            return iou
        if hasattr(boxes1, "box_iou"):
            return boxes1.box_iou(boxes2)

    b1 = _to_np(boxes1).astype(np.float32)
    b2 = _to_np(boxes2).astype(np.float32)

    area1 = np.maximum(0.0, b1[:, 2] - b1[:, 0]) * np.maximum(0.0, b1[:, 3] - b1[:, 1])
    area2 = np.maximum(0.0, b2[:, 2] - b2[:, 0]) * np.maximum(0.0, b2[:, 3] - b2[:, 1])

    lt = np.maximum(b1[:, None, :2], b2[None, :, :2])  # [N, M, 2]
    rb = np.minimum(b1[:, None, 2:], b2[None, :, 2:])  # [N, M, 2]

    wh = np.clip(rb - lt, 0.0, None)  # [N, M, 2]
    intersection = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - intersection
    iou = intersection / np.maximum(union, 1e-12)
    return _from_numpy(np.ascontiguousarray(iou, dtype=np.float32))


def generalized_box_iou(boxes1: _Tensor, boxes2: _Tensor) -> _Tensor:
    """Computes generalized IoU (GIoU) between boxes1 [N, 4] and boxes2 [M, 4]."""
    if isinstance(boxes1, _Tensor) and isinstance(boxes2, _Tensor):
        b2 = boxes2.to_dtype(boxes1.dtype) if boxes2.dtype != boxes1.dtype else boxes2
        iou, union, enc_area, _, _, _ = _box_iou_tensor(boxes1, b2)
        return iou - (enc_area - union) / enc_area.clamp_min(1e-12)

    b1 = _to_np(boxes1).astype(np.float32)
    b2 = _to_np(boxes2).astype(np.float32)

    area1 = np.maximum(0.0, b1[:, 2] - b1[:, 0]) * np.maximum(0.0, b1[:, 3] - b1[:, 1])
    area2 = np.maximum(0.0, b2[:, 2] - b2[:, 0]) * np.maximum(0.0, b2[:, 3] - b2[:, 1])

    lt = np.maximum(b1[:, None, :2], b2[None, :, :2])
    rb = np.minimum(b1[:, None, 2:], b2[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    intersection = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - intersection
    iou = intersection / np.maximum(union, 1e-12)

    # Smallest enclosing box
    enclosing_lt = np.minimum(b1[:, None, :2], b2[None, :, :2])
    enclosing_rb = np.maximum(b1[:, None, 2:], b2[None, :, 2:])
    enclosing_wh = np.clip(enclosing_rb - enclosing_lt, 0.0, None)
    enclosing_area = enclosing_wh[:, :, 0] * enclosing_wh[:, :, 1]

    giou = iou - (enclosing_area - union) / np.maximum(enclosing_area, 1e-12)
    return _from_numpy(np.ascontiguousarray(giou, dtype=np.float32))


def nms(boxes: _Tensor, scores: _Tensor, iou_threshold: float) -> _Tensor:
    """Performs Non-Maximum Suppression (NMS) on bounding boxes according to confidence scores.

    Args:
        boxes: Tensor of shape [N, 4] in (x1, y1, x2, y2) format.
        scores: Tensor of shape [N] confidence scores.
        iou_threshold: Discard threshold for overlapping boxes.

    Returns:
        Tensor of shape [K] containing kept box indices in decreasing order of scores.
    """
    if isinstance(boxes, _Tensor) and isinstance(scores, _Tensor) and hasattr(boxes, "nms"):
        return boxes.nms(scores, float(iou_threshold))

    b = _to_np(boxes).astype(np.float32)
    s = _to_np(scores).astype(np.float32)

    order = np.argsort(s)[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        # Compute IoU of the current box with the rest
        xx1 = np.maximum(b[i, 0], b[order[1:], 0])
        yy1 = np.maximum(b[i, 1], b[order[1:], 1])
        xx2 = np.minimum(b[i, 2], b[order[1:], 2])
        yy2 = np.minimum(b[i, 3], b[order[1:], 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        area_i = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
        area_rest = (b[order[1:], 2] - b[order[1:], 0]) * (b[order[1:], 3] - b[order[1:], 1])
        ovr = inter / np.maximum(area_i + area_rest - inter, 1e-12)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return _from_numpy(np.ascontiguousarray(keep, dtype=np.int64))


def masks_to_boxes(masks: _Tensor) -> _Tensor:
    """Computes bounding boxes [N, 4] enclosing binary segmentation masks [N, H, W]."""
    m = _to_np(masks)
    n = m.shape[0]
    boxes = np.zeros((n, 4), dtype=np.float32)

    for i in range(n):
        rows, cols = np.where(m[i] > 0.5)
        if len(rows) == 0:
            boxes[i] = [0.0, 0.0, 0.0, 0.0]
        else:
            boxes[i] = [cols.min(), rows.min(), cols.max() + 1.0, rows.max() + 1.0]

    return _from_numpy(np.ascontiguousarray(boxes, dtype=np.float32))


__all__ = [
    "box_iou",
    "generalized_box_iou",
    "nms",
    "masks_to_boxes",
]
