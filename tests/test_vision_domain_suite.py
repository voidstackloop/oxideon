"""Comprehensive test suite for Oxideon Computer Vision Domain Engine (models, transforms, ops)."""

import numpy as np
import pytest

import oxideon as ox
from oxideon.vision import models, transforms, ops


def test_vision_transforms_compose_pipeline():
    pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((48, 48)),
        transforms.CenterCrop(32),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    raw_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    processed = pipeline(raw_image)

    assert isinstance(processed, ox.Tensor)
    assert processed.shape == [3, 32, 32]
    assert processed.dtype in ("f32", "float32")

    # ColorJitter and RandomErasing
    jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)
    jittered = jitter(processed)
    assert jittered.shape == [3, 32, 32]

    eraser = transforms.RandomErasing(p=1.0, value=0.0)
    erased = eraser(processed)
    assert erased.shape == [3, 32, 32]


def test_vision_ops_box_iou_and_nms():
    boxes1 = ox.tensor([
        [0.0, 0.0, 10.0, 10.0],
        [5.0, 5.0, 15.0, 15.0],
        [20.0, 20.0, 30.0, 30.0],
    ])
    boxes2 = ox.tensor([
        [0.0, 0.0, 10.0, 10.0],
        [0.0, 0.0, 5.0, 5.0],
    ])

    # 1. Box IoU
    iou = ops.box_iou(boxes1, boxes2)
    assert iou.shape == [3, 2]
    # Box 0 with itself: IoU = 1.0
    assert np.isclose(iou.to_numpy()[0, 0], 1.0)
    # Box 2 with Box 0: Disjoint, IoU = 0.0
    assert np.isclose(iou.to_numpy()[2, 0], 0.0)

    # 2. Generalized Box IoU (GIoU)
    giou = ops.generalized_box_iou(boxes1, boxes2)
    assert giou.shape == [3, 2]
    assert np.isclose(giou.to_numpy()[0, 0], 1.0)
    assert giou.to_numpy()[2, 0] < 0.0  # Disjoint boxes have negative GIoU

    # 3. Non-Maximum Suppression (NMS)
    scores = ox.tensor([0.9, 0.85, 0.5])
    keep = ops.nms(boxes1, scores, iou_threshold=0.3)
    assert len(keep) >= 2
    assert keep.to_numpy()[0] == 0  # Highest score kept first

    # 4. Masks to Boxes
    masks = np.zeros((2, 20, 20), dtype=np.float32)
    masks[0, 2:8, 4:12] = 1.0
    boxes_from_masks = ops.masks_to_boxes(ox.tensor(masks))
    assert boxes_from_masks.shape == [2, 4]
    assert np.allclose(boxes_from_masks.to_numpy()[0], [4.0, 2.0, 12.0, 8.0])


def test_vision_ops_roi_align_and_pool():
    # Feature map: [1, 4, 16, 16]
    features = ox.randn([1, 4, 16, 16])
    # RoIs: [batch_idx, x1, y1, x2, y2]
    rois = ox.tensor([
        [0.0, 1.0, 1.0, 8.0, 8.0],
        [0.0, 4.0, 4.0, 12.0, 12.0],
    ])

    # 1. RoI Align
    aligned = ops.roi_align(features, rois, output_size=(7, 7), spatial_scale=1.0)
    assert aligned.shape == [2, 4, 7, 7]

    # 2. RoI Pool
    pooled = ops.roi_pool(features, rois, output_size=(7, 7), spatial_scale=1.0)
    assert pooled.shape == [2, 4, 7, 7]


def test_resnet_and_mobilenet_architectures():
    x = ox.randn([2, 3, 32, 32])

    # ResNet-18
    resnet = models.resnet18(num_classes=5)
    y_res = resnet(x)
    assert y_res.shape == [2, 5]

    # MobileNetV3-Small
    mnet = models.mobilenet_v3_small(num_classes=5)
    y_mnet = mnet(x)
    assert y_mnet.shape == [2, 5]


def test_unet_segmentation_forward_backward():
    unet = models.UNet(in_channels=3, num_classes=1, features=[16, 32])
    x = ox.randn([1, 3, 32, 32])

    mask_logits = unet(x)
    assert mask_logits.shape == [1, 1, 32, 32]

    loss = mask_logits.sum()
    loss.backward()
    assert unet.final_conv.weight.grad is not None
    assert unet.final_conv.weight.grad.shape == unet.final_conv.weight.shape
