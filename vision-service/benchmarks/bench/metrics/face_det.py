from __future__ import annotations

import numpy as np


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detection_metrics(
    pred_boxes: list[np.ndarray],
    pred_scores: list[np.ndarray],
    gt_boxes: list[np.ndarray],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
) -> dict[str, float]:
    tp = fp = fn = 0
    for preds, scores, gts in zip(pred_boxes, pred_scores, gt_boxes):
        keep = scores >= score_threshold
        preds = preds[keep]
        matched_gt = set()
        for pb in preds:
            best_iou, best_j = 0.0, -1
            for j, gb in enumerate(gts):
                iou = _iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_threshold and best_j not in matched_gt:
                tp += 1
                matched_gt.add(best_j)
            else:
                fp += 1
        fn += len(gts) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"precision": precision, "recall": recall, "tp": float(tp), "fp": float(fp), "fn": float(fn)}
