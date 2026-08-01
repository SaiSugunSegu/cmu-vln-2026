"""Debug overlay for /annotated_image.

Plain cv2 rather than `supervision`, so this pulls in no dependency the node does not
already have, and so the colour of an object is a pure function of its SAM 3 id — an
object keeps its colour for as long as it is tracked, which is what makes id switches
visible at a glance when scrubbing the topic in Foxglove.
"""
from __future__ import annotations

import cv2
import numpy as np


def _color_for(obj_id: int) -> tuple[int, int, int]:
    """Deterministic, well-spread BGR colour for an object id.

    Golden-ratio hue stepping keeps consecutive ids visually distinct. Background
    objects (negative ids) are deliberately muted so instances stand out.
    """
    if obj_id < 0:
        return (110, 110, 110)
    hue = int((obj_id * 137.508) % 180)          # 137.508 deg = golden angle, in OpenCV hue units
    hsv = np.uint8([[[hue, 200, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(b), int(g), int(r))


def annotate_frame(image: np.ndarray, detections: dict, mask_alpha: float = 0.45) -> np.ndarray:
    """image: (H,W,3) BGR. detections: the 5-key dict. Returns a new BGR image."""
    out = image.copy()
    masks = detections['masks']
    boxes = detections['bboxes']
    labels = detections['labels']
    ids = detections['ids']
    scores = detections['confidences']

    if len(ids) == 0:
        return out

    # Masks first, so boxes and text stay legible on top of them.
    overlay = out.copy()
    for mask, obj_id in zip(masks, ids):
        if mask.shape[:2] != out.shape[:2]:
            continue
        overlay[mask] = _color_for(int(obj_id))
    cv2.addWeighted(overlay, mask_alpha, out, 1 - mask_alpha, 0, out)

    for box, label, obj_id, score in zip(boxes, labels, ids, scores):
        color = _color_for(int(obj_id))
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        text = f'{label}#{int(obj_id)} {float(score):.2f}'
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Keep the caption inside the frame when the box touches the top edge.
        ty = max(y1, th + baseline + 1)
        cv2.rectangle(out, (x1, ty - th - baseline - 1), (x1 + tw + 2, ty), color, -1)
        cv2.putText(out, text, (x1 + 1, ty - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return out
