"""Detection overlays: the /annotated_image debug view and the best-view silhouette copy.

Plain cv2 rather than `supervision`, so this pulls in no dependency the node does not
already have, and so the colour of an object is a pure function of its SAM 3 id — an
object keeps its colour for as long as it is tracked, which is what makes id switches
visible at a glance when scrubbing the topic in Foxglove.

`annotate_frame` is the debug view: filled masks, boxes, `label#id conf`. `silhouette_frame`
is the readable one: mask outline and class name only, over untouched pixels.
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


def _draw_caption(out: np.ndarray, x: int, y: int, text: str, color: tuple) -> None:
    """Filled colour tab with black text, anchored above (x, y)."""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # Keep the caption inside the frame when the box touches the top edge.
    ty = max(y, th + baseline + 1)
    cv2.rectangle(out, (x, ty - th - baseline - 1), (x + tw + 2, ty), color, -1)
    cv2.putText(out, text, (x + 1, ty - baseline),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


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

        _draw_caption(out, x1, y1, f'{label}#{int(obj_id)} {float(score):.2f}', color)

    return out


def silhouette_frame(image: np.ndarray, detections: dict, thickness: int = 2) -> np.ndarray:
    """Mask outline + class name only — no fill, no box, no id, no confidence.

    Same inputs as `annotate_frame`. Leaves every pixel inside the mask untouched, so the
    object itself stays readable: this is the copy meant to be looked at (or handed to a
    VLM), where the id and score text of the debug overlay is noise.
    """
    out = image.copy()
    masks = detections['masks']
    boxes = detections['bboxes']
    labels = detections['labels']
    ids = detections['ids']

    if len(ids) == 0:
        return out

    for mask, box, label, obj_id in zip(masks, boxes, labels, ids):
        if mask.shape[:2] != out.shape[:2]:
            continue
        color = _color_for(int(obj_id))
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, thickness)

        # The box positions the caption only; it is never drawn.
        x1, y1 = (int(round(v)) for v in box[:2])
        _draw_caption(out, x1, y1, str(label), color)

    return out
