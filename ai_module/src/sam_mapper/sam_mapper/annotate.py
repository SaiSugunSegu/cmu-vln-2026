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


_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICKNESS = 1


class _CaptionLayout:
    """Collects captions, then draws them so that no two tabs overlap.

    A caption wants to sit just above its object's top-left corner. Objects in one crop
    routinely stack — a tv in front of a cabinet, two chairs at the same depth — and two
    tabs at the same spot leave both unreadable, which defeats the point of the overlay.
    So a caption whose slot is taken slides straight down in tab-height steps until it
    finds clear space, staying in its own object's column.
    """

    def __init__(self) -> None:
        self._pending: list[tuple] = []

    def add(self, x: float, y: float, text: str, color: tuple) -> None:
        self._pending.append((int(round(x)), int(round(y)), str(text), color))

    def draw(self, out: np.ndarray) -> None:
        """Draw every collected caption. Call once, after the masks and boxes."""
        height, width = out.shape[:2]
        placed: list[tuple] = []
        # Top-to-bottom, left-to-right, so which caption keeps its preferred slot depends
        # on the geometry and not on the order detections happen to arrive in.
        for x, y, text, color in sorted(self._pending, key=lambda c: (c[1], c[0])):
            (tw, th), baseline = cv2.getTextSize(text, _FONT, _FONT_SCALE, _FONT_THICKNESS)
            tab_w, tab_h = tw + 2, th + baseline + 1
            # Keep the tab inside the frame: shifted right at the left edge, left at the
            # right edge, and pushed below the anchor when the object touches the top.
            x0 = min(max(x, 0), max(width - tab_w, 0))
            top = self._free_top(max(y - tab_h, 0), tab_h, x0, tab_w, height, placed)

            placed.append((x0, top, x0 + tab_w, top + tab_h))
            cv2.rectangle(out, (x0, top), (x0 + tab_w, top + tab_h), color, -1)
            cv2.putText(out, text, (x0 + 1, top + th + 1),
                        _FONT, _FONT_SCALE, (0, 0, 0), _FONT_THICKNESS, cv2.LINE_AA)

    @staticmethod
    def _free_top(top: int, tab_h: int, x0: int, tab_w: int, height: int,
                  placed: list[tuple]) -> int:
        """First y at or below `top` where this tab clears every tab already placed.

        Falls back to `top` when the column is full all the way down: overlapping one
        caption beats dropping it, and a crop that dense is degenerate anyway.
        """
        candidate = top
        step = tab_h + 2
        while candidate + tab_h <= height:
            if not any(x0 < px1 and px0 < x0 + tab_w
                       and candidate < py1 and py0 < candidate + tab_h
                       for px0, py0, px1, py1 in placed):
                return candidate
            candidate += step
        return top


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

    captions = _CaptionLayout()
    for box, label, obj_id, score in zip(boxes, labels, ids, scores):
        color = _color_for(int(obj_id))
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        captions.add(x1, y1, f'{label}#{int(obj_id)} {float(score):.2f}', color)

    # Last, so no box can land on top of a caption.
    captions.draw(out)
    return out


def mark_frame(image: np.ndarray, marks, thickness: int = 1) -> np.ndarray:
    """Numbered tabs over a crop that has already been drawn: `marks` is (bbox, id, text).

    The handle a model needs to answer "which one" with an id instead of a description. A
    category-2 answer is one entry of the 3D map, and the map's keys mean nothing to a model
    looking at pixels unless the pixels carry the same keys — so the tab text is the track
    id, and its colour is `_color_for(id)`, the same colour `silhouette_frame` outlined that
    object with. Marking the silhouette copy therefore leaves each object with an outline and
    a tab in one colour.

    A thin box is drawn as well as the tab, unlike the silhouette: when two objects stack, a
    tab alone does not say which of them it belongs to, and a mislabelled tab is worse than a
    box over readable pixels.
    """
    out = image.copy()
    captions = _CaptionLayout()
    for box, obj_id, text in marks:
        color = _color_for(int(obj_id))
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        captions.add(x1, y1, str(text), color)
    captions.draw(out)
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

    captions = _CaptionLayout()
    for mask, box, label, obj_id in zip(masks, boxes, labels, ids):
        if mask.shape[:2] != out.shape[:2]:
            continue
        color = _color_for(int(obj_id))
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, thickness)
        # The box positions the caption only; it is never drawn.
        captions.add(box[0], box[1], str(label), color)

    # Last, so no outline can land on top of a caption.
    captions.draw(out)
    return out
