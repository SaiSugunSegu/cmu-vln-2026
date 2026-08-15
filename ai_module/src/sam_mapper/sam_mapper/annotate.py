"""Detection overlays: the /annotated_image debug view and the best-view silhouette copy.

Plain cv2 rather than `supervision`, so this pulls in no dependency the node does not
already have, and so the colour of an object is a pure function of its SAM 3 id — an
object keeps its colour for as long as it is tracked, which is what makes id switches
visible at a glance when scrubbing the topic in Foxglove.

`annotate_frame` is the debug view: filled masks, boxes, `label#id conf`. `silhouette_frame`
is the readable one: mask outline and class name only, over untouched pixels.
"""
from __future__ import annotations

import math

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

# Leash: past this a tab reads as belonging to whatever it landed on, which is worse than
# two tabs sharing an edge. ~4 tab heights, so three captions still fit on one anchor.
MAX_SLIDE_PX = 64


def _mask_anchor(mask, bbox) -> tuple[int, int]:
    """The point a caption should point at: top-centre of the mask's largest blob.

    A SAM mask routinely carries a stray component far from the object, and the bbox is
    their union — one column's spans 1091 px for an object 80 px wide, putting its caption
    over empty sky. The largest connected component is the object; the rest is bleed.

    No mask (all `mark_frame` ever has) falls back to the bbox top edge.
    """
    x0, y0, x1, _ = (int(round(v)) for v in bbox)
    fallback = ((x0 + x1) // 2, y0)
    if mask is None:
        return fallback
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(mask, dtype=np.uint8), 8)
    if count < 2:                       # background only — the mask is empty
        return fallback
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))   # row 0 is the background
    left = int(stats[largest, cv2.CC_STAT_LEFT])
    return left + int(stats[largest, cv2.CC_STAT_WIDTH]) // 2, int(stats[largest, cv2.CC_STAT_TOP])


class _CaptionLayout:
    """Collects captions, then draws them so that no two tabs overlap.

    A caption wants to sit just above the point it names. Objects in one crop routinely
    stack — a tv in front of a cabinet, two chairs at the same depth — and two tabs at the
    same spot leave both unreadable, which defeats the point of the overlay.

    So a caption whose slot is taken moves to the nearest free one, up, down or sideways,
    but never further than `MAX_SLIDE_PX`; beyond the leash it takes the least-overlapping
    slot instead. Any tab that had to move gets a leader line back to its anchor: colour
    alone cannot carry the association, since `_color_for` steps hue by the golden angle
    over OpenCV's 0-179 scale and ids 44 apart collide.
    """

    # Offsets tried around the preferred slot, in units of (tab_w // 2, tab_h + 2).
    _DX_UNITS = (0, -1, 1, -2, 2)
    _DY_UNITS = (0, -1, 1, -2, 2, -3, 3, -4, 4)

    def __init__(self) -> None:
        self._pending: list[tuple] = []

    def add(self, x: float, y: float, text: str, color: tuple) -> None:
        """(x, y) is the anchor — the point on the object this caption names.

        Empty text draws nothing, rather than a two-pixel sliver: a caller marking an image
        that already carries its own labels wants the box alone.
        """
        if not str(text):
            return
        self._pending.append((int(round(x)), int(round(y)), str(text), color))

    def draw(self, out: np.ndarray) -> None:
        """Draw every collected caption. Call once, after the masks and boxes."""
        height, width = out.shape[:2]
        placed: list[tuple] = []
        laid_out: list[tuple] = []      # (rect, text_baseline_h, anchor, text, color)
        # Top-to-bottom, left-to-right, so which caption keeps its preferred slot depends
        # on the geometry and not on the order detections happen to arrive in.
        for x, y, text, color in sorted(self._pending, key=lambda c: (c[1], c[0])):
            (tw, th), baseline = cv2.getTextSize(text, _FONT, _FONT_SCALE, _FONT_THICKNESS)
            tab_w, tab_h = tw + 2, th + baseline + 1
            x0, top = self._place(x, y, tab_w, tab_h, width, height, placed)
            rect = (x0, top, x0 + tab_w, top + tab_h)
            placed.append(rect)
            laid_out.append((rect, th, (x, y), text, color))

        # Leaders first: a line drawn afterwards would cross the text it points away from.
        for rect, _, anchor, _, color in laid_out:
            self._leader(out, rect, anchor, color)

        for (x0, top, x1, y1), th, _, text, color in laid_out:
            cv2.rectangle(out, (x0, top), (x1, y1), color, -1)
            cv2.putText(out, text, (x0 + 1, top + th + 1),
                        _FONT, _FONT_SCALE, (0, 0, 0), _FONT_THICKNESS, cv2.LINE_AA)

    @classmethod
    def _place(cls, ax: int, ay: int, tab_w: int, tab_h: int, width: int, height: int,
               placed: list[tuple]) -> tuple[int, int]:
        """Top-left of the nearest free slot to the preferred one, inside the leash."""
        x_pref, y_pref = ax - tab_w // 2, ay - tab_h   # centred on the anchor, just above it

        best, best_overlap = None, None
        for dx, dy in cls._offsets(max(tab_w // 2, 1), tab_h + 2):
            x0, top = x_pref + dx, y_pref + dy
            if not (0 <= x0 <= width - tab_w and 0 <= top <= height - tab_h):
                continue
            overlap = sum(cls._overlap((x0, top, x0 + tab_w, top + tab_h), r) for r in placed)
            if overlap == 0:
                return x0, top
            if best_overlap is None or overlap < best_overlap:
                best, best_overlap = (x0, top), overlap
        if best is not None:
            return best
        # Every candidate fell outside the frame — a tab wider or taller than the crop.
        # Clamp the preferred slot and accept it; overlapping beats not drawing.
        return (min(max(x_pref, 0), max(width - tab_w, 0)),
                min(max(y_pref, 0), max(height - tab_h, 0)))

    @classmethod
    def _offsets(cls, x_step: int, y_step: int) -> list[tuple]:
        """(dx, dy) around the preferred slot, nearest first, inside `MAX_SLIDE_PX`.

        Distance first, so a small sideways nudge beats a long slide; then `(dy, dx)`, so up
        and left win ties — the anchor is the object's TOP, and the space above is usually
        empty. Tie-breaking at all keeps placement a function of geometry, not iteration order.
        """
        offsets = [(dx * x_step, dy * y_step)
                   for dx in cls._DX_UNITS for dy in cls._DY_UNITS]
        offsets = [o for o in offsets if math.hypot(*o) <= MAX_SLIDE_PX]
        offsets.sort(key=lambda o: (o[0] * o[0] + o[1] * o[1], o[1], o[0]))
        return offsets

    @staticmethod
    def _overlap(a: tuple, b: tuple) -> int:
        overlap_w = min(a[2], b[2]) - max(a[0], b[0])
        overlap_h = min(a[3], b[3]) - max(a[1], b[1])
        return overlap_w * overlap_h if overlap_w > 0 and overlap_h > 0 else 0

    @staticmethod
    def _leader(out: np.ndarray, rect: tuple, anchor: tuple, color: tuple) -> None:
        """Thin line plus a dot, drawn only for a tab that had to leave its anchor.

        Aliased on purpose: anti-aliasing blends the line into the pixels under it, and the
        leader's whole job is to be recognisably the object's own colour.
        """
        x0, y0, x1, y1 = rect
        ax, ay = anchor
        near = (min(max(ax, x0), x1), min(max(ay, y0), y1))
        if math.hypot(ax - near[0], ay - near[1]) <= y1 - y0:   # still touching its object
            return
        cv2.line(out, near, (ax, ay), color, 1)
        cv2.circle(out, (ax, ay), 3, color, -1)


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
    for mask, box, label, obj_id, score in zip(masks, boxes, labels, ids, scores):
        color = _color_for(int(obj_id))
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        usable = mask if mask.shape[:2] == out.shape[:2] else None
        captions.add(*_mask_anchor(usable, box),
                     f'{label}#{int(obj_id)} {float(score):.2f}', color)

    # Last, so no box can land on top of a caption.
    captions.draw(out)
    return out


def mark_frame(image: np.ndarray, marks, thickness: int = 1) -> np.ndarray:
    """Highlight boxes over a crop that has already been drawn: `marks` is (bbox, id, text).

    The handle a model needs to answer "which one" with an id instead of a description. The
    colour is `_color_for(id)`, the same one `silhouette_frame` outlined that object with, so
    marking a silhouette leaves each object outline, box and tab in one colour.

    Empty `text` draws the box alone — the case for a finalized silhouette, whose own
    captions already carry the map id; repeating it would put two tabs on one object. The tab
    is for the bare crop, which carries no ids at all.

    A box is drawn as well as any tab, unlike the silhouette: when two objects stack, a tab
    alone does not say which of them it belongs to.
    """
    out = image.copy()
    captions = _CaptionLayout()
    for box, obj_id, text in marks:
        color = _color_for(int(obj_id))
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        # No mask here, only the manifest's box — `_mask_anchor` degrades to its top edge.
        captions.add(*_mask_anchor(None, box), str(text), color)
    captions.draw(out)
    return out


def silhouette_frame(image: np.ndarray, detections: dict, thickness: int = 2) -> np.ndarray:
    """Mask outline + class name only — no fill, no box, no id, no confidence.

    Same inputs as `annotate_frame`. Leaves the mask's pixels untouched bar a caption's
    anchor dot, so the object stays readable: this is the copy meant to be looked at (or
    handed to a VLM), where the debug overlay's id and score text is noise.
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
        # The mask positions the caption only; the box is never drawn.
        captions.add(*_mask_anchor(mask, box), str(label), color)

    # Last, so no outline can land on top of a caption.
    captions.draw(out)
    return out
