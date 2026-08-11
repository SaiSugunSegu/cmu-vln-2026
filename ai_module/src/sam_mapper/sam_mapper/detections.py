"""Translate SAM 3 output into the detections dict that ObjMapper.update_map consumes.

The 3D mapper's contract (unchanged from semantic_mapper, see docs/M2_perception.md 2.4):

    {'bboxes':      (M,4)   float xyxy pixels,
     'confidences': (M,)    float,
     'labels':      (M,)    str,   meta-class
     'ids':         (M,)    int,   >=0 tracked instance, <0 background
     'masks':       (M,H,W) bool}

SAM 3 fills all five directly. The two things this module has to get right:

  * labels come from prompt_to_obj_ids rather than a spaCy phrase match, so there is
    no 'None' fallthrough and no meta-class ambiguity;
  * background classes (wall/floor/ceiling) must be renumbered to NEGATIVE ids, which
    is how ObjMapper routes them to background_obj_list instead of tracking them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def default_label(prompt: str) -> str:
    """Meta-class name for a prompt, when the config does not give one explicitly.

    Spaces stripped, lowercased — matches how single_object.py's DIMENSION_PRIORS spells
    multi-word classes ('pottedplant', 'fireextinguisher'). Give two prompts the same
    explicit `label:` to merge them into one class (e.g. "sofa" and "loveseat" -> sofa).
    """
    return prompt.strip().replace(" ", "").lower()


@dataclass(frozen=True)
class ObjectSpec:
    prompt: str
    instance: bool
    label: str


class PromptTable:
    """Config `objects:` list, indexed for per-frame lookup."""

    def __init__(self, objects: list[dict]):
        if not objects:
            raise ValueError("config 'objects' list is empty — nothing to detect")

        self.specs = [
            ObjectSpec(
                prompt=o["prompt"],
                instance=bool(o.get("instance", True)),
                label=o.get("label") or default_label(o["prompt"]),
            )
            for o in objects
        ]

        duplicates = {s.prompt for s in self.specs if
                      sum(1 for t in self.specs if t.prompt == s.prompt) > 1}
        if duplicates:
            raise ValueError(f"duplicate prompts in config 'objects': {sorted(duplicates)}")
        self.by_prompt = {s.prompt: s for s in self.specs}

        # Background classes get ids -1, -2, -3, ... one per unique LABEL (not per
        # prompt, since several prompts may share a label). Fixed up front rather than
        # grown at runtime, so ids stay stable across frames — semantic_mapper grows a
        # module global instead, which leaks state between instances (docs quirk #7).
        background_labels = []
        for spec in self.specs:
            if not spec.instance and spec.label not in background_labels:
                background_labels.append(spec.label)
        self.background_ids = {label: -(i + 1) for i, label in enumerate(background_labels)}

    @property
    def prompts(self) -> list[str]:
        return [s.prompt for s in self.specs]

    def label_template(self) -> dict:
        """The shape ObjMapper.__init__ wants: {label: {'is_instance': bool, 'prompts': [...]}}."""
        template: dict[str, dict] = {}
        for spec in self.specs:
            entry = template.setdefault(spec.label, {"is_instance": spec.instance, "prompts": []})
            entry["prompts"].append(spec.prompt)
            entry["is_instance"] = entry["is_instance"] or spec.instance  # any instance prompt -> instance label
        return template


def _invert_prompt_map(prompt_to_obj_ids: dict[str, list[int]],
                       scores_by_id: dict[int, float]) -> dict[int, str]:
    """obj_id -> prompt.

    SAM 3 can report the same object under more than one prompt (e.g. "sofa" and
    "chair" both firing on a loveseat). Keep the highest-scoring claim; ties resolve
    by prompt order, which is deterministic.
    """
    owner: dict[int, str] = {}
    best: dict[int, float] = {}
    for prompt, obj_ids in prompt_to_obj_ids.items():
        for obj_id in obj_ids:
            score = scores_by_id.get(int(obj_id), 0.0)
            if int(obj_id) not in owner or score > best[int(obj_id)]:
                owner[int(obj_id)] = prompt
                best[int(obj_id)] = score
    return owner


def to_detections(result, table: PromptTable) -> dict:
    """Sam3FrameResult -> the 5-key detections dict.

    Objects whose id no prompt claims are dropped: without a label the mapper cannot
    place them in a class, and a mislabelled object is worse than a missing one.
    """
    if len(result) == 0:
        return _empty()

    scores_by_id = {int(i): float(s) for i, s in zip(result.object_ids, result.scores)}
    owner = _invert_prompt_map(result.prompt_to_obj_ids, scores_by_id)

    rows, labels, ids = [], [], []
    for row, obj_id in enumerate(result.object_ids):
        prompt = owner.get(int(obj_id))
        if prompt is None:
            continue
        spec = table.by_prompt.get(prompt)
        if spec is None:
            continue

        rows.append(row)
        labels.append(spec.label)
        # Background objects are not tracked: renumber onto the negative-id convention.
        ids.append(int(obj_id) if spec.instance else table.background_ids[spec.label])

    if not ids:
        return _empty()

    # Select rows rather than append-then-restack: masks are 1.2 MB per object, and when
    # nothing is dropped (the usual case) this passes the backend's array through with no
    # copy. Consumers only read masks, so sharing the buffer is safe.
    keep_all = len(rows) == len(result.object_ids)
    select = slice(None) if keep_all else np.asarray(rows, dtype=int)
    return {
        "bboxes": np.asarray(result.boxes[select], dtype=float),
        "confidences": np.asarray(result.scores[select], dtype=float),
        "labels": np.asarray(labels, dtype=object),
        "ids": np.asarray(ids, dtype=int),
        "masks": result.masks[select],
    }


def _empty() -> dict:
    return {
        "bboxes": np.zeros((0, 4), dtype=float),
        "confidences": np.zeros(0, dtype=float),
        "labels": np.zeros(0, dtype=object),
        "ids": np.zeros(0, dtype=int),
        "masks": np.zeros((0, 0, 0), dtype=bool),
    }


def encode_instance_id(obj_id: int) -> int:
    """Real obj_id (>=0 instance, <0 background) -> positive uint16 pixel value for the
    /sam3/instance_map wire format (see docs/M2_perception.md 3.6-split). 0 is reserved for
    'no detection'; background ids get the top of the uint16 range so they can never
    collide with an instance id. sam_node encodes with this, map_node decodes with it.
    """
    return obj_id + 1 if obj_id >= 0 else 65536 + obj_id


def build_id_map(ids: np.ndarray, masks: np.ndarray, height: int, width: int) -> np.ndarray:
    """Masks -> the mono16 /sam3/instance_map image, beside encode_instance_id because the
    two together are the wire format map_node decodes.

    Vectorised: argmax over the REVERSED mask stack resolves every pixel in one pass and
    keeps the old per-object loop's tie-break (later entries win on overlaps).
    """
    if len(ids) == 0:
        return np.zeros((height, width), dtype=np.uint16)

    codes = np.array([encode_instance_id(int(i)) for i in ids], dtype=np.uint16)
    winner = len(codes) - 1 - masks[::-1].argmax(axis=0)
    return np.where(masks.any(axis=0), codes[winner], np.uint16(0)).astype(np.uint16)
