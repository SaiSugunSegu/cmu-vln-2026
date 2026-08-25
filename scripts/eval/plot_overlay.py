#!/usr/bin/env python3
"""Draw ground truth and prediction on the room's floor plan, for one question.

    python3 scripts/eval/plot_overlay.py --scene arabic_room --category 3 --qid Q05
    just overlay /data/runs/challenge_report_sim_cat3.json arabic_room Q05

A report row says a box scored 0.31 IoU or a route missed a constraint. It cannot say WHY,
and the three explanations call for completely different work: the object is in the wrong
place, or in the right place with a bad box, or not in the map at all. One picture separates
them, which is the whole reason this exists.

THE TRANSFORM. `data/scenes/<scene>/map.jpg` is a screenshot (its own readme.txt says so) with
no documented extent, so it had to be established by measurement: fit map.ply's XY bounding box
to the full image, y flipped. Checked on all 15 scenes -- aspect error median 2.6%, max 4.7%.
Each axis is fitted independently, so that residual is the screenshot's margin rather than a
distortion. Registration is good to roughly +/-0.3 m on arabic_room's 8.3 m room: an
alternative fit anchored on the wall lines disagrees with this one by at most that, and
neither could be shown better -- the render is nearly all ink, so both 2-D silhouette IoU
(all eight flips score ~0.31) and 1-D density registration (corr ~0.3) fail to discriminate.

    THIS IS FOR LOOKING AT, NOT FOR MEASURING. A couple of decimetres of registration error is
    invisible to the eye and fatal to a metric. Every number in this repo comes from the
    scorers; nothing should ever be read off this picture.

Two alternatives were tried and rejected -- recorded so they are not tried again:
  * wall-peak detection plus the floor extent from object_list.txt: good on 7 of 15 scenes,
    badly wrong on the rest (japanese_room 31%, livingroom_1 37%, home_building_1 48%),
    because one `floor` row does not span the render in a multi-room scene;
  * solving the orientation by point-cloud/ink IoU: all eight flip combinations score ~0.31
    and are indistinguishable, because the stippled render is nearly all ink.

The y-flip is pinned by landmarks instead -- see landmark_check().

Host-side: PIL and matplotlib are installed there, cv2 is not.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("MAP3D_DATA_ROOT") or (REPO / "data"))

#: Two colours only: ground truth RED, prediction BLUE. Every other distinction is carried by
#: line style, not hue -- a constraint object is solid and a reference object dashed, both red,
#: because they are equally ground truth and the question is only ever "did the blue land on
#: the red". More colours made that harder to see, not easier.
GT = "#d00000"
PRED = "#0353a4"
GT_CONSTRAINT = GT
GT_REFERENCE = GT
GT_ROUTE = GT
PRED_OBJECT = PRED
PRED_ROUTE = PRED
# The VLM's own waypoints, deliberately neither the GT red nor the predicted blue. Everything
# else on this figure that came from us is one blue -- mapped objects, exploration, the driven
# route -- so the one layer you check the robot against was the hardest thing to pick out.
WAYPOINT = "#e01e8a"
# Where the waypoint was MOVED to before publishing, and so where the robot was actually
# aimed. Its own colour because the whole question a cat-3 overlay answers is a chain of three
# points -- what the model asked for, what we aimed at, where the robot stopped -- and two of
# them sharing a colour hides the step in between.
SNAP = "#f4a300"

#: Object name size. Small enough not to bury the floor plan, large enough to read at a glance
#: without zooming -- the point of the figure is to be glanceable.
LABEL_PT = 9
NUMBER_PT = 12


# -- geometry ---------------------------------------------------------------

def ply_xy_bounds(path: Path, stride: int = 97) -> tuple[float, float, float, float]:
    """(x_min, x_max, y_min, y_max) of a binary PLY, without loading it.

    These files are ~60 MB and only the extent is wanted, so memmap and stride rather than
    read. The stride is coprime with any plausible row structure, so it cannot sample one
    scan line repeatedly.
    """
    with open(path, "rb") as handle:
        header = handle.read(400).partition(b"end_header\n")[0]
    offset = len(header) + len(b"end_header\n")
    count = (path.stat().st_size - offset) // 12
    points = np.memmap(path, dtype="<f4", mode="r", offset=offset, shape=(int(count), 3))
    sample = np.asarray(points[::stride])
    return (float(sample[:, 0].min()), float(sample[:, 0].max()),
            float(sample[:, 1].min()), float(sample[:, 1].max()))


class Frame:
    """World (x, y) in metres -> image (u, v) in pixels, for one scene."""

    def __init__(self, scene: str):
        self.scene = scene
        scene_dir = DATA / "scenes" / scene
        self.image = Image.open(scene_dir / "map.jpg").convert("RGB")
        self.width, self.height = self.image.size
        self.x0, self.x1, self.y0, self.y1 = ply_xy_bounds(scene_dir / "map.ply")
        self.px_per_m_x = (self.width - 1) / (self.x1 - self.x0)
        # v grows downward while world y grows upward, hence the flip below.
        self.px_per_m_y = (self.height - 1) / (self.y1 - self.y0)

    def __call__(self, x, y):
        u = (np.asarray(x, float) - self.x0) * self.px_per_m_x
        v = (self.y1 - np.asarray(y, float)) * self.px_per_m_y
        return u, v

    @property
    def anisotropy(self) -> float:
        return abs(self.px_per_m_x - self.px_per_m_y) / self.px_per_m_x * 100.0


def landmark_check(frame: Frame) -> str:
    """Re-run the check that pinned the y-flip, so a mis-registered figure says so.

    arabic_room only, and deliberately only ONE landmark: its 1.45 x 2.17 m carpet renders as
    a large uniform LIGHT rectangle against the stipple, at world (-2.42, -0.05) -- left of
    centre, mid-room. Get the flip wrong and that lands in the dark seating area at the far
    end, so the grey inverts.

    Furniture centres were tried as landmarks and dropped: this is a line drawing, so a sofa
    is dark EDGES around a light interior, and a patch at its centre measures the fill style
    rather than the registration. It read "suspect" under a transform that was fine.
    """
    if frame.scene != "arabic_room":
        return "no landmark for this scene; trust the anisotropy only"
    grey = np.asarray(frame.image.convert("L"), float)
    background = float(grey.mean())
    u, v = frame(-2.42, -0.05)
    u, v = int(round(float(u))), int(round(float(v)))
    patch = grey[max(v - 7, 0):v + 7, max(u - 7, 0):u + 7]
    carpet = float(patch.mean()) if patch.size else background
    ok = carpet > background + 30
    return (f"carpet {carpet:.0f} vs bg {background:.0f} "
            f"-> {'OK' if ok else 'SUSPECT — check the flip'}")


def draw_box(ax, frame: Frame, centre, size, colour, *, dashed=False, label=None, lw=1.8,
             zorder=4):
    """An axis-aligned footprint. Yaw is ignored: obj_map publishes axis-aligned boxes and
    that is what every score in this repo is computed on."""
    if centre is None or size is None or len(centre) < 2 or len(size) < 2:
        return
    u0, v0 = frame(centre[0] - size[0] / 2.0, centre[1] + size[1] / 2.0)
    u1, v1 = frame(centre[0] + size[0] / 2.0, centre[1] - size[1] / 2.0)
    ax.add_patch(Rectangle((float(u0), float(v0)), float(u1 - u0), float(v1 - v0),
                           fill=False, edgecolor=colour, linewidth=lw,
                           linestyle="--" if dashed else "-", label=label, zorder=zorder))


# -- data -------------------------------------------------------------------

def place_label(ax, frame: Frame, placed: list, u: float, v: float, text: str) -> None:
    """Object name near its box, kept on the canvas and off its neighbours.

    Both matter once the type is big enough to read: a label on an object at the wall runs off
    the edge and loses its first characters, and rooms cluster small objects (a tray ON a
    table, a hookah beside it) so several names land on the same few pixels. Nudged down until
    clear, and pinned inside the frame -- a name that cannot be read identifies nothing.
    """
    if not text:
        return
    y = v - 9.0
    while any(abs(y - py) < 12.0 and abs(u - px) < 60.0 for px, py in placed):
        y += 12.0
    placed.append((u, y))
    # Flip the anchor near an edge rather than clamping the position. A CENTRED label on an
    # object against the left wall still runs off the canvas -- half its width lies outside --
    # so "potted plant" rendered as "otted plant". Anchoring left there makes it grow inward.
    if u < frame.width * 0.14:
        align, x = "left", max(u - 8.0, 2.0)
    elif u > frame.width * 0.86:
        align, x = "right", min(u + 8.0, frame.width - 2.0)
    else:
        align, x = "center", u
    ax.text(x, y, text, fontsize=LABEL_PT, color=GT, ha=align, zorder=9, weight="bold",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])


def load_question(scene: str, category: int, qid: str) -> dict:
    path = (DATA / "benchmark" / scene / f"category_{category}"
            / f"{scene}_category{category}_qa.json")
    questions = json.load(open(path, encoding="utf-8")).get("questions") or []
    entry = next((q for q in questions if str(q.get("id")) == qid), None)
    if entry is None:
        raise SystemExit(f"{scene} has no {qid} in category {category} "
                         f"(has {[q.get('id') for q in questions]})")
    return entry


def run_path(best_view_dir, report: Path | None = None) -> Path | None:
    """A row's run directory on the host. Container `/data/...` paths are rewritten.

    Falls back to the REPORT's own name when the recorded directory is gone. The crops folder
    is named after the report's filename stem *at run time*, so renaming a report afterwards --
    or renaming the crops folder to keep two sweeps apart -- silently breaks every row's
    recorded path. Everything downstream then reads an empty map and the figure claims the run
    mapped nothing, while the files sit under the new name. Retrying there costs one stat.
    """
    if not best_view_dir:
        return None
    path = str(best_view_dir)
    if path.startswith("/data/"):
        path = str(DATA / path[len("/data/"):])
    resolved = Path(path)
    if resolved.is_dir() or report is None:
        return resolved
    tail = resolved.parts[-2:]                     # <scene>/<qid>-<question>
    if len(tail) < 2:
        return resolved
    alt = DATA / "crops" / Path(report).stem / tail[0] / tail[1]
    return alt if alt.is_dir() else resolved


def _read_json(best_view_dir, name: str, report: Path | None = None) -> dict:
    run = run_path(best_view_dir, report)
    if run is None:
        return {}
    try:
        return json.load(open(run / name, encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {}


def load_grid(best_view_dir, report: Path | None = None) -> dict | None:
    """The floor grid a question accumulated, or None if it was never recorded.

    Resolved through `run_path` like every other run artifact, so a report moved between
    sweeps finds its own crops instead of quietly drawing an empty room.
    """
    run = run_path(best_view_dir, report)
    if run is None:
        return None
    npz = run / "traversable_area.npz"
    if not npz.is_file():
        return None
    try:
        with np.load(npz) as data:
            return {"state": data["state"], "cell_m": float(data["cell_m"]),
                    "corner": np.asarray(data["corner"], dtype=float)}
    except (OSError, ValueError, KeyError):
        return None


def load_obj_map(best_view_dir, report: Path | None = None) -> dict:
    """The predicted map for a row."""
    return _read_json(best_view_dir, "obj_map.json", report)


def load_plan(best_view_dir, report: Path | None = None) -> dict:
    """The route the model chose, from instruction_plan.json beside obj_map.json.

    THIS is the category-3 answer. `row["waypoints"]` is only the topic traffic that carried
    it: TARE's exploration goals land on the same topic, the reasoner republishes each target
    at 2 Hz because every republish resets the waypoint converter's arrival latch, and the
    stall ladder pokes recovery points at it too. One run put 215 messages and 113 distinct
    positions on that topic for a route of two waypoints, 64 of them outside the room.

    Empty when the file is absent -- an older report, or a run that died before it planned,
    still has ground truth worth drawing.
    """
    return _read_json(best_view_dir, "instruction_plan.json", report)


def drive_start(row: dict, route: list) -> float | None:
    """When the reasoner took over from TARE, in the trajectory's own timebase.

    The trajectory covers the WHOLE question, which is right for scoring -- a constraint
    satisfied while exploring still counts -- and wrong for reading: one run wandered 48 m
    exploring and then drove 4.7 m following the plan, and a single line through both says
    nothing about either.

    Prefers `drive_started_s`, which the orchestrator records at the `execute` transition.
    Falls back to the first published waypoint matching a planned one, so a report written
    before that field existed still splits instead of drawing the wander as the route.
    None when neither is available, and the caller then draws the trajectory whole.
    """
    recorded = row.get("drive_started_s")
    if isinstance(recorded, (int, float)):
        return float(recorded)
    for wp in row.get("waypoints") or []:
        if any(abs(wp[1] - w.get("x", 1e9)) < 0.02 and abs(wp[2] - w.get("y", 1e9)) < 0.02
               for w in route):
            return float(wp[0])
    return None


def find_row(report: Path, scene: str, qid: str) -> dict | None:
    if not report or not report.is_file():
        return None
    try:
        payload = json.load(open(report, encoding="utf-8"))
    except (OSError, ValueError):
        return None
    rows = payload.get("results", payload) if isinstance(payload, dict) else payload
    return next((r for r in rows
                 if r.get("scene") == scene and str(r.get("id")) == qid), None)


def plan_panel(plan: dict) -> str:
    """The model's own words, as a block to print under the floor plan.

    Its reason, then one line per waypoint: the number that appears on the marker, the role,
    the coordinate, the map objects it cited, the phrase it came from, and -- from the `drive`
    block -- whether the robot actually got there.

    A panel rather than labels on the map: prose over a floor plan collides with the GT object
    names already there, and gets worse the more crowded the room is. Here it stays readable
    and the numbers tie each line back to its marker.
    """
    route = plan.get("route") or []
    if not route:
        return ""
    # `model_reply` is the raw answer; `route` is the same thing after validation. Prefer the
    # model's own phrasing, and fall back so a `fallback` plan still describes what drove.
    reply = plan.get("model_reply") or {}
    said = {i: w for i, w in enumerate(reply.get("waypoints") or [])}
    drive = plan.get("drive") or []

    lines = []
    reason = str(reply.get("reason") or "").strip()
    if reason:
        head = f'VLM plan · "{reason}"'
        lines += textwrap.wrap(head, width=108, subsequent_indent="    ")
    for i, wp in enumerate(route):
        why = str((said.get(i) or wp).get("why") or "").strip()
        if len(why) > 40:
            why = why[:39] + "…"
        ids = ",".join(wp.get("object_ids") or []) or "-"
        row = (f"  {i + 1}  {wp.get('role', '?'):<4} "
               f"({wp.get('x', 0.0):6.2f},{wp.get('y', 0.0):6.2f})  "
               f"obj {ids:<6} · {why:<40}")
        leg = drive[i] if i < len(drive) else None
        if isinstance(leg, dict) and leg.get("closest_m") is not None:
            # Plain distances, no verdict. A goal is scored on where the robot ENDED, so show
            # that first and the best it ever managed second; a pass constraint is credited
            # anywhere on the trajectory, so for those the closest is what matters.
            if leg.get("final_m") is not None:
                row += f"  ended {leg['final_m']:.2f} m"
                if abs(leg["final_m"] - leg["closest_m"]) >= 0.05:
                    row += f" (closest {leg['closest_m']:.2f})"
            else:
                row += f"  closest {leg['closest_m']:.2f} m"
            # Three separate facts, so three separate fields. They used to be concatenated
            # as `snap 0.07x4`, which reads as multiplication and is really "moved the
            # waypoint 7 cm" and "re-aimed 4 times".
            if leg.get("snap_m"):
                row += f" · snap {leg['snap_m']:.2f} m"
                if (leg.get("snaps") or 0) > 1:
                    row += f" ({leg['snaps']} aims)"
            off = leg.get("final_target_m")
            # Where the robot stopped relative to what we AIMED at, as opposed to what the
            # model asked for. When the snap moved nothing the two are the same fact, and
            # printing both just invites reading a difference that is not there.
            if off is not None and leg.get("final_m") is not None \
                    and abs(off - leg["final_m"]) >= 0.05:
                row += f" · {off:.2f} m off aim"
            arrived = leg.get("arrived_at")
            row += f" · reached @{'waypoint' if arrived == 'goal' else 'aim'}" if arrived \
                else " · not reached"
        lines.append(row.rstrip())
    return "\n".join(lines)


# -- drawing ----------------------------------------------------------------

def plot(scene: str, category: int, qid: str, row: dict | None, out: Path,
         *, show_published: bool = False, report: Path | None = None) -> str:
    frame = Frame(scene)
    entry = load_question(scene, category, qid)
    fig, ax = plt.subplots(figsize=(frame.width / 100 * 1.6, frame.height / 100 * 1.6), dpi=110)
    ax.imshow(frame.image)
    ax.set_xlim(0, frame.width)
    ax.set_ylim(frame.height, 0)
    # Freeze the limits to the image. Every add_patch/plot below updates the data limits, so
    # without this a single object mapped outside the room re-autoscales the axes, the floor
    # plan shrinks, and the stray is drawn in fresh canvas as though it were part of the scene.
    ax.set_autoscale_on(False)
    ax.axis("off")
    missing: list[str] = []
    # Bound here, not only in the `row` branch below: the panel at the end reads it, and a
    # ground-truth-only figure never enters that branch.
    plan: dict = {}
    seen: set[str] = set()
    placed: list[tuple[float, float]] = []

    def once(name):
        """One legend entry per layer, not one per drawn item."""
        if name in seen:
            return None
        seen.add(name)
        return name

    # -- ground truth objects, split by role. A `reference` is a thing to FIND so the
    # instruction can be grounded; it is never a place to drive to, and treating the two
    # alike is exactly the mistake the current planner makes.
    objects = entry.get("objects") or {}
    # `objects` (category 3) carries the role split; categories 1-2 instead name an `answer`
    # and its `anchors`. Truthiness matters here: an empty dict is still a dict, and testing
    # the type alone silently drew no ground truth at all for every category-2 figure.
    if objects and isinstance(objects, dict):
        for obj in objects.values():
            reference = str(obj.get("role", "")).lower() == "reference"
            draw_box(ax, frame, obj.get("center"), obj.get("size"),
                     GT_REFERENCE if reference else GT_CONSTRAINT, dashed=reference, zorder=8,
                     label=once("GT reference object" if reference else "GT constraint object"))
            u, v = frame(obj["center"][0], obj["center"][1])
            place_label(ax, frame, placed, float(u), float(v), obj.get("label", ""))
    elif isinstance(entry.get("answer"), dict):
        answer = entry["answer"]
        draw_box(ax, frame, answer.get("center"), answer.get("size"), GT_CONSTRAINT,
                 lw=2.4, zorder=8, label=once("GT answer object"))
        for anchor in entry.get("anchors") or []:
            draw_box(ax, frame, anchor.get("center"), anchor.get("size"), GT_REFERENCE,
                     dashed=True, zorder=7, label=once("GT anchor"))

    # -- ground truth route and constraints (category 3)
    gt = entry.get("gt") or {}
    for i, cons in enumerate(list(gt.get("pass_near") or []) + ([gt["goal"]] if gt.get("goal") else []), 1):
        u, v = frame(cons["center"][0], cons["center"][1])
        radius_px = cons.get("radius", 1.5) * frame.px_per_m_x
        ax.add_patch(Circle((float(u), float(v)), radius_px, fill=False, edgecolor=GT_ROUTE,
                            linewidth=1.4, linestyle=":", zorder=5,
                            label=once("GT constraint radius")))
        ax.text(float(u), float(v), str(i), fontsize=NUMBER_PT, color=GT, weight="bold",
                ha="center", va="center", zorder=9,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    for zone in gt.get("avoid") or []:
        pts = [frame(px, py) for px, py in zone.get("polygon", [])]
        if pts:
            ax.add_patch(Polygon([(float(a), float(b)) for a, b in pts], closed=True,
                                 facecolor=GT_ROUTE, alpha=0.18, edgecolor=GT_ROUTE,
                                 zorder=3, label=once("GT avoid zone")))
    polyline = (entry.get("reference_trajectory") or {}).get("polyline") or []
    if polyline:
        arr = np.asarray(polyline, float)
        u, v = frame(arr[:, 0], arr[:, 1])
        ax.plot(u, v, color=GT_ROUTE, linewidth=2.4, zorder=5, label=once("GT route"))
        ax.plot(u[0], v[0], "o", color=GT_ROUTE, markersize=7, zorder=6)
        ax.plot(u[-1], v[-1], "*", color=GT_ROUTE, markersize=15, zorder=6)
    elif category == 3:
        missing.append("no GT reference route in the benchmark")

    # -- prediction
    if row is None:
        missing.append("no report row — ground truth only")
    else:
        obj_map = load_obj_map(row.get("best_view_dir"), report)
        if obj_map:
            for obj in obj_map.values():
                box = obj.get("bbox3d") or {}
                draw_box(ax, frame, box.get("center"), box.get("extent"), PRED_OBJECT,
                         lw=1.2, label=once("predicted object"))
        else:
            missing.append("no obj_map.json — nothing was mapped")

        marker = row.get("marker")
        if category == 2 and marker and not marker.get("placeholder"):
            draw_box(ax, frame, marker.get("center"), marker.get("size"), PRED_ROUTE,
                     lw=2.6, label=once("predicted answer"))
        elif category == 2:
            missing.append("no answer marker was published")

        # The category-3 answer: the places the model chose, and how the drive went.
        # NOT row["waypoints"] -- see load_plan for why that is traffic rather than route.
        plan = load_plan(row.get("best_view_dir"), report) if category == 3 else {}
        route = plan.get("route") or []
        drive = {i: leg for i, leg in enumerate(plan.get("drive") or [])}

        traj = row.get("trajectory") or []
        if traj:
            began = drive_start(row, route) if category == 3 else None
            explored = [p for p in traj if began is not None and p[0] < began]
            driven = [p for p in traj if began is None or p[0] >= began]

            # Exploration is context, not the answer: it says where the ten-minute budget
            # went and why the map is as sparse as it is, so it stays -- behind everything.
            if len(explored) > 1:
                arr = np.asarray([(p[1], p[2]) for p in explored], float)
                u, v = frame(arr[:, 0], arr[:, 1])
                ax.plot(u, v, color=PRED_ROUTE, linewidth=0.9, alpha=0.3, zorder=4,
                        label=once("exploration path"))
                ax.plot(u[0], v[0], "o", color=PRED_ROUTE, markersize=5, alpha=0.4, zorder=4)

            if len(driven) > 1:
                arr = np.asarray([(p[1], p[2]) for p in driven], float)
                u, v = frame(arr[:, 0], arr[:, 1])
                ax.plot(u, v, color=PRED_ROUTE, linewidth=1.8, alpha=0.95, zorder=6,
                        label=once("driven route" if began is not None else "driven path"))
                ax.plot(u[0], v[0], "o", color=PRED_ROUTE, markersize=7, zorder=7)
                # The final pose is what the goal constraint is scored on.
                ax.plot(u[-1], v[-1], "*", color=PRED_ROUTE, markersize=15, zorder=7)
        elif category == 3:
            missing.append("no trajectory recorded — the robot never moved")

        outside = sum(
            1 for o in obj_map.values()
            if (o.get("bbox3d") or {}).get("center")
            and not (frame.x0 <= o["bbox3d"]["center"][0] <= frame.x1
                     and frame.y0 <= o["bbox3d"]["center"][1] <= frame.y1))
        if outside:
            missing.append(f"{outside} predicted object(s) mapped outside the room")

        if route:
            arr = np.asarray([(w["x"], w["y"]) for w in route], float)
            u, v = frame(arr[:, 0], arr[:, 1])
            # Dashed, so it reads as intent against the solid GT route and the driven path.
            ax.plot(u, v, "--", color=WAYPOINT, linewidth=2.2, alpha=0.9, zorder=9,
                    label=once("VLM route"))
            for i, (wp, uu, vv) in enumerate(zip(route, u, v)):
                goal = wp.get("role") == "goal"
                # reach_m is where the robot was allowed to stop, so drawing it makes
                # "stopped 1.19 m short" visible instead of arithmetic.
                if wp.get("reach_m"):
                    ax.add_patch(Circle((uu, vv), wp["reach_m"] * frame.px_per_m_x,
                                        fill=False, color=WAYPOINT, alpha=0.55,
                                        linestyle=":", zorder=9))
                # Big, and edged in white: these sit on a stippled floor plan and are the
                # thing the whole figure is for.
                ax.plot(uu, vv, "*" if goal else "o", color=WAYPOINT,
                        markersize=30 if goal else 17, markeredgecolor="white",
                        markeredgewidth=1.6 if goal else 2.0, zorder=10,
                        label=once("VLM goal" if goal else "VLM waypoint"))
                # Order is scored, so the plot has to show it.
                ax.text(uu, vv, str(i + 1), fontsize=NUMBER_PT, color="white",
                        weight="bold", ha="center", va="center", zorder=13)
                leg = drive.get(i) or {}

                # The chain this figure exists to show: what the model asked for (the marker
                # above), what we actually aimed at after snapping onto known floor, and where
                # the robot ended up. Reading those three as two segments is the difference
                # between "the plan was wrong" and "the plan was right and the base autonomy
                # re-aimed us"; with the snap clearance at zero the second is now the common
                # failure, and it is invisible if only the first and last are drawn.
                published = leg.get("published")
                aim = None
                if published and leg.get("snap_m"):
                    # Skipped at snap_m == 0: the published point IS the waypoint, and a second
                    # marker on top of the first only adds clutter.
                    pu, pv = frame(published[0], published[1])
                    aim = (float(pu), float(pv))
                    # This segment IS the snap, and its length is snap_m.
                    ax.plot([uu, aim[0]], [vv, aim[1]], "-", color=SNAP,
                            linewidth=1.8, alpha=0.95, zorder=11)
                    # ON TOP of the waypoint marker, not under it. A good snap is a few
                    # centimetres, which at ~75 px/m is a handful of pixels -- entirely inside
                    # the 17 px disc. Drawn beneath, the one layer this overlay was extended to
                    # show would be invisible exactly when the snap is working.
                    ax.plot(aim[0], aim[1], "D", color=SNAP, markersize=7,
                            markeredgecolor="white", markeredgewidth=1.2, zorder=12,
                            label=once("snapped aim point"))

                final = leg.get("final_pose")
                if final:
                    fu, fv = frame(final[0], final[1])
                    # Dotted back to whatever we aimed at, so the segment reads as
                    # final_target_m -- how much of the error the DRIVE owns, as opposed to
                    # the snap.
                    su, sv = aim if aim else (uu, vv)
                    ax.plot([su, float(fu)], [sv, float(fv)], ":", color=PRED_ROUTE,
                            linewidth=1.0, alpha=0.7, zorder=6)
                    ax.plot(float(fu), float(fv), "o", color=PRED_ROUTE, markersize=7,
                            markeredgecolor="white", markeredgewidth=1.0, zorder=10,
                            label=once("robot stopped"))

                closest = leg.get("closest_pose")
                if closest and not leg.get("reached"):
                    cu, cv = frame(closest[0], closest[1])
                    # The best it ever managed, which for a pass constraint is what scores --
                    # only worth drawing when it differs from where it ended.
                    if not final or math.dist(closest, final) >= 0.05:
                        ax.plot(float(cu), float(cv), "x", color=PRED_ROUTE, markersize=7,
                                zorder=8, label=once("closest approach"))
        elif category == 3:
            missing.append("no instruction_plan.json — the reasoner never planned a route")

        # The raw topic stream, off by default: it answers "did the planner emit anything at
        # all", not "where was it going".
        if show_published:
            for wp in row.get("waypoints") or []:
                u, v = frame(wp[1], wp[2])
                ax.plot(float(u), float(v), "x", color=PRED_ROUTE, markersize=5, alpha=0.35,
                        zorder=4, label=once("raw waypoint topic"))

    scale = {1: "/1", 2: "/2", 3: "/6"}.get(category, "")
    score = row.get("score") if row else None
    head = f"{scene}  {qid}   {entry.get('question', '')}"
    sub = "ground truth only" if row is None else (
        f"score {score:.2f}{scale}" if isinstance(score, (int, float)) else "no score")
    if row and row.get("note"):
        sub += f"   ({row['note']})"
    ax.set_title(f"{head}\n{sub}", fontsize=9, loc="left")

    # Absent and wrong are the two things this figure exists to tell apart, so say which.
    if missing:
        ax.text(0.01, 0.01, "\n".join("• " + m for m in missing), transform=ax.transAxes,
                fontsize=8, color="#7a0d0d", va="bottom",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="#7a0d0d", pad=4))
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # markerscale: the VLM markers are deliberately oversized ON THE MAP, which
        # would otherwise burst their legend row.
        ax.legend(handles, labels, loc="upper right", fontsize=7, framealpha=0.9,
                  markerscale=0.6)

    # Clip every artist to the image. Objects DO get mapped outside the room -- bleed, or a
    # detection behind a wall -- and without this bbox_inches="tight" grows the canvas around
    # them, so the floor plan shrinks and the stray box looks like part of the scene.
    ax.set_xlim(0, frame.width)
    ax.set_ylim(frame.height, 0)
    for artist in list(ax.patches) + list(ax.lines) + list(ax.texts):
        artist.set_clip_path(ax.patch)
        artist.set_clip_on(True)
    # The model's own answer, printed under the plot. fig.text (not ax.text) so it sits
    # outside the floor plan: bbox_inches="tight" grows the canvas to include it, and the
    # clip loop above only touches ax artists, so nothing crops it.
    panel = plan_panel(plan) if category == 3 else ""
    if panel:
        fig.text(0.01, -0.012, panel, transform=ax.transAxes, family="monospace",
                 fontsize=7.5, va="top", ha="left", color="#222222",
                 bbox=dict(facecolor="white", edgecolor=WAYPOINT, linewidth=1.2, pad=5))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return (f"{out}  [{frame.px_per_m_x:.1f} px/m x, {frame.px_per_m_y:.1f} px/m y, "
            f"anisotropy {frame.anisotropy:.1f}%]  {landmark_check(frame)}")


# -- the floor the snap could see -------------------------------------------

def plot_traversable(scene: str, category: int, qid: str, row: dict | None, out: Path,
                     *, report: Path | None = None) -> str:
    """Companion figure: the accumulated floor grid, on the same axes as the main overlay.

    The main figure shows where the robot was aimed and where it went. It cannot show WHY a
    waypoint was moved, because the reason is an absence -- floor that was never observed. On
    hotel_room_1 Q04 the run recorded 4776 free cells, a healthy-looking number, while the
    floor beside the target had never been seen at all; that hole is what a 2.82 m snap was
    reaching around, and it is invisible in every other artifact.

    UNKNOWN is left transparent rather than given a colour. Unobserved floor and floor known
    to be blocked fail a snap for opposite reasons -- one says "go and look", the other says
    "that is furniture" -- and drawing them alike would erase the only distinction this figure
    exists to make.
    """
    frame = Frame(scene)
    entry = load_question(scene, category, qid)
    grid = load_grid((row or {}).get("best_view_dir"), report)
    plan = load_plan((row or {}).get("best_view_dir"), report)

    fig, ax = plt.subplots(figsize=(frame.width / 100 * 1.6, frame.height / 100 * 1.6), dpi=110)
    ax.imshow(frame.image)
    ax.set_xlim(0, frame.width)
    ax.set_ylim(frame.height, 0)
    ax.set_autoscale_on(False)
    ax.axis("off")
    missing: list[str] = []
    free_cells = 0

    if grid is None:
        missing.append("no traversable_area.npz — this run predates the grid dump")
    else:
        state, cell, corner = grid["state"], grid["cell_m"], grid["corner"]
        free_cells = int((state == 1).sum())
        # Scattered squares rather than an imshow: the grid is world-axis-aligned while the
        # backdrop is a photograph of the same room, and resampling one onto the other blurs
        # exactly the seen/unseen boundary that matters here.
        for value, colour, alpha, name in ((1, "#2a9d8f", 0.55, "floor seen FREE"),
                                           (2, "#4a4a4a", 0.75, "seen OBSTACLE")):
            cells = np.argwhere(state == value)
            if not cells.size:
                continue
            u, v = frame(corner[0] + (cells[:, 0] + 0.5) * cell,
                         corner[1] + (cells[:, 1] + 0.5) * cell)
            ax.scatter(u, v, s=2.0, c=colour, marker="s", linewidths=0, alpha=alpha,
                       label=name, zorder=3)

    if row and row.get("trajectory"):
        arr = np.asarray([(p[1], p[2]) for p in row["trajectory"]], float)
        u, v = frame(arr[:, 0], arr[:, 1])
        # The path is the explanation for every hole: a region is UNKNOWN because the robot
        # never went there and never looked.
        ax.plot(u, v, color=PRED, linewidth=1.4, alpha=0.9, zorder=5, label="driven path")

    seen_label: set = set()
    for leg in plan.get("drive") or []:
        want, got = leg.get("waypoint"), leg.get("published")
        if not (want and got):
            continue
        wu, wv = frame(want[0], want[1])
        gu, gv = frame(got[0], got[1])
        # Same three-point chain the main overlay uses, and the same colours: what the model
        # asked for, what we aimed at, and the gap between them.
        ax.annotate("", xy=(float(gu), float(gv)), xytext=(float(wu), float(wv)),
                    arrowprops=dict(arrowstyle="-|>", color=SNAP, lw=2.0,
                                    shrinkA=0, shrinkB=0), zorder=8)
        lbl = None if "VLM waypoint" in seen_label else "VLM waypoint"
        ax.plot(float(wu), float(wv), "o", color=WAYPOINT, markersize=9,
                markeredgecolor="white", markeredgewidth=1.0, zorder=9, label=lbl)
        lbl2 = None if "VLM waypoint" in seen_label else "published (snapped)"
        ax.plot(float(gu), float(gv), "D", color=SNAP, markersize=7,
                markeredgecolor="white", markeredgewidth=1.0, zorder=9, label=lbl2)
        seen_label.add("VLM waypoint")
        ax.text(float((wu + gu) / 2), float((wv + gv) / 2) - 9,
                f"snap {float(leg.get('snap_m', 0.0)):.2f} m", fontsize=LABEL_PT, color=SNAP,
                weight="bold", ha="center", zorder=9,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    gt = entry.get("gt") or {}
    for i, cons in enumerate(list(gt.get("pass_near") or [])
                             + ([gt["goal"]] if gt.get("goal") else []), 1):
        u, v = frame(cons["center"][0], cons["center"][1])
        lbl = None if "GT constraint radius" in seen_label else "GT constraint radius"
        seen_label.add("GT constraint radius")
        ax.add_patch(Circle((float(u), float(v)), cons.get("radius", 1.5) * frame.px_per_m_x,
                            fill=False, edgecolor=GT, linewidth=1.4, linestyle=":", zorder=6,
                            label=lbl))
        ax.text(float(u), float(v), str(i), fontsize=NUMBER_PT, color=GT, weight="bold",
                ha="center", va="center", zorder=9,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    ax.set_title(f"{scene}  {qid}   the floor the snap could see — {free_cells} free cells\n"
                 f"{entry.get('question', '')}", fontsize=9, loc="left")
    if missing:
        ax.text(0.01, 0.01, "\n".join("• " + m for m in missing), transform=ax.transAxes,
                fontsize=8, color="#7a0d0d", va="bottom",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="#7a0d0d", pad=4))
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=7, framealpha=0.9,
                  markerscale=0.6)

    ax.set_xlim(0, frame.width)
    ax.set_ylim(frame.height, 0)
    for artist in (list(ax.patches) + list(ax.lines) + list(ax.texts)
                   + list(ax.collections)):
        artist.set_clip_path(ax.patch)
        artist.set_clip_on(True)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return f"{out}  [{free_cells} free cells]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path, default=None,
                    help="sweep report; omit to draw ground truth alone")
    ap.add_argument("--scene", default=None, help="omit with --report to do every row")
    ap.add_argument("--qid", default=None)
    ap.add_argument("--category", type=int, default=None, choices=(1, 2, 3))
    ap.add_argument("--out-dir", type=Path, default=DATA / "runs" / "overlays")
    ap.add_argument("--show-published", action="store_true",
                    help="also mark every /way_point_with_heading message -- raw topic traffic, "
                         "not the per-leg aim point the overlay already draws. Off by default: "
                         "TARE's exploration goals, the reasoner's 2 Hz republishes and the "
                         "stall ladder all land on that topic, so it is traffic rather than "
                         "route. Useful to answer 'did the planner emit anything at all'.")
    args = ap.parse_args()

    targets: list[tuple[str, int, str, dict | None]] = []
    if args.report and args.report.is_file():
        payload = json.load(open(args.report, encoding="utf-8"))
        rows = payload.get("results", payload) if isinstance(payload, dict) else payload
        for row in rows:
            if args.scene and row.get("scene") != args.scene:
                continue
            if args.qid and str(row.get("id")) != args.qid:
                continue
            targets.append((row["scene"], int(row.get("category", args.category or 2)),
                            str(row["id"]), row))
    else:
        # Ground truth alone. Useful before any run exists -- it is the reference set you
        # compare a later sweep against, and it is what proves the registration.
        if not args.category:
            ap.error("without a readable --report, give at least --category")
        scenes = ([args.scene] if args.scene else
                  sorted(p.name for p in (DATA / "benchmark").iterdir()
                         if (p / f"category_{args.category}").is_dir()))
        for scene in scenes:
            qa = (DATA / "benchmark" / scene / f"category_{args.category}"
                  / f"{scene}_category{args.category}_qa.json")
            if not qa.is_file():
                continue
            for entry in json.load(open(qa, encoding="utf-8")).get("questions") or []:
                if args.qid and str(entry.get("id")) != args.qid:
                    continue
                targets.append((scene, args.category, str(entry["id"]), None))

    if not targets:
        print("nothing matched", file=sys.stderr)
        return 1
    for scene, category, qid, row in targets:
        out = args.out_dir / f"{scene}_{qid}.png"
        try:
            print(plot(scene, category, qid, row, out,
                       show_published=args.show_published, report=args.report))
            # Category 3 only: the floor grid is what the snap reads, and no other category
            # has one. It is a second file rather than a panel because the two are read at
            # different moments -- "what happened" first, "why was it aimed there" after.
            if category == 3:
                print(plot_traversable(scene, category, qid, row,
                                       args.out_dir / f"{scene}_{qid}_traversable.png",
                                       report=args.report))
        except SystemExit as exc:
            print(f"{scene} {qid}: {exc}", file=sys.stderr)
        except (OSError, KeyError, ValueError) as exc:
            print(f"{scene} {qid}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
