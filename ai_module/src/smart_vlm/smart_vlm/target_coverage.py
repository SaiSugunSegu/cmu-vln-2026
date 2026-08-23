#!/usr/bin/env python3
"""target_coverage — the decision layer behind target_explorer, with no ROS in it.

Split out for the same reason mission_clock and report_utils are: every rule here is
arithmetic over two JSON documents (the mapper's object list, TARE's verdicts) and belongs
under `just test`, not under a live sim.

WHAT THE PROBLEM ACTUALLY IS
----------------------------
`object_reference_reasoner` answers from `obj_map.json`, and `serialize_map_to_dict` skips
any object whose `infer_centroid()` is None. `infer_centroid` weights each voxel by how many
distinct azimuths it has been seen from, so an object nobody circled has zero weight, no
centroid, and is *absent* from the map — not merely a loose box. It cannot be selected and it
cannot even be the fallback. So the first job of exploration is not box quality, it is
**making every target object exist**.

That gives the priority its shape: `published: false` objects (invisible to the answer path)
outrank published ones (box quality only). Distance orders the route inside a tier; it never
decides whether a goal is worked. Nothing is ever dropped — coverage stays mandatory, it is
only sequenced by what the map cannot answer without.

THE INVERSION
-------------
`view_bins[k]` is set when the robot stood such that the object lay at world azimuth
theta_k = -pi + (k + 0.5) * 2pi/n from it. To fill bin k the robot has to stand opposite:

    stand_at = centre - r * (cos theta_k, sin theta_k)

Coverage is by SECTOR, not by bin: six adjacent bins is a 108-degree arc a robot curving past
an object fills in one sweep — one viewpoint, not six. Sectors keep the rule general: a
free-standing object is pursued for the full 360 degrees, a window flat against a wall is
satisfied by the half that physically exists, and no scene-specific view count appears.

EVERY SECTOR REACHES A TERMINAL STATE
-------------------------------------
OPEN -> REQUESTED -> COVERED | BLOCKED. A sector is BLOCKED when TARE refused it at every
radius the escalation will try, or when the robot *stood there* and the object still did not
register from that side. Without the second, an occluded side stays OPEN forever and the
robot re-asks for a viewpoint it has already visited — which is the difference between
orbiting an object and oscillating in front of it.

WHAT TARE DOES WITH THE OUTPUT  (sensor_coverage_planner_ground.cpp)
--------------------------------------------------------------------
  * positions only, z = robot z; orientation is discarded (360-degree panorama)
  * the whole array is replaced per message; <= 8 poses (kMaxTargetViewPointNum)
  * ORDER IS THE PRIORITY SIGNAL: TARE drives at the first ACCEPTED pose and holds it until
    it moves more than kTargetViewPointSnapMaxDist (0.5 m)
  * a pose is accepted only if a candidate viewpoint — collision-free, in line of sight,
    graph-connected — lies within 0.5 m of it; otherwise `unreachable`. A pose INSIDE the 9 m
    lattice that is not a candidate square at all answers `unreachable` too: the robot is
    already there and still cannot stand on it, which is a fact about the world
  * `far` means outside the lattice entirely — purely a distance statement, so it is
    deliberately NOT reported (driving closer changes the answer); it pins the subspace
    EXPLORING instead so the global tour routes there
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from itertools import zip_longest

from sam_mapper.mapping_config import MappingConfig

COVERED, BLOCKED, OPEN = "covered", "blocked", "open"

#: What the tare_planner C++ falls back to when a scenario yaml omits these. Present only so
#: this module stays importable without ROS; target_explorer replaces both with the values
#: from tare_planner's scenario yaml, which is the single source of truth for them.
TARE_DEFAULTS = {"max_target_viewpoints": 8, "snap_max_dist_m": 0.5}

#: waypointXYRadius in the base autonomy's waypoint_converter — how close the robot must get
#: before it declares a waypoint reached and stops. Provided code we do not configure, so it
#: is a constant here rather than another knob.
WAYPOINT_XY_RADIUS_M = 0.3

#: Half the panorama's vertical FOV. The camera is 360x120 degrees
#: (sam_mapper/cloud_image_fusion.py), and `bounds_mode: clip` pins anything outside it to
#: row 0/639 where edge masks swallow it — so standing closer than |dz| / tan(60 deg)
#: mis-assigns the parts of an object that fall out of frame.
_HALF_VFOV_RAD = math.radians(60.0)

#: Emitted positions are snapped to this grid. It is far below TARE's 0.5 m snap tolerance,
#: so quantisation can never change which candidate viewpoint a request resolves to — but it
#: stops sub-decimetre centroid jitter from moving the request, which is what made TARE
#: re-adopt its target and made feedback attribution miss.
_POSITION_QUANTUM_M = 0.1

#: Identity tolerance for matching a verdict back to the request that produced it. TARE echoes
#: the requested position verbatim at six decimals, so this only absorbs JSON round-tripping.
_ECHO_EPS_M = 0.01

@dataclass
class Sector:
    """One 90-degree arc around a goal, and everything we know about trying to reach it."""
    index: int
    bins: tuple                     # the azimuth bins this sector spans
    #: How many times the request has been pushed further out after a refusal. A sector is
    #: only written off once these are spent, so "too close to stand there" cannot be
    #: mistaken for "nothing can see this side".
    step: int = 0
    #: Bins TARE evaluated in-horizon and could not place a viewpoint for. Never cleared:
    #: it is what makes the next request pick a different angle as well as a longer radius.
    refused: set = field(default_factory=set)
    #: TARE snapped this sector's current request onto a real candidate viewpoint and put it
    #: in its tour. Until then the robot has no reason to go there, so coming within
    #: arrival_radius means it drove past on the way to something else — see `arrived`.
    accepted: bool = False
    #: TARE returned ANY verdict for this sector's request, accepted or unreachable. A request
    #: in neither list is `far`, which is TARE's own answer to "is this within the local
    #: planning horizon", and is what `preempt` reads rather than re-deriving the lattice
    #: geometry in Python.
    verdict: bool = False
    #: The robot came within arrival_radius of this sector's ACCEPTED request. Latched.
    arrived: bool = False
    #: Mapper updates for this goal observed since arrival. Once this passes min_info_frames
    #: with the sector still unfilled, we stood there and it did not register.
    updates_since_arrival: int = 0
    blocked_reason: str = ""

    def state(self, filled: bool) -> str:
        if filled:
            return COVERED
        return BLOCKED if self.blocked_reason else OPEN


@dataclass
class Goal:
    """One physical object to inspect — possibly several tracked fragments of it.

    Keyed by centroid rather than by map id: a world merge renames the surviving object to
    the winner's obj_id[0] and absorbs the loser's ids, so an id-keyed goal would look like a
    brand new one the moment the map tidied itself up.
    """
    label: str
    center: tuple                   # cluster centroid, xyz
    extent: tuple                   # cluster bounding extent
    bins: list                      # union of the members' view_bins
    published: bool                 # any member good enough for /obj_map_json
    members: int
    sectors: list = field(default_factory=list)
    #: Consecutive mapper updates in which no cluster matched this goal. `describe_objects`
    #: reports EVERY tracked object every frame, so a run of absences means the mapper let go
    #: of the object -- it pruned it, or a world merge folded it into a neighbour.
    absences: int = 0
    #: Retired: out of `pending`, out of the request array, out of `coverage_complete`. Kept
    #: for the report and revived on a rematch, so a transient drop-out does not cost the
    #: sector history we paid to gather.
    dormant: bool = False
    #: When the deadlock guard last released this goal. A COOLDOWN, not a tally: it breaks the
    #: tie against the goal we just gave up on and then stops mattering, so route order does
    #: not decay into round-robin as a run goes on.
    deferred_at: float | None = None
    #: How many times that has happened, for the report only -- never for ranking.
    deferrals: int = 0
    #: Stamped from the caller's clock, not time.monotonic(): the model is driven with an
    #: injected `now` so tests are deterministic, and two clocks would make `age_s` nonsense.
    first_seen: float = 0.0

    @property
    def tier(self) -> int:
        """0 = the map cannot answer with this object at all. 1 = only its box can improve."""
        return 1 if self.published else 0

    @property
    def in_horizon(self) -> bool:
        """TARE has judged at least one of this goal's live requests, so it is close enough to
        judge. Everything else it stayed silent about is `far`."""
        return any(sector.verdict for sector in self.sectors)


class CoverageModel:
    """Tracks every target object's per-sector coverage and says where to stand next.

    Owns no clock of its own beyond what callers pass in, so tests drive it deterministically.
    """

    def __init__(self, params: dict, dimension_priors=None):
        self.p = params
        # The mapper's own per-class upper bounds, used to cap a bled extent before it becomes
        # a standoff. Injectable so a test can pin one class without touching the real table.
        self.dimension_priors = (dimension_priors if dimension_priors is not None
                                 else MappingConfig().dimension_priors)
        self.targets: set[str] = set()
        self.goals: dict[tuple, Goal] = {}
        self.committed = None
        self.committed_since = 0.0
        self._committed_signature = None
        #: Recent published requests, oldest first, each a list of
        #: (x, y, goal_key, sector_index, bin_index). A short HISTORY rather than just the
        #: latest array, because TARE's feedback runs at ~1 Hz (every 5th /registered_scan)
        #: against our 2 Hz publish — matching only the newest array silently discards every
        #: verdict that crossed a republish, which is most of them.
        self.emitted: deque = deque(maxlen=8)
        self.last_request: list[tuple] = []
        #: Latched value of `preempt`, and when its raw condition was first seen false. See
        #: `preempt` for why the raw condition cannot be published directly.
        self._preempt = False
        self._preempt_false_since = None

    # -- inputs ---------------------------------------------------------------

    def set_targets(self, labels: set[str]) -> bool:
        """Adopt a new question's labels. Returns True if anything changed.

        A re-arm wipes the map (map_node drops everything on a new run_id), so goals built
        from the old one are meaningless and go with it.
        """
        if labels == self.targets:
            return False
        self.targets = set(labels)
        self.reset_map_state()
        return True

    def reset_map_state(self) -> None:
        """Drop everything derived from the 3D map, keeping the target labels.

        Called on a new `run_id`, which is when map_node drops its own map (`_reset_map`).
        Goals, the commitment and the emitted history are all statements about objects that
        no longer exist, and a verdict attributed across that boundary would land on a sector
        of a goal that has nothing to do with the request it answers.
        """
        self.goals, self.committed, self._committed_signature = {}, None, None
        self.emitted.clear()
        self.last_request = []
        self._preempt, self._preempt_false_since = False, None

    def ingest(self, objects: list[dict], robot, now: float) -> None:
        """Fold one /exploration/object_targets frame into the persistent goals."""
        seen_keys = set()
        for cluster in self._cluster(self._admit(objects)):
            # `seen_keys` excludes goals another cluster already claimed this frame. Two
            # clusters in one frame are two objects — the mapper's own list separated them and
            # _cluster kept them apart — so letting both collapse onto the nearest goal would
            # undo that one step later, at the goal layer instead of the cluster layer.
            key = (self._match(cluster, claimed=seen_keys)
                   or (cluster["label"], _round(cluster["center"])))
            goal = self.goals.get(key)
            if goal is None:
                goal = Goal(label=cluster["label"], center=tuple(cluster["center"]),
                            extent=tuple(cluster["extent"]), bins=list(cluster["bins"]),
                            published=cluster["published"], members=cluster["members"],
                            first_seen=now)
                goal.sectors = self._build_sectors(len(goal.bins))
                self.goals[key] = goal
            else:
                goal.center = tuple(cluster["center"])
                goal.extent = tuple(cluster["extent"])
                goal.bins = list(cluster["bins"])
                # Latched: an object drops out of obj_map.json whenever regularize_shape has a
                # bad frame, and un-publishing it would re-promote a goal we already paid for.
                goal.published = goal.published or cluster["published"]
                goal.members = cluster["members"]
                if len(goal.sectors) != self._n_sectors() or not goal.sectors:
                    goal.sectors = self._build_sectors(len(goal.bins))
            goal.absences, goal.dormant = 0, False
            seen_keys.add(key)

        self._retire_absent(seen_keys)

        # Arrival patience counts MAPPER updates for this goal, not wall time: the question is
        # "did the object register from here", and only a frame in which the mapper saw it can
        # answer that. A goal absent from this frame has nothing to say either way.
        for key in seen_keys:
            goal = self.goals[key]
            patience = int(self.p["arrival_patience_updates"])
            for sector in goal.sectors:
                if not sector.arrived:
                    continue
                sector.updates_since_arrival += 1
                if self._filled(goal, sector) or sector.updates_since_arrival < patience:
                    continue
                # We stood there and the object did not register from that side. Escalate
                # rather than write it off outright: the commonest cause is the lidar's blind
                # cone below the sensor, which swallows floor-level objects at close range
                # (measured: stool 82% of detections with zero points) — and the cure for that
                # is standing further back, which is exactly what an escalation asks for.
                self._escalate(goal, sector, reason="stood-there-no-detection")
                sector.arrived, sector.updates_since_arrival = False, 0

        self.note_position(robot)
        self._recommit(now, robot)

    def _retire_absent(self, seen_keys: set) -> None:
        """Let go of goals the mapper has let go of.

        `describe_objects` reports every TRACKED object every frame, published or not, so a
        goal that no cluster matched is one the mapper no longer holds: it pruned the object
        (`valid_indices_regularized < 20 and inactive_frame > 5`), or a world merge folded it
        into a neighbour, or -- the case that bites hardest -- the object is unpublished and
        its `provisional_centroid` walked further in one frame than `_match` will reach, so
        the goal we are looking at is a stale copy of an object that now has a newer one.

        Without this, goals are immortal. A SAM false positive on a target label leaves a goal
        that is `pending` for the rest of the run, and since `coverage_complete` requires an
        empty pending list, ONE phantom disables the early stop permanently -- and the robot
        keeps being sent to a place where nothing is.

        Dormant rather than deleted: an object dropping out for a few frames is ordinary, and
        deleting would throw away the sectors we already paid to cover and start the orbit
        again from scratch on the rebound.
        """
        limit = int(self.p["goal_absence_limit"])
        for key, goal in self.goals.items():
            if key in seen_keys:
                continue
            goal.absences += 1
            if goal.absences >= limit:
                goal.dormant = True

    def note_position(self, robot) -> None:
        """Mark the sectors whose request the robot has now actually stood at.

        Without this a side that is reachable but occluded — or that SAM simply does not fire
        on from that angle — never leaves the work list, and the robot keeps being sent back
        to a spot it has already visited.
        """
        if robot is None:
            return
        radius = float(self.p["arrival_radius_m"])
        for x, y, key, sector_index, _bin_index in self.last_request:
            goal = self.goals.get(key)
            if goal is None or sector_index >= len(goal.sectors):
                continue
            sector = goal.sectors[sector_index]
            # `accepted` is the whole point of the guard. Being within arrival_radius of a
            # request TARE never placed means the robot clipped it on the way to something
            # else -- and treating a drive-by as an inspection escalates, then blocks, a
            # sector nobody ever went to look at.
            if not sector.accepted or sector.arrived:
                continue
            if math.hypot(x - robot[0], y - robot[1]) <= radius:
                sector.arrived = True
                sector.updates_since_arrival = 0

    def note_feedback(self, payload: dict) -> set:
        """Apply one /exploration/target_viewpoint_feedback message. Returns keys touched.

        Three verdicts, and the third is the silent one:

          accepted    — TARE snapped it onto a real candidate viewpoint and put it in its
                        tour. The robot has a reason to go there, which is what makes a later
                        arrival mean something.
          unreachable — the robot is already within the 9 m lattice and still cannot stand
                        there: no collision-free, in-line-of-sight, graph-connected candidate
                        within kTargetViewPointSnapMaxDist, or the square is not a candidate
                        at all. The only one that is a statement about the world, and the
                        reason wall-mounted objects can ever finish — two of a picture's four
                        sectors are behind the wall and answer this way every cycle.
          neither     — `far`: outside the lattice entirely. Purely a distance statement, so
                        TARE deliberately does not report it: driving closer changes the
                        answer. It does say the goal is out of reach for now, which is exactly
                        what `preempt` needs.
        """
        touched = set()
        live = {(key, sector_index) for _x, _y, key, sector_index, _b in self.last_request}
        for key, sector_index in live:                  # a fresh message re-answers everything
            sector = self._sector(key, sector_index)
            if sector is not None:
                sector.verdict = False

        for point in payload.get("accepted") or []:
            hit = self._attribute(point)
            if hit is None:
                continue
            sector = self._sector(hit[0], hit[1])
            if sector is not None:
                sector.accepted, sector.verdict = True, True

        for point in payload.get("unreachable") or []:
            hit = self._attribute(point)
            if hit is None:
                continue
            key, sector_index, bin_index = hit
            goal = self.goals.get(key)
            sector = self._sector(key, sector_index)
            if goal is None or sector is None:
                continue
            sector.verdict = True
            sector.refused.add(bin_index)
            self._escalate(goal, sector)
            touched.add(key)
        return touched

    def _sector(self, key, sector_index):
        goal = self.goals.get(key)
        if goal is None or sector_index >= len(goal.sectors):
            return None
        return goal.sectors[sector_index]

    # -- goal building --------------------------------------------------------

    def _admit(self, objects: list[dict]) -> list[dict]:
        """Target-labelled entries substantial enough to be worth driving to."""
        out = []
        for o in objects:
            if str(o.get("label", "")).lower() not in self.targets:
                continue
            if o.get("center") is None:
                continue
            if o.get("published"):
                # The mapper already judged this good enough for /obj_map_json, so it is by
                # definition good enough to go look at again — a stricter bar here than the
                # map's own would be incoherent. It measurably was: arabic_room's `book`
                # reached obj_map.json yet never became a goal, because a small object does
                # not clear min_voxels. The gate below exists only to keep UNPUBLISHED
                # candidates from being noise.
                out.append(o)
                continue
            if (o.get("life", 0) <= self.p["min_life"]
                    or o.get("info_frames", 0) < self.p["min_info_frames"]
                    or o.get("voxels_total", 0) < self.p["min_voxels"]):
                continue
            out.append(o)
        return out

    def _cluster(self, objects: list[dict]) -> list[dict]:
        """Fuse same-label entries the mapper would have merged if it could.

        Unpublished objects never enter world merge — that path needs a non-None centroid on
        BOTH sides, and an under-observed object has none, which is exactly the object we are
        chasing. Left alone, several fragments of one real object become several goals a few
        tens of centimetres apart and the robot re-inspects one object repeatedly.
        """
        clusters: list[dict] = []
        for o in sorted(objects, key=lambda x: -x.get("voxels_total", 0)):
            center, extent = o["center"], o.get("extent") or [0.0, 0.0, 0.0]
            weight = max(int(o.get("voxels_total", 0)), 1)
            for c in clusters:
                if c["label"] != o["label"]:
                    continue
                # Two PUBLISHED entries have both been through the mapper's world merge,
                # which refuses to fuse ids SAM saw in the same frame (`block_covisible`)
                # precisely because distance alone fused 0.44 m-spaced pillows. Re-fusing them
                # here would undo that guard and leave one goal orbiting the midpoint of two
                # real objects. Clustering exists only for UNPUBLISHED fragments, which world
                # merge cannot see at all — it needs a non-None centroid on both sides.
                if c["published"] and o.get("published"):
                    continue
                if _dist(center, c["center"]) > self._merge_threshold(extent, c["extent"]):
                    continue
                # zip_longest, not zip: a fragment carrying a shorter (or absent) bin vector
                # would otherwise truncate the union to [], leaving a goal that yields no
                # viewpoint and can never be satisfied.
                c["bins"] = [bool(a) or bool(b) for a, b in
                             zip_longest(c["bins"], self._bins_of(o), fillvalue=False)]
                # Voxel-weighted, not first-wins: the fragments are pieces of one surface, and
                # aiming at the largest piece's centroid puts the inspection circle off-centre
                # by however far the pieces are apart.
                total = c["weight"] + weight
                c["center"] = tuple((cc * c["weight"] + oc * weight) / total
                                    for cc, oc in zip(c["center"][:3], center[:3]))
                c["extent"] = tuple(max(ce, oe) for ce, oe in
                                    zip_longest(c["extent"][:3], extent[:3], fillvalue=0.0))
                c["weight"] = total
                c["published"] = c["published"] or bool(o.get("published"))
                c["members"] += 1
                break
            else:
                clusters.append({
                    "label": o["label"], "center": tuple(center[:3]),
                    "extent": tuple(extent[:3]), "bins": self._bins_of(o), "weight": weight,
                    "published": bool(o.get("published")), "members": 1,
                })
        return clusters

    @staticmethod
    def _bins_of(o: dict) -> list:
        """Which sides the ROBOT has stood on.

        `view_bins` is one bin per observation; `angle_bins` is the OR over every voxel's
        azimuth, which spans 71 degrees for a sofa at its standoff and so marks four of twenty
        bins from a single pose. Falling back to it keeps this working against an older mapper,
        at the cost of over-reporting coverage on large objects.
        """
        bins = o.get("view_bins")
        if bins is None:
            bins = o.get("angle_bins") or []
        return [bool(b) for b in bins]

    def _merge_threshold(self, ext_a, ext_b) -> float:
        """The mapper's own rule (object_mapper world-merge test), same two branches."""
        half = [(a / 2 + b / 2) / 2 for a, b in zip_longest(ext_a[:3], ext_b[:3], fillvalue=0.0)]
        scaled = math.sqrt(sum(h * h for h in half)) * self.p["cluster_extent_scale"]
        return max(scaled, self.p["cluster_distance_m"])

    def _match(self, cluster: dict, claimed: set = frozenset()):
        """Existing goal for this physical object, by centroid proximity — never by map id."""
        best, best_d = None, None
        for key, goal in self.goals.items():
            if goal.label != cluster["label"] or key in claimed:
                continue
            d = _dist(cluster["center"], goal.center)
            if d <= self._match_radius(cluster, goal) and (best_d is None or d < best_d):
                best, best_d = key, d
        return best

    def _match_radius(self, cluster: dict, goal: Goal) -> float:
        """How far a goal's centroid may have moved and still be the same object.

        For a PUBLISHED goal, the mapper's own world-merge threshold: its centroid is the
        diversity-weighted mean of regularized voxels and moves in centimetres.

        For an UNPUBLISHED one it is `provisional_centroid()`, the raw voxel mean over
        everything that bled through the mask, and it walks — reproducibly further in one
        frame than the merge threshold reaches, which spawned four goals for one object, each
        demanding four sectors. So the radius becomes the class's own footprint: two
        detections of one label within one object-length of each other, neither of which the
        map can even see, are the same object. Same table as the extent clamp, no new number.
        """
        threshold = self._merge_threshold(cluster["extent"], goal.extent)
        if goal.published and cluster.get("published"):
            return threshold
        cap = self.dimension_priors.for_label(goal.label.lower())
        return max(threshold, 0.5 * math.hypot(cap[0], cap[1]))

    # -- sectors --------------------------------------------------------------

    def _n_sectors(self) -> int:
        return max(int(self.p["n_sectors"]), 1)

    def _build_sectors(self, n_bins: int) -> list[Sector]:
        """Quadrants. Not a tuned threshold — it is what "a distinct viewpoint" means. Four
        90-degree sectors over the mapper's 20 azimuth bins is 5 bins each."""
        n_sectors = self._n_sectors()
        members: list[list[int]] = [[] for _ in range(n_sectors)]
        for k in range(n_bins):
            members[(k * n_sectors) // max(n_bins, 1)].append(k)
        return [Sector(index=i, bins=tuple(m)) for i, m in enumerate(members)]

    def _filled(self, goal: Goal, sector: Sector) -> bool:
        return any(k < len(goal.bins) and goal.bins[k] for k in sector.bins)

    def sector_states(self, goal: Goal) -> list[str]:
        return [BLOCKED if not s.bins else s.state(self._filled(goal, s))
                for s in goal.sectors]

    def _escalate(self, goal: Goal, sector: Sector, reason: str = "unreachable") -> None:
        """Answer a failed attempt by asking from further out — once — then write it off.

        Both ways an attempt can fail come through here: TARE refusing to place a viewpoint,
        and the robot standing there and seeing nothing. They deserve the same response,
        because backing off is the cure for both the "you asked me to stand inside it" refusal
        and the blind-cone non-detection.

        "Refused" at 1.1 m is ambiguous: nothing may be able to see that side, or we may
        simply have asked the robot to stand inside kViewPointCollisionMargin of the object
        itself. Backing off separates the two — without it a closer inspect_radius silently
        CONVERTS coverage into "blocked" and the object reports satisfied having never been
        viewed from that side.

        ONE failure is enough to escalate. Requiring every bin of the sector to fail first —
        which is what this used to do — cost five feedback round trips per radius step, and a
        sector that could never be reached simply stayed open for the whole run.

        The two paths retry differently, correctly. A refusal has marked its bin in `refused`
        (never cleared), so the retry moves in BOTH angle and distance — the next bin outward
        from the sector centre, one step further out; nothing can stand there, so asking the
        identical question again is not a retry. A non-detection has marked nothing, so the
        retry is the same side from further back, which is exactly what the blind cone needs.
        """
        if (self._requested_bin(goal, sector) is None
                or sector.step >= int(self.p["radius_retries"])):
            sector.blocked_reason = reason
            return
        sector.step += 1
        # The retry is a different place, so TARE's verdict on the old one does not carry.
        # Leaving `accepted` set would let the robot's current position count as an arrival at
        # a request nobody has agreed to yet.
        sector.accepted, sector.verdict = False, False

    def _requested_bin(self, goal: Goal, sector: Sector):
        """The bin this sector is currently asking from — its CENTRE, then outward.

        One request per sector, so one verdict changes one sector state — asking for the
        cheapest-to-reach bin instead spent a feedback round trip per bin. The centre is the
        sector's honest direction; the outward walk is the fallback for a sector that is
        blocked at its centre but still viewable from its edge, and it happens across radius
        steps rather than within one, so each retry differs in distance too.
        """
        if not sector.bins:
            return None
        middle = (len(sector.bins) - 1) / 2.0
        order = sorted(range(len(sector.bins)), key=lambda i: (abs(i - middle), i))
        for i in order:
            k = sector.bins[i]
            if k not in sector.refused and not (k < len(goal.bins) and goal.bins[k]):
                return k
        return None

    # -- geometry -------------------------------------------------------------

    def radius(self, goal: Goal, robot_z: float, step: int = 0) -> float:
        """How far from the CENTRE to stand, in metres.

        Reasoned about as a standoff from the object's SURFACE, then half the footprint added
        back, because every constraint below is really about the surface: a 4 m sofa and a
        0.3 m book both want the same clearance from the thing they are looking at. Every term
        is sensor geometry or the mapper's own limits, not a tuned number.

          * standoff  — room for the robot and for TARE to place a viewpoint outside
                        kViewPointCollisionMargin, pushed out one step per refusal so that
                        "nothing can stand there" and "you asked me to stand inside it" stay
                        distinguishable;
          * frame fit — |dz| / tan(60 deg), the horizontal distance at which the object's top
                        AND bottom still fall inside the 120-degree vertical FOV. Closer than
                        this, `bounds_mode: clip` pins the out-of-frame parts to the image edge
                        and their lidar is assigned to whatever mask happens to sit there;
          * range cap — past range_filter.max_distance the mapper assigns no lidar to this
                        object's mask at all, so a viewpoint further out cannot improve it.
                        It caps the SURFACE standoff, which is what the filter measures;
          * inspect   — a floor on the centre distance for small objects. Best-view score is
                        mask_px/frame_px, which grows as 1/d^2, so halving the distance is
                        worth about 4x.
        """
        ext = self._capped_extent(goal)
        half_xy = 0.5 * max(ext[0], ext[1])
        half_z = 0.5 * ext[2]
        dz = max(abs(goal.center[2] + half_z - robot_z), abs(robot_z - (goal.center[2] - half_z)))
        standoff = max(self.p["min_standoff_m"] + step * self.p["radius_retry_step_m"],
                       dz / math.tan(_HALF_VFOV_RAD))
        standoff = min(standoff, self.p["max_range_m"])
        inspect = self.p["inspect_radius_m"] + step * self.p["radius_retry_step_m"]
        return min(max(half_xy + standoff, inspect), half_xy + self.p["max_range_m"])

    def _capped_extent(self, goal: Goal) -> list:
        """The goal's extent, held to the mapper's own upper bound for its class.

        An UNPUBLISHED object's extent is `np.ptp` over every raw voxel — no DBSCAN, no prior
        cap, no outlier trim — so it carries whatever bled through the mask. A `book` inflated
        to 4.5 x 3.8 m asks for a 3.05 m standoff, where a book is about 30 px across a 1920 px
        panorama and no lidar reaches it. Those are precisely the tier-0 objects the whole
        priority scheme exists to rescue, so the geometry must not send them away.

        `dimension_priors.for_label` is the same lookup `regularize_shape` rejects over-large
        clusters with (VLA-3D ground truth, 263 classes, 97.6% of real objects fit, with a
        `default` for the rest), so this adds no new number and cannot disagree with the mapper
        about how big a class can be.
        """
        ext = list(goal.extent[:3]) + [0.0, 0.0, 0.0]
        cap = self.dimension_priors.for_label(goal.label.lower())
        return [min(abs(e), c) for e, c in zip(ext[:3], cap)]

    def _stand_at(self, goal: Goal, bin_index: int, radius: float) -> tuple:
        n = len(goal.bins) or 1
        theta = -math.pi + (bin_index + 0.5) * (2 * math.pi / n)
        return (_quantise(goal.center[0] - radius * math.cos(theta)),
                _quantise(goal.center[1] - radius * math.sin(theta)))

    # -- scheduling -----------------------------------------------------------

    def pending(self) -> list[tuple]:
        """Keys of live goals with at least one sector still open.

        Dormancy is the only thing besides coverage that takes a goal off this list, and it is
        not a budget: it means the MAPPER no longer holds the object. There is still no attempt
        cap and no age cutoff — coverage of a real object is mandatory regardless of travel
        cost.
        """
        return [k for k, g in self.goals.items()
                if not g.dormant and OPEN in self.sector_states(g)]

    def live_goals(self) -> list[Goal]:
        return [g for g in self.goals.values() if not g.dormant]

    def satisfied(self, goal: Goal) -> bool:
        return OPEN not in self.sector_states(goal)

    def coverage_complete(self) -> bool:
        """Every label accounted for and every goal closed out.

        Requires a goal per LABEL, not merely that the goals we have are done: with one label
        still undiscovered there is exploring left to do, and reporting complete would end the
        run before the robot ever looked for it.
        """
        if not self.targets or not self.live_goals():
            return False
        if self.found_labels() != self.targets:
            return False
        return not self.pending()

    def found_labels(self) -> set:
        return {g.label.lower() for g in self.live_goals()}

    def preempt(self, now: float) -> bool:
        """May targets take priority over frontier exploration right now? Latched.

        The raw condition below is frame-volatile in both directions: one newly discovered far
        goal turns it off, one covered sector turns it back on. Published raw it flipped 4-8
        times per question over a 13-scene sweep (194 transitions in 50 questions, japanese_room
        7.8/question), and every flip switches TARE's global tour between priority-only and
        stock (`grid_world.cpp` `priority_only = target_preempt_ && ...`), so the tour is
        re-solved against a different cell set every second or two.

        So: ON immediately when the condition holds — that is a real signal and delaying it
        costs exploration time — and OFF only once the condition has been continuously false
        for `coverage_hold_s`, the same "outlast one perception + planning round trip" duration
        `_publish_coverage_done` uses and for the same reason. Asymmetric on purpose; a
        symmetric debounce would delay the useful edge to suppress the useless one.
        """
        raw = self._preempt_raw()
        if raw:
            self._preempt_false_since = None
            self._preempt = True
            return True
        if self._preempt_false_since is None:
            self._preempt_false_since = now
        if (now - self._preempt_false_since) >= float(self.p["coverage_hold_s"]):
            self._preempt = False
        return self._preempt

    def _preempt_raw(self) -> bool:
        """The unlatched condition. See `preempt` for why nothing publishes this directly.

        Preempt buys three things from TARE: the global tour is offered only the subspaces
        holding a target, the exploration-finished latch is blocked, and the lookahead is
        pinned at our first accepted request. The middle and last are what actually drive the
        robot at a target; the FIRST is the one that can starve discovery, because a label
        with no instance can be found by exactly one mechanism and that mechanism is the
        frontier tour.

        Two disjuncts, and each closes one of the two failure modes we have already had:

          * every label found — nothing left to discover, so restricting the tour costs
            nothing. This is the rule the yaml documents.
          * every pending goal in-horizon — the restriction is INERT here, and provably so:
            `priority_cell_indices_` is populated only from the `far` branch of
            UpdateTargetViewPoints, so with nothing far it is empty, `priority_only` is false,
            and SolveGlobalTSP runs stock. We take the lookahead pin and leave the frontier
            tour alone.

        Without the second, one unfindable anchor label — "the pillow closest to the book on
        the *stool*" in a scene whose stool SAM calls a chair — switches the whole mechanism
        down to opportunistic must-visit nodes for the entire run.
        """
        pending = self.pending()
        if not self.targets or not pending:
            return False
        if self.found_labels() == self.targets:
            return True
        return all(self.goals[k].in_horizon for k in pending)

    def rank(self, keys: list, robot, prefer_committed: bool = True, now: float = 0.0) -> list:
        """Answerability first, then commitment, then distance.

        Tier is the whole point: an unpublished object is ABSENT from obj_map.json, so its
        first sector is worth more than a published object's fourth however much nearer that
        one is. Distance orders the route inside a tier; it never decides whether a goal is
        worked, so a stuck goal only ever drifts back — it is never dropped.

        The cooldown sits between them and is deliberately NOT a tally. Ranking on a
        deferral COUNT means a twice-deferred goal loses to a once-deferred one on the far
        side of the room forever, so route order decays into round-robin as a run goes on. A
        goal released by the deadlock guard is held back for one dwell — long enough that it
        cannot immediately re-elect itself — and after that competes on distance like the rest.

        `prefer_committed` must be False when CHOOSING the next goal. With it True the current
        goal sorts first by construction and would be re-elected every time.
        """
        cooldown = float(self.p["goal_dwell_s"])

        def key_fn(key):
            goal = self.goals[key]
            cooling = (goal.deferred_at is not None
                       and (now - goal.deferred_at) < cooldown)
            return (goal.tier,
                    prefer_committed and key != self.committed,
                    cooling,
                    _dist(goal.center, robot) if robot else 0.0)
        return sorted(keys, key=key_fn)

    def _signature(self, key) -> tuple:
        """What counts as progress on a goal: sectors it has actually COVERED.

        Twice now this has been too generous and left the deadlock guard unfireable.

        First it counted filled bins, which grow whenever the robot passes anywhere near the
        object — including on the way to a different goal.

        Then it counted the whole sector-state tuple, which is worse than it looks: a sector
        going open -> BLOCKED is a state change, so a goal that was only failing renewed its
        own hold on every failure. Measured on a live run, one `sofa` churned
        open/covered/open/open -> blocked/covered/open/open -> blocked/covered/open/blocked
        and stayed committed for 84 s with nine other goals pending, because every refusal
        looked like progress.

        Giving up on a sector is attrition, not progress. Only coverage renews the hold.
        """
        goal = self.goals.get(key)
        return () if goal is None else (self.sector_states(goal).count(COVERED),)

    def _recommit(self, now: float, robot=None) -> None:
        """Hold one object until it is done, blocked, or demonstrably going nowhere.

        The unit of work is "walk to an object and orbit it". Rotating on a fixed timer
        reshuffled the target set before any of that could finish and the robot was yanked
        between goals covering none, so the only two exits are: the goal leaves `pending`, or
        it produces no sector-state change for goal_dwell_s. That second one is a DEADLOCK
        GUARD, not a rotation timer — it must exceed one traverse of the 9 m local horizon
        plus an orbit, and it fires only on a goal making no progress at all.
        """
        pending = self.pending()
        if not pending:
            self.committed, self._committed_signature = None, None
            return

        signature = self._signature(self.committed)
        if self.committed in pending and signature != self._committed_signature:
            self._committed_signature = signature
            self.committed_since = now          # progress renews the hold

        expired = (now - self.committed_since) > self.p["goal_dwell_s"]
        # A goal that just turned up in a strictly better tier takes over immediately. It is
        # the one preemption the dwell must not block: tier 0 means the map cannot answer with
        # that object at all, so polishing a published object's box while it sits there is
        # spending the window on the wrong thing. It also keeps `committed` honest — the
        # emitted array is ordered by rank(), so anything that outranks the committed goal
        # would otherwise take index 0 and TARE would drive at a goal we did not commit to.
        outranked = any(self.goals[k].tier < self.goals[self.committed].tier for k in pending) \
            if self.committed in self.goals else False

        if self.committed in pending and not expired and not outranked:
            return
        if self.committed in pending and expired:
            # A dwell spent without closing a sector. Sorts this goal behind its tier-mates so
            # every target gets worked; it stays pending regardless.
            self.goals[self.committed].deferrals += 1
            self.goals[self.committed].deferred_at = now

        self.committed = self.rank(pending, robot, prefer_committed=False, now=now)[0]
        self.committed_since = now
        self._committed_signature = self._signature(self.committed)

    # -- output ---------------------------------------------------------------

    def requests(self, robot, now: float) -> list[tuple]:
        """Where to stand, best first, as (x, y, goal_key, sector_index, bin_index).

        The committed goal's open sectors lead the list because TARE drives at the first
        ACCEPTED pose and holds it; everything after that is a must-visit node in its local
        TSP, which routes them by path length. One pose per sector — bins inside an already
        covered sector add no new viewpoint, and spending the in-flight budget on them is what
        made coverage look complete while every view came from the same side.
        """
        if robot is None:
            return []
        self._recommit(now, robot)
        pending = self.pending()
        if not pending:
            self.last_request = []
            return []

        budget = int(self.p["max_viewpoints"])
        out: list[tuple] = []
        ordered = self.rank(pending, robot, now=now)
        for position, key in enumerate(ordered):
            poses = self._goal_requests(key, robot)
            # The committed goal gets its whole orbit; every other goal contributes its single
            # cheapest side. Filling the array with one object's four sectors AND another's
            # would spend TARE's eight-viewpoint budget before the third object is mentioned,
            # and each must-visit viewpoint marks its covered points covered before the
            # coverage queues are scored — so a long array actively suppresses exploration.
            for pose in (poses if position == 0 else poses[:1]):
                if len(out) >= budget:
                    break
                out.append(pose)
            if len(out) >= budget:
                break

        self.last_request = out
        self.emitted.append(list(out))
        return out

    def _goal_requests(self, key, robot) -> list[tuple]:
        """One request per OPEN sector of this goal, cheapest first."""
        goal = self.goals[key]
        robot_z = robot[2] if len(robot) > 2 else goal.center[2]
        scored = []
        for sector, state in zip(goal.sectors, self.sector_states(goal)):
            if state != OPEN:
                continue
            bin_index = self._requested_bin(goal, sector)
            if bin_index is None:
                continue
            x, y = self._stand_at(goal, bin_index, self.radius(goal, robot_z, sector.step))
            scored.append((math.hypot(x - robot[0], y - robot[1]),
                           x, y, key, sector.index, bin_index))
        scored.sort(key=lambda c: c[0])
        return [entry[1:] for entry in scored]

    def _attribute(self, point):
        """Which (goal, sector, bin) asked for this position.

        Newest array first: the same sector's request moves as its radius escalates, so the
        newest match is the one this verdict is about.
        """
        if len(point) < 2:
            return None
        px, py = float(point[0]), float(point[1])
        for batch in reversed(self.emitted):
            for x, y, key, sector_index, bin_index in batch:
                if math.hypot(x - px, y - py) <= _ECHO_EPS_M:
                    return key, sector_index, bin_index
        return None

    # -- reporting ------------------------------------------------------------

    def status(self, now: float) -> dict:
        return {
            "targets": sorted(self.targets),
            "found": sorted(self.found_labels()),
            "unseen": sorted(self.targets - self.found_labels()),
            "committed": None if self.committed is None else self.committed[0],
            "preempt": self.preempt(now),
            "coverage_complete": self.coverage_complete(),
            "requested_viewpoints": len(self.last_request),
            "goals": [self._goal_status(g, now) for g in self.goals.values()],
            "goals_dormant": sum(1 for g in self.goals.values() if g.dormant),
        }

    def _goal_status(self, goal: Goal, now: float) -> dict:
        states = self.sector_states(goal)
        return {
            "label": goal.label,
            "center": [round(v, 2) for v in goal.center],
            "extent": [round(v, 2) for v in goal.extent],
            # What the standoff was actually computed from. A large gap between the two is the
            # mapper bleeding into an unpublished object, not a big object.
            "extent_capped": [round(v, 2) for v in self._capped_extent(goal)],
            "tier": goal.tier,
            "published": goal.published,
            "cluster_size": goal.members,
            "sectors": states,
            "sectors_covered": states.count(COVERED),
            "sectors_blocked": states.count(BLOCKED),
            "view_bins": int(sum(1 for b in goal.bins if b)),
            "arrived": [s.arrived for s in goal.sectors],
            "accepted": [s.accepted for s in goal.sectors],
            "in_horizon": goal.in_horizon,
            "dormant": goal.dormant,
            "blocked_reason": [self._blocked_reason(s, st)
                               for s, st in zip(goal.sectors, states)],
            "deferrals": goal.deferrals,
            "satisfied": OPEN not in states,
            "age_s": round(now - goal.first_seen, 1),
        }

    @staticmethod
    def _blocked_reason(sector: Sector, state: str) -> str:
        if state != BLOCKED:
            return ""
        return sector.blocked_reason or "no-bins"

    def summary(self) -> dict:
        """The one-line verdict a report row carries: did coverage finish before exploring did."""
        goals = self.live_goals()
        states = [self.sector_states(g) for g in goals]
        return {
            "targets": sorted(self.targets),
            "labels_found": sorted(self.found_labels()),
            "labels_unseen": sorted(self.targets - self.found_labels()),
            "goals": len(goals),
            # Goals the mapper stopped holding: pruned false positives, and the stale copies a
            # walking provisional centroid leaves behind. A large number here means the object
            # list is churning, not that the robot failed at anything.
            "goals_dormant": sum(1 for g in self.goals.values() if g.dormant),
            "goals_satisfied": sum(1 for s in states if OPEN not in s),
            "goals_outstanding": sum(1 for s in states if OPEN in s),
            "goals_unpublished": sum(1 for g in goals if not g.published),
            "sectors_covered": sum(s.count(COVERED) for s in states),
            "sectors_blocked": sum(s.count(BLOCKED) for s in states),
            "sectors_open": sum(s.count(OPEN) for s in states),
            "coverage_complete": self.coverage_complete(),
        }

    def close_out(self) -> str:
        """The log line that says whether target coverage finished before exploration did."""
        if not self.live_goals():
            return "targets at close: none tracked"
        parts = []
        for goal in self.live_goals():
            states = self.sector_states(goal)
            blocked = f" ({states.count(BLOCKED)} blocked)" if BLOCKED in states else ""
            parts.append(f"{goal.label} {states.count(COVERED)}/{len(states)} sectors{blocked} "
                         f"{'OUTSTANDING' if OPEN in states else 'satisfied'}"
                         f"{'' if goal.published else ' UNPUBLISHED'}")
        unseen = sorted(self.targets - self.found_labels())
        tail = f" | never found: {unseen}" if unseen else ""
        return "targets at close: " + " | ".join(parts) + tail


def default_params() -> dict:
    """Every knob, with the reason it has the value it has.

    Sourced from the mapper's own configuration wherever one exists, so the planner clusters
    and ranges on the rules the mapper would have used had it been able to.
    """
    config = MappingConfig()
    merge = config.world_merge
    return {
        # Fragment clustering. The mapper cannot do this itself: world merge needs a non-None
        # centroid on BOTH sides, and an under-observed object has none — which is exactly the
        # object we are chasing.
        "cluster_distance_m": merge.absolute_distance,
        "cluster_extent_scale": merge.extent_scale,
        # Admission. Below these an unpublished entry is noise, and chasing noise costs the
        # whole window.
        "min_life": 5,                  # world merge only considers 5 < life < 1000
        "min_info_frames": 3,           # seen across frames, not one blob in one frame
        "min_voxels": 30,
        #: Mapper updates the robot must stand at an ACCEPTED request without the sector
        #: filling before we conclude the object does not register from that side. A timing
        #: question, kept separate from min_info_frames (an admission question) even though
        #: both happen to be 3 -- one is about how substantial an object is, the other about
        #: how long to wait for it.
        "arrival_patience_updates": 3,
        #: Consecutive updates with no matching cluster before a goal goes dormant.
        #: describe_objects reports every tracked object every frame, so a short run of
        #: absences is already strong evidence; this only rides out a frame the mapper
        #: dropped for its own reasons.
        "goal_absence_limit": 5,
        "n_sectors": 4,
        "inspect_radius_m": 1.0,
        "min_standoff_m": 0.8,
        #: Escalation steps a sector gets before it is written off. 2, not 1, because TARE now
        #: answers `unreachable` for any square inside the lattice that is not a candidate --
        #: which is where two of a wall-mounted object's four sectors land every cycle. At 1
        #: that verdict blocked the sector after a single retry: measured, 412 of 446 goals
        #: closed with blocked sectors and only 51% of sectors were ever covered.
        #:
        #: This buys RECALL, not box quality. Measured on the same sweep, mean IoU by sectors
        #: covered was 1 -> 0.24, 2 -> 0.34, 3 -> 0.37, 4 -> 0.31: circling an object more does
        #: not tighten its box. What it does is get more objects into the map at all, which is
        #: where 18% (label absent) + 14% (label present, wrong instance) of questions are
        #: lost. Do not re-test this against IoU and conclude it did nothing.
        "radius_retries": 2,
        "radius_retry_step_m": 1.0,
        #: Beyond this the mapper assigns no lidar to a mask at all, so a viewpoint further
        #: out cannot improve the object it was requested for.
        "max_range_m": config.range_filter.max_distance,
        #: TARE's own kMaxTargetViewPointNum and kTargetViewPointSnapMaxDist. These are the
        #: values the C++ falls back to when a scenario omits them; target_explorer overwrites
        #: both from tare_planner's scenario yaml at startup, so the yaml stays the single
        #: source and this file cannot silently disagree with the planner it is talking to.
        #: Poses past the Nth ACCEPTED one get no verdict and no priority cell from TARE, so
        #: asking for more than it will take loses information rather than being refused.
        "max_viewpoints": TARE_DEFAULTS["max_target_viewpoints"],
        "max_viewpoints_cap": TARE_DEFAULTS["max_target_viewpoints"],
        "snap_max_dist_m": TARE_DEFAULTS["snap_max_dist_m"],
        #: How close the robot has to come before a request counts as visited: TARE's snap
        #: radius plus the base planner's waypointXYRadius, which together bound how far from
        #: the asked-for spot the robot can legitimately come to rest. Derived, not chosen --
        #: recomputed by the node whenever the yaml moves the snap radius.
        "arrival_radius_m": TARE_DEFAULTS["snap_max_dist_m"] + WAYPOINT_XY_RADIUS_M,
        "goal_dwell_s": 30.0,
        #: Decoupled from /exploration/object_targets on purpose: map_node skips publishing
        #: entirely on a zero-detection frame, and SAM is prompted only with the target labels
        #: — so the input goes quiet exactly while the robot is travelling round to the far
        #: side of an object, which is when the goal must stay alive. Without this the request
        #: expires against TARE's kTargetViewPointTimeout mid-orbit, every time.
        "publish_hz": 2.0,
        #: Coverage-complete must outlast one full perception + planning round trip (map_node
        #: ~2.2 Hz, TARE feedback ~1 Hz) so a single flickering frame cannot end exploration.
        "coverage_hold_s": 3.0,
    }


def validate_params(params: dict) -> None:
    """Relations between knobs that are load-bearing rather than incidental.

    Raising here is right: every one of these is silent when violated — the run looks normal
    and the coverage logic is simply wrong — and they are all set at construction, so a bad
    combination cannot appear halfway through a mission.
    """
    if params["radius_retry_step_m"] <= params["arrival_radius_m"]:
        raise ValueError(
            "radius_retry_step_m must exceed arrival_radius_m "
            f"({params['radius_retry_step_m']} <= {params['arrival_radius_m']}): an escalated "
            "request would land within arrival range of where the robot already is, so the "
            "retry would be consumed as an arrival without the robot moving at all")
    cap = params.get("max_viewpoints_cap", TARE_DEFAULTS["max_target_viewpoints"])
    if params["max_viewpoints"] > cap:
        raise ValueError(
            f"max_viewpoints must not exceed kMaxTargetViewPointNum "
            f"({params['max_viewpoints']} > {cap}): TARE breaks out of its loop after that "
            "many ACCEPTED poses, so the tail gets no verdict and no priority cell — the "
            "request is not refused, it is unheard")
    if params["min_standoff_m"] <= 0 or params["inspect_radius_m"] <= 0:
        raise ValueError("standoff and inspect radius must be positive")


def _dist(a, b) -> float:
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(a[:3], b[:3])))


def _quantise(value: float) -> float:
    return round(round(value / _POSITION_QUANTUM_M) * _POSITION_QUANTUM_M, 6)


def _round(center) -> tuple:
    return tuple(round(float(v), 2) for v in center[:3])
