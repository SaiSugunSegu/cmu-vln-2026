"""The floor the robot has seen, as a grid, built up over the whole run.

The challenge withdrew the ground-truth traversable area this year (README, *System Outputs*),
so terrain analysis is the only reading of the floor we are allowed at test time. This
accumulates `/terrain_map_ext` into a persistent occupancy grid and answers one question:
**where is the nearest place the robot could stand?**

Why a grid rather than an accumulated point cloud. The cloud version merged with `vstack` and
de-duplicated with `np.unique`, which costs O(everything seen so far) on every message --
measured at 37 ms with 50k points accumulated, 101 ms at 165k and 241 ms at 400k, growing for
the length of the run. Scattering into a fixed grid is O(points in *this* message): 0.11 ms for
10k, 0.86 ms for 60k, constant however long the run goes on, in 0.16 MB for a 40 m square at
10 cm. That is the difference between throttling terrain to every 0.25 m of travel and simply
reading it.

Cells hold ACCUMULATED EVIDENCE, not the last reading. That distinction is the whole reason
this class is more than a scatter: /terrain_map_ext runs `decayTime = 4.0` with
`noDecayDis = 0.0`, so it is a four-second rolling window, and `obstacleHeightThre` is a hard
cut on a noisy height estimate. Under last-write-wins a single grazing return stamped a floor
cell OBSTACLE and it stayed that way until some later cloud happened to disagree. Measured
between two consecutive question snapshots of one hotel_room_1 run, **1020 cells changed state
and 464 of them went FREE -> OBSTACLE**, roughly a tenth of the known floor flickering per
question. `nearest_free` reads the grid at one instant, so that flicker is not cosmetic: on
Q04 it snapped a waypoint 2.83 m because the floor beside the target was OBSTACLE at the
moment all six of the leg's snaps ran, while a snapshot eleven seconds later had free floor
0.18 m away. Voting fixes it at the source -- a cell seen free forty times and blocked three
is floor.

`nearest_free` is likewise a lookup rather than a search. One `distance_transform_edt` pass
answers "which free cell is nearest" for *every* cell at once (9.6 ms at 10 cm), so each query
after it is an array index -- against 104 ms for the pairwise scan this replaced.

Pure numpy and scipy: no ROS, no cv2, so it runs under `pytest` on the host.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy import ndimage

__all__ = ["TraversableArea"]


class TraversableArea:
    """Cells the terrain analysis has called floor, and where the nearest one is.

    Three states, and the third one matters: a cell nobody has looked at is `UNKNOWN`, not
    free. Aiming the robot at ground that was never observed is how a waypoint ends up inside
    furniture the map simply has no reading for.
    """

    UNKNOWN, FREE, OBSTACLE = 0, 1, 2

    #: How much evidence a cell may bank, in units of one observation. A cell at the clamp
    #: needs this many contrary readings in a row to change its mind, which at terrain's ~5 Hz
    #: is a couple of seconds of consistent disagreement -- long enough to outlast the single
    #: -message flicker that motivated the counter, short enough that genuinely wrong ground
    #: still corrects itself. Not a probability: the readings are far from independent and
    #: calling this a log-odds would dress up a vote as inference.
    CLAMP = 12

    def __init__(self, cell_m: float = 0.10, half_span_m: float = 20.0,
                 origin: Sequence[float] = (0.0, 0.0), clamp: int = CLAMP) -> None:
        if cell_m <= 0.0:
            raise ValueError(f"cell_m must be positive, got {cell_m}")
        if half_span_m <= 0.0:
            raise ValueError(f"half_span_m must be positive, got {half_span_m}")
        self.cell_m = float(cell_m)
        # Lower-left corner in world metres. Kept as the anchor rather than the centre so that
        # growing the grid is a corner move and an offset copy, with no re-indexing.
        self._corner = np.array([float(origin[0]) - half_span_m,
                                 float(origin[1]) - half_span_m], dtype=float)
        if clamp < 1:
            raise ValueError(f"clamp must be at least 1, got {clamp}")
        self.clamp = int(clamp)
        width = max(1, int(round(2.0 * half_span_m / self.cell_m)))
        #: The 3-state view everything else reads. Derived from `_score`, kept materialised
        #: because `nearest_free` and the overlay want a plain array.
        self.state = np.zeros((width, width), dtype=np.uint8)
        #: Signed evidence per cell: positive is floor, negative is obstacle, zero is either
        #: never seen or exactly contradicted. int16 so a raised clamp cannot silently wrap.
        self._score = np.zeros((width, width), dtype=np.int16)

        # `nearest_free` is answered from a cached transform, rebuilt only when the grid has
        # changed or the clearance asked for differs from the cached one.
        self._nearest: Optional[np.ndarray] = None
        self._free_mask: Optional[np.ndarray] = None
        self._cached_clearance = -1.0
        self._dirty = True

    # -- writing ------------------------------------------------------------

    def add(self, xy: np.ndarray, obstacle: np.ndarray,
            origin: Optional[Sequence[float]] = None,
            max_range_m: float = 0.0, weight: int = 1) -> None:
        """Fold one terrain reading in as a VOTE, not as the new truth.

        Each cell this cloud covers moves `weight` toward floor or toward obstacle and is
        clamped, so a cell's state is what it has mostly looked like rather than what the last
        message said. That is the difference between a map and a snapshot: terrain analysis
        decays everything after four seconds and thresholds a noisy height at a hard 0.05 m, so
        under overwrite semantics one grazing return erased floor the robot had already seen
        forty times -- and `nearest_free`, reading the grid at a single instant, then walked
        metres away to find ground. See the module docstring for the measurement.

        A cell is counted ONCE per message however many points land in it. Terrain point
        density falls off with range, so per-point voting would let a patch of floor three
        metres away be outvoted by the same patch of floor at half a metre -- which is a
        statement about the lidar, not about the floor.

        `weight` is how much this source is trusted per message. /terrain_map is worth more
        than /terrain_map_ext over the ground they share: half the voxel size, and
        `noDecayDis = 1.75` means its near field is not a rolling window at all.

        `origin` + `max_range_m` discard readings too far from the sensor to be real.
        /terrain_map_ext is 20 m wide BY CONSTRUCTION, so a point beyond that is not distant
        floor, it is bad data -- and bad floor is worse than no floor, because `nearest_free`
        will happily snap a waypoint onto it. Measured: livingroom_4 accumulated 198,280 free
        cells (1,983 m2) against a room whose real traversable area is 24.7 m2, and loft
        158,370, both from runs where the robot ended up outside the building. Off when
        `max_range_m` is 0, so callers that have no pose still work.
        """
        xy = np.asarray(xy, dtype=float)
        if xy.size == 0:
            return
        xy = xy.reshape(len(xy), -1)[:, :2]
        obstacle = np.asarray(obstacle, dtype=bool).reshape(-1)
        if len(obstacle) != len(xy):
            raise ValueError(f"{len(xy)} points but {len(obstacle)} obstacle flags")
        weight = max(1, int(weight))

        if origin is not None and max_range_m > 0.0:
            near = np.linalg.norm(xy - np.asarray(origin, dtype=float)[:2], axis=1) <= max_range_m
            if not near.any():
                return
            xy, obstacle = xy[near], obstacle[near]

        self._grow_to_fit(xy)
        ix, iy, inside = self._index(xy)
        if not inside.any():
            return
        ix, iy, obstacle = ix[inside], iy[inside], obstacle[inside]

        # Collapse this message to one vote per cell. Two boolean grids rather than a unique()
        # over the indices: allocation is 0.16 MB and the scatter is a single pass, against a
        # sort over every point in the cloud.
        blocked = np.zeros(self.state.shape, dtype=bool)
        clear = np.zeros(self.state.shape, dtype=bool)
        blocked[ix[obstacle], iy[obstacle]] = True
        clear[ix[~obstacle], iy[~obstacle]] = True
        # A cell holding both in one message has something standing on it. Obstacle wins the
        # tie: the free points are the floor AROUND the thing, sharing a 10 cm cell with it.
        clear &= ~blocked

        self._score[clear] = np.minimum(self._score[clear] + weight, self.clamp)
        self._score[blocked] = np.maximum(self._score[blocked] - weight, -self.clamp)

        # Only the cells this message touched can have changed state, so the derived view is
        # refreshed over those and nowhere else. Ties (score back to zero) read as OBSTACLE:
        # the cell HAS been observed, so it is not UNKNOWN, and blocked is the safe reading of
        # evidence that cancels out.
        seen = clear | blocked
        self.state[seen] = np.where(self._score[seen] > 0, self.FREE, self.OBSTACLE)
        self._dirty = True

    # -- reading ------------------------------------------------------------

    def nearest_free(self, xy: Sequence[float],
                     clearance_m: float = 0.0) -> Optional[tuple[float, float]]:
        """The centre of the free cell nearest `xy`, or None if there is no answer.

        Nearest to the POINT, not to the cell containing it. The transform below works in cell
        indices, so on its own it answers for the query's cell centre and can be up to a cell
        diagonal off -- measured 0.22 m where the true nearest was 0.18 m. That is small, but
        the whole job of this method is to give up as little distance to the model's waypoint
        as possible, so the transform's answer is used as a bound and the free cells inside
        that bound are then compared exactly.

        None means one of: nothing has been observed as free yet, `xy` lies outside the grid, or
        `clearance_m` was set high enough to erode every free cell away. The caller keeps the
        original waypoint in all three cases.
        """
        query = np.asarray(xy, dtype=float).reshape(1, 2)
        ix, iy, inside = self._index(query)
        if not inside[0]:
            return None

        nearest = self._transform(max(0.0, float(clearance_m)))
        if nearest is None:
            return None
        fx = int(nearest[0, ix[0], iy[0]])
        fy = int(nearest[1, ix[0], iy[0]])
        return self._refine(query[0], fx, fy)

    def _refine(self, point: np.ndarray, fx: int, fy: int) -> tuple[float, float]:
        """The free cell truly nearest `point`, given the transform's cell-wise answer.

        The transform's cell can be wrong by at most one cell diagonal, so every better
        candidate lies within that bound of it -- a window of a few cells for a close snap.
        Searched exactly rather than trusted, because being 4 cm closer to the model's waypoint
        costs one array slice.
        """
        def centre(cx: int, cy: int) -> np.ndarray:
            return self._corner + (np.array([cx, cy], dtype=float) + 0.5) * self.cell_m

        best = centre(fx, fy)
        bound = float(np.hypot(*(best - point))) + self.cell_m * np.sqrt(2.0)
        span = int(np.ceil(bound / self.cell_m)) + 1
        qx, qy, _ = self._index(point.reshape(1, 2))
        x0, x1 = max(0, qx[0] - span), min(self.state.shape[0], qx[0] + span + 1)
        y0, y1 = max(0, qy[0] - span), min(self.state.shape[1], qy[0] + span + 1)

        free = self._free_mask
        window = np.argwhere(free[x0:x1, y0:y1]) if free is not None else np.empty((0, 2), int)
        if not len(window):
            return (float(best[0]), float(best[1]))
        window += (x0, y0)
        centres = self._corner + (window + 0.5) * self.cell_m
        return tuple(float(v) for v in centres[np.linalg.norm(centres - point, axis=1).argmin()])

    def counts(self) -> dict:
        """Cell tallies, for the leg record and for judging whether a run saw enough floor.

        `bounds` is the world box the OBSERVED cells occupy, not the grid's own extent: a grid
        that grew to 40 m because of one stray reading looks identical to a good one by shape
        alone, and this is what lets a run be checked against the room it claims to be in.
        """
        free = int(np.count_nonzero(self.state == self.FREE))
        obstacle = int(np.count_nonzero(self.state == self.OBSTACLE))
        seen = np.argwhere(self.state != self.UNKNOWN)
        bounds = None
        if seen.size:
            lo = self._corner + seen.min(axis=0) * self.cell_m
            hi = self._corner + (seen.max(axis=0) + 1) * self.cell_m
            bounds = [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]
        return {"free": free, "obstacle": obstacle, "cell_m": self.cell_m,
                "shape": list(self.state.shape), "bounds": bounds}

    def snapshot(self) -> dict:
        """Everything needed to redraw this grid somewhere else.

        Written per question so the floor that decided every snap can be looked at afterwards.
        Only `free_cells` used to survive a run, and a single number cannot show that the
        floor beside the target was never observed -- which is the failure it kept hiding.

        `score` goes with it: a cell at +1 and a cell at +12 both draw as floor, and telling
        them apart is how "the grid is thin here" is distinguished from "the grid is wrong
        here".
        """
        return {"state": self.state, "score": self._score, "cell_m": float(self.cell_m),
                "corner": self._corner.astype(float)}

    # -- internals ----------------------------------------------------------

    def _index(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Grid indices for world points, plus which of them actually land on the grid."""
        cells = np.floor((xy - self._corner) / self.cell_m).astype(np.int64)
        ix, iy = cells[:, 0], cells[:, 1]
        inside = ((ix >= 0) & (ix < self.state.shape[0])
                  & (iy >= 0) & (iy < self.state.shape[1]))
        return np.clip(ix, 0, self.state.shape[0] - 1), \
            np.clip(iy, 0, self.state.shape[1] - 1), inside

    def _grow_to_fit(self, xy: np.ndarray) -> None:
        """Enlarge the grid when a reading lands outside it. Rare, so cost does not matter."""
        low = np.minimum(self._corner, xy.min(axis=0))
        high = np.maximum(self._corner + np.array(self.state.shape) * self.cell_m,
                          xy.max(axis=0) + self.cell_m)
        if np.array_equal(low, self._corner) and \
                tuple(np.round((high - low) / self.cell_m).astype(int)) == self.state.shape:
            return

        # Pad by a margin so a robot walking steadily outward does not reallocate every message.
        margin = 5.0
        low -= margin
        high += margin
        corner = np.floor(low / self.cell_m) * self.cell_m
        shape = tuple(int(np.ceil((high[i] - corner[i]) / self.cell_m)) for i in (0, 1))
        off = np.round((self._corner - corner) / self.cell_m).astype(int)
        sx, sy = self.state.shape
        grown = np.zeros(shape, dtype=np.uint8)
        grown[off[0]:off[0] + sx, off[1]:off[1] + sy] = self.state
        grown_score = np.zeros(shape, dtype=np.int16)
        grown_score[off[0]:off[0] + sx, off[1]:off[1] + sy] = self._score
        self.state, self._score, self._corner = grown, grown_score, corner
        # The cached transform indexes the OLD shape, so it is not merely stale, it is unsafe.
        self._nearest = None
        self._dirty = True

    def _transform(self, clearance_m: float) -> Optional[np.ndarray]:
        """Indices of the nearest free cell, for every cell. Rebuilt only when it must be."""
        if not self._dirty and clearance_m == self._cached_clearance:
            return self._nearest

        free = self.state == self.FREE
        if clearance_m > 0.0:
            # Erode by measuring how far each cell is from the nearest obstacle. One grid pass,
            # rather than a distance test per candidate. Off by default: the clearance the base
            # autonomy applies to its own aiming is not something we need to reproduce here.
            obstacle = self.state == self.OBSTACLE
            if obstacle.any():
                gap = ndimage.distance_transform_edt(~obstacle) * self.cell_m
                free &= gap >= clearance_m

        self._cached_clearance = clearance_m
        self._dirty = False
        # Kept so `_refine` compares against the SAME cells the transform did -- rebuilding it
        # there would silently ignore `clearance_m`.
        self._free_mask = free
        if not free.any():
            self._nearest = None
            return None

        # `distance_transform_edt` measures to the nearest ZERO, so the free cells must be the
        # zeros; the returned indices are then the nearest free cell for every cell on the grid.
        self._nearest = ndimage.distance_transform_edt(
            ~free, return_distances=False, return_indices=True)
        return self._nearest
