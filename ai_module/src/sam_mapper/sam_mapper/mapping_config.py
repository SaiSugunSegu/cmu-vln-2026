"""Every tunable in the 2D->3D mapper, loadable from yaml. Stage letters: docs/map_node_pipeline.md.

Defaults here are the source of truth and match the last 13-scene bench run; the yaml only
carries overrides. `just map3d-replay` reads the same file the node does, so an A/B is a
config edit rather than a code change.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields

#: Per-class size caps, generated from VLA-3D ground truth by
#: scripts/eval/build_dimension_priors.py and shipped inside the package (setup.py
#: package_data) because it is needed at RUNTIME — data/ is a gitignored host bind-mount
#: that will not exist on the challenge machine.
_PRIORS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "dimension_priors.json")


def _load_priors() -> tuple[tuple, dict]:
    with open(_PRIORS_JSON) as handle:
        payload = json.load(handle)
    return (tuple(payload["default"]),
            {k: tuple(v) for k, v in payload["priors"].items()})


DEFAULT_PRIOR, DERIVED_PRIORS = _load_priors()


def _sub(cls, value):
    """Build a nested config from a dict, tolerating None/missing."""
    if isinstance(value, cls):
        return value
    return cls.from_dict(value or {})


class _FromDict:
    @classmethod
    def from_dict(cls, data: dict):
        if data is None:
            data = {}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            # Loud, not silent: a typo'd key in a tuning yaml would otherwise read as "that
            # knob did nothing", which is indistinguishable from a real negative result.
            raise KeyError(f"{cls.__name__}: unknown config key(s) {sorted(unknown)}; "
                           f"known keys are {sorted(known)}")
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ErosionConfig(_FromDict):
    """B2 - shrink masks away from the silhouette.

    OFF: measured a net win (recall@0.50 0.138 -> 0.186). It cost centroid accuracy, which
    B7 now recovers by filtering on depth instead of geometry.
    """
    enabled: bool = False
    fraction: float = 0.10          # iterations = clip(round(fraction * mask short side), min, max)
    min_iterations: int = 1
    max_iterations: int = 5


@dataclass
class RangeFilterConfig(_FromDict):
    """B3 - drop points far from the robot, before the projection.

    5.0 m is the peak of a measured 3/5/6 m curve. Shortening trades real distant
    observations for less bleed: 3 m gave the best quality numbers we have measured and the
    worst coverage (recall -0.081, no-candidate doubled).

    No longer a lever. Since B7 removes bleed directly, 5 -> 6 m costs 0.009 cat2 where it
    used to cost 0.025. Retune only if B7 changes.
    """
    enabled: bool = True
    max_distance: float = 5.0


@dataclass
class VoxelDownsampleConfig(_FromDict):
    """B9 - voxel_down_sample, on creation only (merges use the raw cloud).

    `voxel_size` also sets the accumulation grid, DBSCAN eps (3x), the extent padding and
    the effective prune threshold — changing it is not a single-variable experiment.
    """
    enabled: bool = True
    voxel_size: float = 0.05


@dataclass
class OutlierRemovalConfig(_FromDict):
    """B10 - statistical outlier removal, on creation only.

    nb_neighbors=20 cannot function on a 3-point detection, so it is near a no-op for the
    small objects that need help most.
    """
    enabled: bool = True
    nb_neighbors: int = 20
    std_ratio: float = 1.0


@dataclass
class RangeGapConfig(_FromDict):
    """B7 - drop points sitting far BEHIND the object their mask claimed.

    B6 has no z-buffer, so a mask claims its whole line of sight: a sofa at 2 m collects the
    wall at 5 m, and B3 cannot help because both are in range. Sort by range, cut at the
    largest gap above `min_gap`, keep the near side. A GAP rather than a depth budget, because
    an object's own returns stay continuous however deep it is."""
    enabled: bool = True
    min_points: int = 6             # fewer than this: skip the gap search, ceiling only
    min_gap: float = 0.35           # metres; smaller gaps are within-object structure
    max_depth_scale: float = 0.0    # 0 disables; else keep range <= front + scale * apparent width


@dataclass
class DbscanConfig(_FromDict):
    """D1 - separate an object's body from strays attached to it. Not instance separation.

    `min_points` was 20 at eps=2 voxels: unreachable for a surface cloud (~2 neighbours per
    voxel), so 100% of pillow voxels were classified noise and the objects vanished.
    """
    eps_voxels: float = 3.0         # eps = eps_voxels * voxel_size
    min_points: int = 3


@dataclass
class DimensionPriorsConfig(_FromDict):
    """D3 - per-class size CAPS for rejecting bled clusters. NOT expected sizes.

    Derived from VLA-3D ground truth (263 classes, 15 scenes) into dimension_priors.json; see
    scripts/eval/build_dimension_priors.py. A cap below the largest real instance deletes
    correct data. The bled multi-instance blobs these reject are a D8 defect surfacing here."""
    enabled: bool = True
    #: MERGED over the derived table, not a replacement — naming three classes must not
    #: silently revert the other 260 to `default`.
    priors: dict = field(default_factory=dict)

    def __post_init__(self):
        merged = {"default": DEFAULT_PRIOR, **DERIVED_PRIORS}
        merged.update(self.priors or {})
        # yaml gives lists; the comparison in _fits_prior wants indexable triples either way,
        # but normalise so equality/repr is stable across yaml and defaults.
        self.priors = {k: tuple(float(x) for x in v) for k, v in merged.items()}

    def for_label(self, label):
        return self.priors.get(label, self.priors["default"])


@dataclass
class PruneConfig(_FromDict):
    """D6 - remove degenerate objects. Both conditions must hold."""
    enabled: bool = True
    min_valid_voxels: int = 20
    min_inactive_frames: int = 5


@dataclass
class WorldMergeConfig(_FromDict):
    """D8 - fuse same-label objects that are close in world space.

    `absolute_distance` fires unconditionally and is why arabic_room's pillows (0.44 m
    apart) fused. `extent_scale` feeds a threshold that grows with the box it just merged,
    so merges chain — `column` was seen climbing 0.077 -> 0.813 m over five ids.
    """
    enabled: bool = True
    extent_scale: float = 0.5       # thresh = ||(ext_a/2 + ext_b/2)/2|| * extent_scale
    absolute_distance: float = 0.5  # ...OR closer than this, unconditionally

    #: Two ids SAM 3 saw in the SAME frame are different objects, at any distance —
    #: something distance alone cannot know. Blocking those merges is what stops the
    #: multi-instance blobs D3 then has to reject wholesale.
    #:
    #: On its own it over-fires, because SAM 3 also splits ONE object across two ids in a
    #: frame: blanket blocking took duplicate_rate 0.029 -> 0.189. Hence the overlap guard
    #: below.
    block_covisible: bool = True

    #: Co-visibility only blocks when the two are also essentially DISJOINT. Overlap is
    #: intersection-over-MINIMUM-volume, not IoU: a fragment is far smaller than the object
    #: it broke off, so containment is the question. Above this, merge anyway.
    #:
    #: 0.7 sits in an empty band: distinct objects reach 0.48 even with 2x-inflated boxes,
    #: fragments start at 0.91. An earlier 0.10 failed at just 1.2x inflation.
    covisible_overlap: float = 0.7


@dataclass
class MappingConfig(_FromDict):
    """The whole mapper's tuning surface. Stage letters refer to docs/map_node_pipeline.md."""

    # -- B: point -> mask assignment (declared in EXECUTION order) ---------------------
    confidence_threshold: float = 0.30      # B1; sam_node already gates harder (0.7 sim)
    erosion: ErosionConfig = field(default_factory=ErosionConfig)                 # B2
    range_filter: RangeFilterConfig = field(default_factory=RangeFilterConfig)    # B3
    #: B5. "clip" pins out-of-FOV points to row 0/639 where an edge mask swallows them;
    #: "reject" drops them.
    bounds_mode: str = "clip"
    range_gap: RangeGapConfig = field(default_factory=RangeGapConfig)             # B7
    min_points_per_detection: int = 5       # B8
    voxel_downsample: VoxelDownsampleConfig = field(default_factory=VoxelDownsampleConfig)  # B9
    outlier_removal: OutlierRemovalConfig = field(default_factory=OutlierRemovalConfig)     # B10

    # -- C: association ---------------------------------------------------------------
    num_angle_bins: int = 20                # C4; observation-diversity bins per voxel

    # -- D: per-object geometry -------------------------------------------------------
    dbscan: DbscanConfig = field(default_factory=DbscanConfig)                    # D1
    cluster_weight_min: float = 10.0        # D2
    dimension_priors: DimensionPriorsConfig = field(default_factory=DimensionPriorsConfig)  # D3
    percentile_threshold: float = 0.8       # D4 accept-stop AND D5 diversity trim
    prune: PruneConfig = field(default_factory=PruneConfig)                       # D6
    world_merge: WorldMergeConfig = field(default_factory=WorldMergeConfig)       # D8

    # -- E: output --------------------------------------------------------------------
    #: Measured: oriented LOSES on every metric (cat2 0.473 -> 0.366). A minimum-area
    #: rectangle on a partial cloud locks onto the visible face at an arbitrary yaw.
    publish_oriented_box: bool = False

    def __post_init__(self):
        if self.bounds_mode not in ("clip", "reject"):
            raise ValueError(f"bounds_mode must be 'clip' or 'reject', got {self.bounds_mode!r}")
        self.erosion = _sub(ErosionConfig, self.erosion)
        self.range_filter = _sub(RangeFilterConfig, self.range_filter)
        self.range_gap = _sub(RangeGapConfig, self.range_gap)
        self.voxel_downsample = _sub(VoxelDownsampleConfig, self.voxel_downsample)
        self.outlier_removal = _sub(OutlierRemovalConfig, self.outlier_removal)
        self.dbscan = _sub(DbscanConfig, self.dbscan)
        self.dimension_priors = _sub(DimensionPriorsConfig, self.dimension_priors)
        self.prune = _sub(PruneConfig, self.prune)
        self.world_merge = _sub(WorldMergeConfig, self.world_merge)

    @property
    def voxel_size(self) -> float:
        """Accumulation grid size. Lives under voxel_downsample because it is the same
        number, but it is read all over the geometry stage."""
        return self.voxel_downsample.voxel_size
