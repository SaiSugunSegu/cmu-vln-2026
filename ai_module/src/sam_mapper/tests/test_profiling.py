"""Unit tests for the SAM 3 stage timer.

Pure python — no GPU, no model, no ROS. The timer is only ever trusted when it is measuring
a real frame, so the properties worth testing here are the structural ones: that it wraps and
UNWRAPS cleanly, that stages land in the frame they belong to, and that disabling it is a
genuine no-op (production runs with profile: false and must pay nothing).

    python -m pytest ai_module/src/sam_mapper/tests/test_profiling.py
"""
import pytest

from sam_mapper.profiling import (MODEL_STAGES, StageTimer, format_per_object_fit,
                                  format_summary)


class FakeDetector:
    def get_vision_features(self, x):
        return ("vision", x)


class FakeModel:
    """Stands in for Sam3VideoModel: the same method names StageTimer wraps."""

    def __init__(self):
        self.detector_model = FakeDetector()
        self.calls = []

    def run_detection(self, **kwargs):
        self.calls.append("run_detection")
        return "det"

    def get_vision_features_for_tracker(self, **kwargs):
        return ("feats", "pos")

    def run_tracker_propagation(self, **kwargs):
        return "prop"

    def run_tracker_update_planning_phase(self, **kwargs):
        return "plan"

    def run_tracker_update_execution_phase(self, **kwargs):
        return None

    def build_outputs(self, **kwargs):
        return {}


def test_attach_wraps_every_declared_stage_and_detach_restores_them():
    model = FakeModel()
    originals = {path: _dig(model, path) for path, _ in MODEL_STAGES}

    timer = StageTimer(enabled=True).attach(model)
    assert timer.missing == []
    assert all(_dig(model, path) is not originals[path] for path, _ in MODEL_STAGES)

    timer.detach()
    for path, _ in MODEL_STAGES:
        assert _dig(model, path) == originals[path]


def test_wrapped_methods_still_return_their_values():
    model = FakeModel()
    StageTimer(enabled=True).attach(model)
    assert model.run_detection(inference_session=None) == "det"
    assert model.detector_model.get_vision_features(1) == ("vision", 1)
    assert model.calls == ["run_detection"]


def test_stages_are_recorded_under_their_report_labels():
    model = FakeModel()
    timer = StageTimer(enabled=True).attach(model)

    model.detector_model.get_vision_features(0)
    model.run_detection()
    model.run_tracker_propagation()
    timer.end_frame(total_ms=10.0, n_objects=3, n_prompts=2)

    assert timer.frames[0].keys() >= {"vision_encoder", "detection", "tracker_propagate"}
    assert timer.frames[0]["_objects"] == 3
    assert timer.frames[0]["_prompts"] == 2


def test_missing_stage_is_reported_not_raised():
    """A transformers upgrade that renames a method must degrade the table, not kill the
    probe mid-sweep."""

    class Renamed(FakeModel):
        run_tracker_propagation = None       # gone, as far as getattr is concerned

    timer = StageTimer(enabled=True).attach(Renamed())
    assert "run_tracker_propagation" in timer.missing
    assert len(timer.missing) == 1           # everything else still wrapped


def test_disabled_timer_wraps_nothing_and_records_nothing():
    model = FakeModel()
    original = model.run_detection
    timer = StageTimer(enabled=False).attach(model)

    assert model.run_detection == original          # not wrapped at all
    with timer.stage("anything"):
        pass
    timer.end_frame(1.0, 1, 1)
    assert timer.frames == []
    assert timer.summary()["frames"] == 0


def test_frame_context_defers_the_close_so_outer_stages_land_in_the_same_record():
    """sam_node's post-SAM stages run after process_frame returns. Without frame() they would
    be recorded against the NEXT frame — the whole table off by one."""
    timer = StageTimer(enabled=True)
    with timer.frame():
        with timer.stage("preprocess"):
            pass
        timer.end_frame(5.0, n_objects=4, n_prompts=2)     # what process_frame does
        with timer.stage("node_publish"):
            pass

    assert len(timer.frames) == 1
    record = timer.frames[0]
    assert {"preprocess", "node_publish"} <= record.keys()
    assert record["_objects"] == 4 and record["_prompts"] == 2


def test_repeated_stage_within_one_frame_accumulates():
    """run_detection is called once per prompt, so its label is entered N times per frame and
    the table must report the total, not the last one."""
    timer = StageTimer(enabled=True)
    for _ in range(3):
        with timer.stage("detection"):
            pass
    timer.end_frame(9.0, 1, 3)
    assert timer.frames[0]["detection"] >= 0.0
    assert "detection" in timer.summary(skip_first=0)["stages"]


def test_summary_drops_warmup_frames_and_uses_medians():
    timer = StageTimer(enabled=True)
    for total in (1000.0, 10.0, 12.0, 11.0):        # frame 0 is the warm-up outlier
        timer._current = {"detection": total}
        timer.end_frame(total, n_objects=2, n_prompts=1)

    summary = timer.summary(skip_first=1)
    assert summary["frames"] == 3
    assert summary["total_ms"] == pytest.approx(11.0)
    assert summary["stages"]["detection"] == pytest.approx(11.0)


def test_summary_accounted_and_residual_add_up():
    timer = StageTimer(enabled=True)
    for _ in range(3):
        timer._current = {"vision_encoder": 30.0, "detection": 20.0}
        timer.end_frame(100.0, n_objects=5, n_prompts=2)

    summary = timer.summary()
    assert summary["accounted_ms"] == pytest.approx(50.0)
    text = format_summary(summary, title="t")
    assert "residual (off-seam)" in text
    assert "a cost is hiding" in text          # 50% unaccounted must be called out


def test_per_object_fit_separates_fixed_cost_from_per_object_cost():
    """The measurement the whole SAM 3.1 question turns on: a stage with a flat per-object
    slope has nothing for Object Multiplex to share."""
    timer = StageTimer(enabled=True)
    for n in (2, 4, 6, 8):
        timer._current = {"vision_encoder": 100.0,            # fixed
                          "build_outputs": 5.0 * n}           # purely per-object
        timer.end_frame(100.0 + 5.0 * n, n_objects=n, n_prompts=1)

    fits = timer.per_object_fit()["fits"]
    assert fits["vision_encoder"]["per_object_ms"] == pytest.approx(0.0, abs=1e-6)
    assert fits["build_outputs"]["per_object_ms"] == pytest.approx(5.0)
    assert fits["build_outputs"]["fixed_ms"] == pytest.approx(0.0, abs=1e-6)


def test_per_object_fit_skips_stages_too_small_to_fit():
    """A stage costing ~0 ms has a range smaller than the timing jitter, so least squares
    hands back a big slope with a nonsense intercept. The first real run reported
    tracker_execute at 42 ms/object while its median was 0.0 ms — exactly this."""
    timer = StageTimer(enabled=True)
    for i, n in enumerate((2, 5, 9, 14)):
        timer._current = {"vision_encoder": 200.0, "tracker_execute": 0.01 * i}
        timer.end_frame(200.0, n_objects=n, n_prompts=1)

    fit = timer.per_object_fit()
    assert "tracker_execute" in fit["skipped"]
    assert "tracker_execute" not in fit["fits"]
    assert "not fitted" in format_per_object_fit(fit)


def test_per_object_fit_flags_an_unreliable_slope():
    timer = StageTimer(enabled=True)
    for n, ms in ((2, 90.0), (5, 130.0), (9, 95.0), (14, 125.0)):   # no real trend
        timer._current = {"detection": ms}
        timer.end_frame(ms, n_objects=n, n_prompts=1)

    fit = timer.per_object_fit()
    assert fit["fits"]["detection"]["reliable"] is False
    assert "low r2" in format_per_object_fit(fit)


def test_per_object_fit_warns_when_the_object_spread_is_too_narrow():
    """3.1 objects/frame in the cat1 regime spans maybe 3-4 objects; slopes read off that
    are not evidence, and the report has to say so rather than look authoritative."""
    timer = StageTimer(enabled=True)
    for n in (3, 4, 3, 4):
        timer._current = {"vision_encoder": 190.0 + n}
        timer.end_frame(190.0 + n, n_objects=n, n_prompts=2)

    text = format_per_object_fit(timer.per_object_fit())
    assert "object count only spans 3-4" in text


def test_per_object_fit_refuses_to_fit_a_single_object_count():
    """Every frame at the same object count means the slope is unidentifiable — say so
    rather than reporting a fabricated number."""
    timer = StageTimer(enabled=True)
    for _ in range(5):
        timer._current = {"detection": 10.0}
        timer.end_frame(10.0, n_objects=3, n_prompts=1)

    fit = timer.per_object_fit()
    assert fit["fits"] == {}
    assert "needs" in format_per_object_fit(fit)


def test_format_summary_handles_a_run_with_no_frames():
    assert "no frames" in format_summary(StageTimer(enabled=True).summary())


def test_every_name_imported_from_profiling_exists():
    """Most of these are imported INSIDE functions (to keep torch off the import path), so a
    rename fails only when that line runs — for report_device, several minutes into a GPU
    profile. py_compile cannot see it either. Resolve them up front instead."""
    import ast
    import pathlib

    import sam_mapper.profiling as profiling

    package = pathlib.Path(profiling.__file__).parent
    missing = []
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "sam_mapper.profiling":
                missing += [f"{path.name}:{node.lineno} {a.name}" for a in node.names
                            if not hasattr(profiling, a.name)]
    assert not missing, f"imported from sam_mapper.profiling but not defined: {missing}"


def _dig(root, path):
    for part in path.split("."):
        root = getattr(root, part)
    return root
