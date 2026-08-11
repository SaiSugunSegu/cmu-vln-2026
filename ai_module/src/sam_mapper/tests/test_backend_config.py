"""Do the yaml tuning knobs actually reach the model?

Regression tests for a bug that ran silently for the life of the file: overrides were
written to `model.config` only, but `Sam3VideoModel.__init__` COPIES each knob onto the
module (`self.score_threshold_detection = config.score_threshold_detection`) and the forward
path reads the copy. Every threshold in sam3_*.yaml was parsed, assigned, and ignored.

Nothing here needs torch — `_apply_override` only touches `self.model` and `self.log`, so it
is exercised unbound against a stand-in shaped like the real model.

    python -m pytest ai_module/src/sam_mapper/tests/test_backend_config.py
"""
from types import SimpleNamespace

from sam_mapper.sam3_backend import Sam3Backend


def make_backend(config_attrs: dict, model_attrs: dict):
    """A Sam3Backend-shaped stand-in. `model` carries both a `.config` and the module-level
    copies, exactly like Sam3VideoModel."""
    logs = []
    model = SimpleNamespace(config=SimpleNamespace(**config_attrs), **model_attrs)
    return SimpleNamespace(model=model, log=logs.append), logs


def test_override_reaches_the_module_attribute_not_just_the_config():
    """The bug: only the config was written, so the model kept the checkpoint default."""
    backend, _ = make_backend({"score_threshold_detection": 0.5},
                              {"score_threshold_detection": 0.5})

    Sam3Backend._apply_override(backend, "score_threshold_detection", 0.7)

    assert backend.model.score_threshold_detection == 0.7    # what the forward path reads
    assert backend.model.config.score_threshold_detection == 0.7  # what a save round-trips


def test_override_applies_when_only_the_config_has_the_key():
    backend, _ = make_backend({"hotstart_delay": 15}, {})
    Sam3Backend._apply_override(backend, "hotstart_delay", 3)
    assert backend.model.config.hotstart_delay == 3


def test_override_applies_when_only_the_module_has_the_key():
    backend, _ = make_backend({}, {"fill_hole_area": 16})
    Sam3Backend._apply_override(backend, "fill_hole_area", 0)
    assert backend.model.fill_hole_area == 0


def test_unknown_key_is_reported_rather_than_silently_dropped():
    """A knob transformers renamed must not vanish quietly — that is the whole failure mode
    this module exists to prevent."""
    backend, logs = make_backend({}, {})
    Sam3Backend._apply_override(backend, "renamed_upstream", 1.0)
    assert any("renamed_upstream" in line for line in logs)


def test_zero_is_applied_not_treated_as_unset():
    """fill_hole_area: 0 is the supported way to disable cv-utils hole filling and sprinkle
    removal, so a falsy-value check here would make that setting unreachable."""
    backend, _ = make_backend({"fill_hole_area": 16}, {"fill_hole_area": 16})
    Sam3Backend._apply_override(backend, "fill_hole_area", 0)
    assert backend.model.fill_hole_area == 0


def test_log_effective_config_reports_module_values():
    backend, logs = make_backend({"score_threshold_detection": 0.5},
                                 {"score_threshold_detection": 0.5, "det_nms_thresh": 0.1})
    Sam3Backend._apply_override(backend, "score_threshold_detection", 0.7)
    Sam3Backend.log_effective_config(
        backend, ("score_threshold_detection", "det_nms_thresh", "not_a_key"))

    line = logs[-1]
    assert "score_threshold_detection=0.7" in line
    assert "det_nms_thresh=0.1" in line
    assert "not_a_key" not in line
