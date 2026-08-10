"""Which scenes the orchestrator considers replayable, and what a resume adopts."""
from __future__ import annotations

import json

from smart_vlm.eval_orchestrator import cached_rows, has_bag


def test_mcap_or_metadata_counts_as_a_bag(tmp_path):
    mcap = tmp_path / "arabic_room"
    (mcap / "iref_vla_metadata").mkdir(parents=True)
    (mcap / "arabic_room_0.mcap").touch()
    assert has_bag(mcap)

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "metadata.yaml").touch()
    assert has_bag(legacy)


def test_metadata_only_stub_is_not_a_bag(tmp_path):
    """Some scenes ship the annotations without the recording.

    Counting the stub as playable cost a whole question's warmup timeout each, waiting
    for a /camera/image that was never going to arrive.
    """
    stub = tmp_path / "home_building_1"
    (stub / "iref_vla_metadata").mkdir(parents=True)
    assert not has_bag(stub)


def _report(tmp_path, rows):
    path = tmp_path / "views_cache.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"results": rows}, handle)
    return path


def _crop_dir(tmp_path, name, *, with_crops=True):
    crops = tmp_path / name
    crops.mkdir(parents=True)
    if with_crops:
        (crops / "best_rank1_sofa.png").touch()
    return str(crops)


def test_resume_adopts_rows_whose_crops_survive(tmp_path):
    rows = [
        {"scene": "arabic_room", "id": "Q01", "question": "How many sofas?",
         "best_view_dir": _crop_dir(tmp_path, "arabic_room/Q01"), "error": None},
        # The scene had no bag, so the pipeline timed out and left the directory empty.
        {"scene": "home_building_1", "id": "Q01", "question": "How many pillows?",
         "best_view_dir": _crop_dir(tmp_path, "home_building_1/Q01", with_crops=False),
         "error": None},
        # Ran, but failed — nothing here worth keeping.
        {"scene": "chinese_room", "id": "Q01", "question": "How many chairs?",
         "best_view_dir": _crop_dir(tmp_path, "chinese_room/Q01"),
         "error": "TimeoutError: no answer within 600s"},
    ]
    assert set(cached_rows(_report(tmp_path, rows))) == {("arabic_room", "Q01")}


def test_resume_ignores_a_missing_or_unreadable_report(tmp_path):
    assert cached_rows(tmp_path / "nothing.json") == {}
    truncated = tmp_path / "half.json"
    truncated.write_text('{"results": [{"scene"')
    assert cached_rows(truncated) == {}
