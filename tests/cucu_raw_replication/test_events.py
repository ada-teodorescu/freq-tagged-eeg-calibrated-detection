from pathlib import Path

import numpy as np
import pytest

from cucu_raw_replication.events import (
    decode_target_description, inspect_vhdr, target_events, validate_events,
)


@pytest.mark.parametrize(("description", "expected"), [
    ("S 11", 1), ("S 19", 9), ("S110", 10), ("S145", 45),
    ("S146", None), ("S195", None), ("S 21", None), ("S  1", None),
    ("Response/R 1", None),
])
def test_presentation_trigger_decoding(description, expected):
    assert decode_target_description(description) == expected


def test_event_validation_requires_ten_trials_per_stream():
    streams = np.repeat(np.arange(1, 46), 10)
    events = np.column_stack([np.arange(streams.size) * 7000, np.zeros(streams.size), streams])
    counts = validate_events(events, 45, 10)
    assert counts[1] == counts[45] == 10
    with pytest.raises(ValueError, match="Expected 450"):
        validate_events(events[:-1], 45, 10)


def test_released_p13_header_and_markers_have_expected_structure():
    root = Path(__file__).resolve().parents[2]
    path = root / "data" / "Raw participant data" / "P13 E2 Oana.vhdr"
    if not path.exists():
        pytest.skip("Public raw data are not present")
    header = inspect_vhdr(path)
    events = target_events(header.vmrk_path)
    assert header.participant == "P13"
    assert header.channels[:3] == ("Fp1", "Fp2", "F7")
    assert len(header.channels) == 32
    assert header.sampling_rate == 1000
    validate_events(events, 45, 10)
