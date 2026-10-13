from pathlib import Path

import numpy as np

from cucu_replication.acoustics import select_landmarks
from cucu_replication.io import EpochFile, open_fdt, trigger_to_stream


def test_trigger_mapping():
    assert trigger_to_stream("S 11") == 1
    assert trigger_to_stream("S145") == 45
    assert trigger_to_stream("S146") is None


def test_fdt_fortran_layout(tmp_path):
    expected = np.arange(2 * 3 * 4, dtype="<f4").reshape((2, 3, 4), order="F")
    path = tmp_path / "data.fdt"
    expected.ravel(order="F").tofile(path)
    epoch = EpochFile("P1", tmp_path / "P1.set", path, 2, 3, 4, 1000.0, np.zeros(4), (), 0)
    np.testing.assert_array_equal(np.asarray(open_fdt(epoch)), expected)


def test_landmarks_are_measured_candidates_and_not_grid_snapped():
    sampling_rate = 1000
    peaks = np.asarray([101 + 250 * index + (index % 3) * 3 for index in range(20)])
    values = np.zeros(5000)
    values[peaks] = np.linspace(1.0, 2.0, peaks.size)
    selected, candidates, _ = select_landmarks(values, sampling_rate, 20, 80, 250, 50)
    assert selected.size == 20
    assert np.all(np.isin(selected, candidates))
    np.testing.assert_array_equal(selected, peaks)
