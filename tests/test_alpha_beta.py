"""The two primitives everything else is built out of.

If step_time is wrong, every model above it is wrong in the same direction and
the comparison between them still looks plausible, which is the worst kind of
bug to have in a cost model. So pin the shape of the function here: a latency
floor, a linear bandwidth term, and nothing else.
"""

import pytest

from simulator.cost_model.alpha_beta import bandwidth_util, step_time


def test_zero_bytes_costs_exactly_alpha():
    # The latency floor. Sending nothing still pays the handshake.
    assert step_time(alpha=5e-6, chunk_bytes=0.0, bandwidth_bytes_s=100e9) == 5e-6


def test_zero_alpha_costs_exactly_the_transfer():
    assert step_time(alpha=0.0, chunk_bytes=1e6, bandwidth_bytes_s=100e9) == pytest.approx(1e6 / 100e9)


def test_step_time_is_affine_in_bytes():
    # Doubling the payload adds the same increment it cost to send the first
    # copy. That is what makes beta a constant per byte rather than a curve.
    a, b = 3e-6, 100e9
    first = step_time(a, 1e6, b) - step_time(a, 0.0, b)
    second = step_time(a, 2e6, b) - step_time(a, 1e6, b)
    assert second == pytest.approx(first)


@pytest.mark.parametrize("bad_bw", [0.0, -1.0, -100e9])
def test_nonpositive_bandwidth_is_rejected(bad_bw):
    with pytest.raises(ValueError):
        step_time(alpha=1e-6, chunk_bytes=1e6, bandwidth_bytes_s=bad_bw)
    with pytest.raises(ValueError):
        bandwidth_util(useful_bytes=1e6, bandwidth_bytes_s=bad_bw, total_time_s=1e-3)


@pytest.mark.parametrize("bad_time", [0.0, -1e-6])
def test_nonpositive_time_is_rejected(bad_time):
    with pytest.raises(ValueError):
        bandwidth_util(useful_bytes=1e6, bandwidth_bytes_s=100e9, total_time_s=bad_time)


def test_utilisation_above_one_is_returned_not_raised():
    # Moving more than the link can physically carry is impossible, so a number
    # above 1 means the assumed peak B is too low and needs recalibrating
    # against measured numbers. The docstring calls that a finding, not a bug,
    # so the function must hand it back rather than swallow it.
    util = bandwidth_util(useful_bytes=1e9, bandwidth_bytes_s=100e9, total_time_s=1e-3)
    assert util > 1.0
