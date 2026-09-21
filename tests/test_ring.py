"""Ring: the bandwidth-optimal end of the spectrum.

ring.all_reduce claims its decomposition holds "by construction, not by a
re-derived formula that could drift". This file is what turns that comment into
something that fails loudly if it ever stops being true.
"""

import pytest

from simulator import CostReport
from simulator.cost_model import ring

SIZES = [2, 3, 4, 8, 16, 128]


@pytest.mark.parametrize("n", SIZES)
def test_all_reduce_is_exactly_its_two_phases(n):
    s = 64 * 1024 * 1024
    rs = ring.reduce_scatter(n, s, 1e-6, 100e9)
    ag = ring.all_gather(n, s, 1e-6, 100e9)
    ar = ring.all_reduce(n, s, 1e-6, 100e9)

    assert ar.total_time_s == pytest.approx(rs.total_time_s + ag.total_time_s)
    assert ar.bytes_per_link == pytest.approx(rs.bytes_per_link + ag.bytes_per_link)
    assert ar.steps == rs.steps + ag.steps
    # bandwidth_util is deliberately not in this list. It is a ratio recomputed
    # over the combined time, not a quantity that adds.


@pytest.mark.parametrize("n", SIZES)
def test_both_phases_move_identical_bytes(n):
    # Reduce-scatter adds on arrival and all-gather only stores, but the bytes
    # on the wire are the same, which is why they share one helper.
    s = 1e8
    assert ring.reduce_scatter(n, s, 1e-6, 100e9) == ring.all_gather(n, s, 1e-6, 100e9)


@pytest.mark.parametrize("n", SIZES)
def test_step_counts(n):
    s = 1e8
    assert ring.reduce_scatter(n, s, 1e-6, 100e9).steps == n - 1
    assert ring.all_reduce(n, s, 1e-6, 100e9).steps == 2 * (n - 1)


@pytest.mark.parametrize("n", [0, 1])
def test_one_core_or_none_costs_nothing(n):
    zero = CostReport(total_time_s=0.0, bytes_per_link=0.0, bandwidth_util=0.0, steps=0)
    assert ring.reduce_scatter(n, 1e8, 1e-6, 100e9) == zero
    assert ring.all_gather(n, 1e8, 1e-6, 100e9) == zero
    assert ring.all_reduce(n, 1e8, 1e-6, 100e9) == zero


@pytest.mark.parametrize("n", SIZES)
def test_ring_saturates_the_link_when_latency_is_free(n):
    # Ring is bandwidth-optimal: strip alpha out and every step is pure payload,
    # so utilisation is 1. Exact in algebra, one ULP off in float64 because
    # step_time multiplies by 1/B rather than dividing, so compare with a
    # tolerance instead of ==.
    ar = ring.all_reduce(n, 1e8, alpha=0.0, bandwidth_bytes_s=100e9)
    assert ar.bandwidth_util == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("n", SIZES)
def test_per_link_bytes_flatten_at_2s(n):
    # The reason to split all-reduce into two phases at all: per-link traffic
    # approaches 2S and stops there, instead of growing with the ring.
    s = 1e8
    ar = ring.all_reduce(n, s, 1e-6, 100e9)
    assert ar.bytes_per_link == pytest.approx(2 * (n - 1) / n * s)
    assert ar.bytes_per_link < 2 * s


def test_time_grows_with_ring_length():
    s = 1e8
    times = [ring.all_reduce(n, s, 1e-6, 100e9).total_time_s for n in SIZES]
    assert times == sorted(times)
    assert len(set(times)) == len(times)


def test_zero_length_tensor_with_free_latency_raises():
    # Known sharp edge, recorded rather than endorsed. A zero-byte collective
    # across a real ring takes zero time, and bandwidth_util refuses to divide
    # by that. Nothing in the project asks for it today, so it is pinned here
    # so that a future change to the guard is a deliberate decision.
    with pytest.raises(ValueError):
        ring.all_reduce(4, tensor_bytes=0.0, alpha=0.0, bandwidth_bytes_s=100e9)
