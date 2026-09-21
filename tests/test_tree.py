"""Tree: the latency-optimal end, and the bandwidth price it pays for it.

tree.py argues in prose that a single tree's utilisation comes out at about
1/log2(N) even at zero latency, because each link idles while the critical path
threads through other links. That is the claim most worth pinning, because it is
the one a reader is most likely to doubt.
"""

import math

import pytest

from simulator import CostReport
from simulator.cost_model import ring, tree

SIZES = [2, 4, 8, 16, 128, 1024]
RAGGED = [3, 5, 17]


@pytest.mark.parametrize("n,depth", [(0, 0), (1, 0), (2, 1), (4, 2), (5, 3), (8, 3), (256, 8)])
def test_tree_depth(n, depth):
    assert tree.tree_depth(n) == depth


@pytest.mark.parametrize("n", SIZES + RAGGED)
def test_all_reduce_is_exactly_its_two_phases(n):
    s = 64 * 1024 * 1024
    up = tree.reduce_up(n, s, 1e-6, 100e9)
    down = tree.broadcast_down(n, s, 1e-6, 100e9)
    ar = tree.all_reduce(n, s, 1e-6, 100e9)

    assert ar.total_time_s == pytest.approx(up.total_time_s + down.total_time_s)
    assert ar.bytes_per_link == pytest.approx(up.bytes_per_link + down.bytes_per_link)
    assert ar.steps == up.steps + down.steps
    assert up == down


@pytest.mark.parametrize("n", SIZES + RAGGED)
def test_step_count_is_twice_the_depth(n):
    assert tree.all_reduce(n, 1e8, 1e-6, 100e9).steps == 2 * math.ceil(math.log2(n))


def test_the_step_count_gap_against_ring_at_256():
    # The comparison tree.py makes by hand in its own docstring.
    s = 1e8
    assert tree.all_reduce(256, s, 1e-6, 100e9).steps == 16
    assert ring.all_reduce(256, s, 1e-6, 100e9).steps == 510


@pytest.mark.parametrize("n", [0, 1])
def test_one_core_or_none_costs_nothing(n):
    zero = CostReport(total_time_s=0.0, bytes_per_link=0.0, bandwidth_util=0.0, steps=0)
    assert tree.reduce_up(n, 1e8, 1e-6, 100e9) == zero
    assert tree.broadcast_down(n, 1e8, 1e-6, 100e9) == zero
    assert tree.all_reduce(n, 1e8, 1e-6, 100e9) == zero


@pytest.mark.parametrize("n", SIZES + RAGGED)
def test_utilisation_is_one_over_the_depth_even_with_free_latency(n):
    # The bandwidth penalty, isolated. alpha is zero here, so nothing but the
    # communication pattern can be responsible for the shortfall.
    ar = tree.all_reduce(n, 1e8, alpha=0.0, bandwidth_bytes_s=100e9)
    assert ar.bandwidth_util == pytest.approx(1 / math.ceil(math.log2(n)), rel=1e-12)


@pytest.mark.parametrize("n", [4, 8, 16, 128, 1024])
def test_tree_wastes_the_link_that_ring_saturates(n):
    # Same tensor, same links, latency removed from both. The only difference
    # left is the routing, and this is what it costs.
    s = 1e8
    t = tree.all_reduce(n, s, alpha=0.0, bandwidth_bytes_s=100e9)
    r = ring.all_reduce(n, s, alpha=0.0, bandwidth_bytes_s=100e9)
    assert t.bandwidth_util < r.bandwidth_util
    assert t.bandwidth_util / r.bandwidth_util == pytest.approx(1 / math.ceil(math.log2(n)), rel=1e-12)


@pytest.mark.parametrize("n", SIZES)
def test_root_edge_carries_2s_regardless_of_n(n):
    # S up then S down over the same edge. Contrast with ring, where per-link
    # bytes are 2(N-1)/N * S and therefore depend on the ring length.
    s = 1e8
    assert tree.all_reduce(n, s, 1e-6, 100e9).bytes_per_link == pytest.approx(2 * s)


def test_the_crossover_against_ring_exists():
    # The headline reason both models are in the repo. Latency-bound at the
    # small end, bandwidth-bound at the large end, and the winner swaps.
    n, alpha, bw = 128, 15e-6, 100e9
    small, large = 4 * 1024, 1024 ** 3
    assert tree.all_reduce(n, small, alpha, bw).total_time_s < ring.all_reduce(n, small, alpha, bw).total_time_s
    assert ring.all_reduce(n, large, alpha, bw).total_time_s < tree.all_reduce(n, large, alpha, bw).total_time_s
