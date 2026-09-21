"""Hierarchical: the claim that only the shard crosses the slow tier.

The whole algorithm exists to keep full-tensor traffic off the inter-node links.
These tests check that it actually does, and they pin the one place where the
report deliberately does not mean what the flat models mean by the same field.
"""

import pytest

from simulator import CostReport
from simulator.cost_model import hierarchical, ring

# Trn1-shaped inputs: NeuronLink inside the box, EFA between boxes.
INTRA_BW, INTRA_ALPHA = 768e9, 1e-6
INTER_BW, INTER_ALPHA = 100e9, 15e-6
S = 128 * 1024 * 1024


def _run(intra_size, inter_size, tensor_bytes=S):
    return hierarchical.all_reduce(intra_size, inter_size, tensor_bytes,
                                   INTRA_ALPHA, INTRA_BW, INTER_ALPHA, INTER_BW)


def test_single_instance_degenerates_to_an_intra_node_ring():
    # hierarchical.py says inter_size=1 makes phase 2 free and the op becomes a
    # plain intra-node all-reduce. True for cost and step count, and bit-exact:
    flat = ring.all_reduce(16, S, INTRA_ALPHA, INTRA_BW)
    h = _run(intra_size=16, inter_size=1)
    assert h.total_time_s == flat.total_time_s
    assert h.steps == flat.steps

    # But not for the link accounting, and that is intentional rather than a
    # rounding artefact. hierarchical reports what crossed the inter tier, and
    # with one instance nothing does. The flat ring has no second tier to
    # exclude, so it reports its own edge. Same collective, two different
    # questions being answered.
    assert h.bytes_per_link == 0.0
    assert h.bandwidth_util == 0.0
    assert flat.bytes_per_link > 0.0


@pytest.mark.parametrize("inter_size", [2, 4, 8, 16, 64])
def test_only_the_shard_crosses_the_slow_tier(inter_size):
    intra_size = 16
    shard = S / intra_size
    h = _run(intra_size, inter_size)
    # A ring all-reduce of the shard across the instances, nothing more.
    assert h.bytes_per_link == pytest.approx(2 * (inter_size - 1) / inter_size * shard)
    assert h.bytes_per_link < 2 * shard


@pytest.mark.parametrize("inter_size", [2, 4, 8, 16, 64])
def test_inter_node_traffic_is_cut_by_roughly_the_box_size(inter_size):
    # What a flat ring would push over the same EFA link is about 2S. The
    # hierarchy should be lighter by close to a factor of intra_size.
    intra_size = 16
    h = _run(intra_size, inter_size)
    assert h.bytes_per_link < 2 * S / intra_size
    assert 2 * S / h.bytes_per_link > intra_size


@pytest.mark.parametrize("intra_size,inter_size", [(4, 2), (8, 4), (16, 8), (32, 16)])
def test_step_count_is_both_tiers_summed(intra_size, inter_size):
    h = _run(intra_size, inter_size)
    assert h.steps == 2 * (intra_size - 1) + 2 * (inter_size - 1)


def test_larger_boxes_push_less_across_the_slow_tier():
    # Fixed total ranks, moved between the tiers. Bigger box, smaller shard.
    bytes_out = [_run(intra, 128 // intra).bytes_per_link for intra in (4, 8, 16, 32)]
    assert bytes_out == sorted(bytes_out, reverse=True)


def test_single_chip_costs_nothing():
    assert _run(intra_size=1, inter_size=1) == CostReport(
        total_time_s=0.0, bytes_per_link=0.0, bandwidth_util=0.0, steps=0)
    # intra_size=0 is not defined: line 42 divides the tensor by it. Out of
    # scope on purpose, so nobody later adds a case expecting it to pass.


def test_low_inter_node_utilisation_is_the_success_signal():
    # A low number here is the point, not a warning. The two NeuronLink phases
    # never touch EFA, so EFA sits idle for most of the op, which is exactly
    # what "this collective is not inter-node-bound" looks like.
    assert _run(intra_size=16, inter_size=8).bandwidth_util < 0.25
