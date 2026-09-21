"""The numbers the README publishes, held to account.

The README's "Current finding" section makes a specific, falsifiable claim: at
eight instances of sixteen chips, hierarchical ring beats flat ring across the
entire 4 KiB to 1 GiB sweep, and the reason is latency, not bandwidth. If one of
these fails, either the model changed or the README is now lying to a reader.
Both are worth stopping for.

The flat ring is costed on EFA here. That is not a handicap, it is the situation:
a single cycle through 128 ranks spread over eight boxes has EFA links on it, and
every step is paced by the slowest link on the ring. That is what the README
means by "254 EFA-gated steps".
"""

import pytest

from simulator.cost_model import hierarchical, ring

INTRA_SIZE, INTER_SIZE = 16, 8
INTRA_BW, INTRA_ALPHA = 768e9, 1e-6
INTER_BW, INTER_ALPHA = 100e9, 15e-6

RANKS = INTRA_SIZE * INTER_SIZE
SWEEP = [2 ** k for k in range(12, 31)]  # 4 KiB to 1 GiB


def _flat(tensor_bytes):
    return ring.all_reduce(RANKS, tensor_bytes, INTER_ALPHA, INTER_BW)


def _tiered(tensor_bytes):
    return hierarchical.all_reduce(INTRA_SIZE, INTER_SIZE, tensor_bytes,
                                   INTRA_ALPHA, INTRA_BW, INTER_ALPHA, INTER_BW)


def test_the_sweep_is_the_range_the_readme_quotes():
    assert SWEEP[0] == 4 * 1024
    assert SWEEP[-1] == 1024 ** 3
    assert len(SWEEP) == 19


def test_the_step_counts_the_readme_prints():
    assert _flat(SWEEP[0]).steps == 254
    assert _tiered(SWEEP[0]).steps == 44


@pytest.mark.parametrize("tensor_bytes", SWEEP)
def test_hierarchical_wins_at_every_size(tensor_bytes):
    assert _tiered(tensor_bytes).total_time_s < _flat(tensor_bytes).total_time_s


def test_the_worst_margin_is_still_large():
    # A floor rather than an equality, so ordinary float drift does not break
    # the build but a real regression does. Measured worst case is about 6.2x.
    margins = [_flat(b).total_time_s / _tiered(b).total_time_s for b in SWEEP]
    assert min(margins) >= 6.0


def test_the_advantage_shrinks_with_size_but_never_crosses():
    # The honest shape of the finding. This is not a crossover that has been
    # pushed past 1 GiB, it is an advantage that narrows monotonically and stays
    # positive. Saying so is the difference between a result and a claim.
    margins = [_flat(b).total_time_s / _tiered(b).total_time_s for b in SWEEP]
    assert margins == sorted(margins, reverse=True)
    assert margins.index(min(margins)) == len(SWEEP) - 1
    assert min(margins) > 1.0
