"""Tree / double-binary-tree cost. Latency-optimal end of the spectrum.

Reduce up a tree then broadcast down: ~2*log2(N) latency-bound steps. 
Double-binary tree keeps links busy in both directions. This is the small-message
competitor to ring; its crossover vs ring is one of the headline results.

Same reduce-then-broadcast skeleton as ring, only the routing changes in this. 
Ring threads one chunk (S/N) through every core, paying N-1 hops per phase. The tree
collapses pairs in parallel, halving the live set each level, so it reaches the
root in ceil(log2(N)) levels, but every message is the WHOLE tensor S, never a chunk. 
That swap, few-big-messages for many-small-ones, is the entire trade:
  - latency term 2*log2(N)*alpha grows logarithmically  -> wins for small S
  - bandwidth term carries full S per level, serialized  -> loses for large S

The trace to this model (binomial tree, distance 1 then 2 then 4 ...) is derived
by hand from a real matmul here: https://medium.com/@varuntej07 

Scope note: this implements the single (binomial) tree the blog traces. The
DOUBLE binary tree is the bandwidth refinement, run two complementary trees so
a leaf in one is internal in the other, keeping every link busy both directions
and recovering the ~log2(N) bandwidth factor lost below. Left as a TODO so the
single-tree cost stays readable; see bytes_per_link note in tree_phase.
"""

import math
from simulator.cost_model.alpha_beta import step_time, bandwidth_util
from simulator.report import CostReport


def tree_depth(n_nodes: int) -> int:
    """ceil(log2(N)) - levels in the tree, hence steps in one phase.

    A binary tree halves the live set each level, so it needs as many levels as
    the number of times you can halve N: log2(N) when N is a power of two,
    rounded up otherwise (a non-power-of-two tree just carries one ragged,
    not-quite-full level). N=4 -> 2, N=8 -> 3, N=256 -> 8, N=5 -> 3.
    """
    if n_nodes < 2:
        return 0
    return math.ceil(math.log2(n_nodes))


def tree_phase(n_nodes: int, tensor_bytes: float, alpha: float,
               bandwidth_bytes_s: float) -> CostReport:
    """Cost of one tree phase (reduce-up or broadcast-down).

    Both phases run the identical pattern: ceil(log2(N)) sequential levels, and
    the signature of the tree is that every level ships the WHOLE tensor, not the
    S/N chunk a ring would send. Reduce-up adds on arrival, broadcast-down just
    stores, but the bytes on the wire are the same, so they share one helper (mirroring ring_phase).

    The key difference from a ring is in the communication pattern. 
    In a ring, bytes_per_link is the per-PHYSICAL-link reading: the busiest link (an edge at the root) 
    carries the full tensor exactly once this phase, so it is S, not
    steps*S. The steps*S of transfer that total_time_s pays for is a CRITICAL-PATH effect - 
    log2(N) full-S messages chained by data dependency (a node cannot forward a partial sum up until it has received and added it), 
    not one link moving steps*S. That gap is the whole story: bandwidth_util then comes
    out ~1/log2(N) even at zero latency, which is precisely the tree's bandwidth
    penalty, each link sits idle log2(N)-1 of every log2(N) steps while the
    critical path threads through other links. A bandwidth-optimal collective (ring) 
    keeps that link busy every step; the single tree cannot.
    """
    if n_nodes < 2:
        # One core: nothing to communicate. A zero-cost, zero-step report.
        return CostReport(total_time_s=0.0, bytes_per_link=0.0, bandwidth_util=0.0, steps=0)

    steps = tree_depth(n_nodes)
    # Full tensor every step - the price the tree pays for log2(N) step count.
    per_step = step_time(alpha, tensor_bytes, bandwidth_bytes_s)
    total_time_s = steps * per_step

    bytes_per_link = tensor_bytes
    util = bandwidth_util(bytes_per_link, bandwidth_bytes_s, total_time_s)

    return CostReport(total_time_s=total_time_s, bytes_per_link=bytes_per_link,
                      bandwidth_util=util, steps=steps)


def reduce_up(n_nodes: int, tensor_bytes: float, alpha: float,
              bandwidth_bytes_s: float) -> CostReport:
    """Tree reduce-up: ceil(log2(N)) levels, the reduction inward.

    The live set halves each level (N -> N/2 -> ... -> 1) until the root core
    alone holds the complete sum across all N cores. The full reduction now
    exists, but on one core only, so this is half of all-reduce -- the broadcast
    still has to spread it back out.
    """
    return tree_phase(n_nodes, tensor_bytes, alpha, bandwidth_bytes_s)


def broadcast_down(n_nodes: int, tensor_bytes: float, alpha: float,
                   bandwidth_bytes_s: float) -> CostReport:
    """Tree broadcast-down: ceil(log2(N)) levels, the result outward.

    The mirror of reduce-up, run backward: the informed set doubles each level
    (1 -> 2 -> ... -> N) as the root unfolds the finished result down the tree,
    until every core holds it and the every-core postcondition is satisfied.
    Same levels, same full-tensor messages, no addition.
    """
    return tree_phase(n_nodes, tensor_bytes, alpha, bandwidth_bytes_s)


def all_reduce(n_nodes: int, tensor_bytes: float, alpha: float,
               bandwidth_bytes_s: float) -> CostReport:
    """Tree all-reduce = reduce-up then broadcast-down = 2*ceil(log2(N)) steps.

    Composed from the two phases so the decomposition
        all_reduce == reduce_up + broadcast_down
    holds by construction, the same way ring stacks reduce-scatter + all-gather.
    The first log2(N) folds the partials inward to the root; the second log2(N)
    unfolds the sum back out so every core holds it. The 2 is the same structural
    2 as the ring - one direction in, one direction out - not a fudge factor.
    What differs from ring is only the per-direction cost: log2(N) here vs N-1
    there. For N=256 that is 2*8 = 16 steps against the ring's 2*255 = 510.
    """
    if n_nodes < 2:
        return CostReport(total_time_s=0.0, bytes_per_link=0.0,
                          bandwidth_util=0.0, steps=0)

    up = reduce_up(n_nodes, tensor_bytes, alpha, bandwidth_bytes_s)
    down = broadcast_down(n_nodes, tensor_bytes, alpha, bandwidth_bytes_s)

    total_time_s = up.total_time_s + down.total_time_s
    # Same root edge carries S up then S down -> 2S on the busiest physical link.
    bytes_per_link = up.bytes_per_link + down.bytes_per_link
    util = bandwidth_util(bytes_per_link, bandwidth_bytes_s, total_time_s)

    return CostReport(total_time_s=total_time_s, bytes_per_link=bytes_per_link,
                      bandwidth_util=util, steps=up.steps + down.steps)
