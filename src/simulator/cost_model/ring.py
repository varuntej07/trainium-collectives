"""Ring collectives cost: all-reduce, reduce-scatter, all-gather. Bandwidth-optimal.

All-reduce is one collective (every core ends up with the full reduction). The
ring *implements* it as reduce-scatter then all-gather, N-1 steps each, hence 2(N-1). 
A naive all-reduce finishes in N-1 steps but moves the whole tensor every step, 
so per-link traffic grows with the ring. Splitting into two phases moves
only one chunk (S/N) per step, so per-link bytes flatten at ~2S regardless of N.
More steps, minimal bytes: the trade you want when bandwidth, not latency, dominates, 
which is why ring is the production choice for large tensors.

all_reduce is built by composing the two phases, not a hard-coded formula. Cost
is in plain (alpha, bandwidth) floats, same shape as alpha_beta. Full derivation
traced through a row-parallel matmul with real partial sums:
https://medium.com/@varuntej07/why-all-reduce-is-reduce-scatter-all-gather-and-why-is-the-cost-2-n-1-not-n-1-229f3cd73d25
"""

from simulator.cost_model.alpha_beta import step_time, bandwidth_util
from simulator.report import CostReport


def ring_phase(n_nodes: int, tensor_bytes: float, alpha: float,
                bandwidth_bytes_s: float, n_steps: int) -> CostReport:
    """Cost of one N-1-step ring phase (reduce-scatter or all-gather).

    Both phases run the identical pattern: n_steps neighbour-only hops, each
    moving one chunk of tensor_bytes/N. Reduce-scatter adds on arrival,
    all-gather just stores, but the bytes on the wire are the same, so they share
    one helper. Only the step count differs, and here it is always N-1.
    All-reduce stacks two of these.
    """
    if n_nodes < 2:
        # One core: nothing to communicate. A zero-cost, zero-step report.
        return CostReport(total_time_s=0.0, bytes_per_link=0.0, bandwidth_util=0.0, steps=0)

    chunk_bytes = tensor_bytes / n_nodes
    per_step = step_time(alpha, chunk_bytes, bandwidth_bytes_s)
    total_time_s = n_steps * per_step

    # Every step pushes exactly one chunk (S/N) across the busiest link, so the
    # phase carries n_steps * S/N = (N-1) * S/N bytes there. The additions are free; 
    # this is the only thing that costs anything.
    bytes_per_link = n_steps * chunk_bytes
    util = bandwidth_util(bytes_per_link, bandwidth_bytes_s, total_time_s)

    return CostReport(total_time_s=total_time_s, bytes_per_link=bytes_per_link,
                      bandwidth_util=util, steps=n_steps)


def reduce_scatter(n_nodes: int, tensor_bytes: float, alpha: float,
                   bandwidth_bytes_s: float) -> CostReport:
    """Ring reduce-scatter: N-1 steps, the reduction inward.

    Ends with each core owning exactly one chunk that is fully summed across all
    N cores, the finished chunks scattered one per core. The complete sum now
    exists in the system but it is shredded, so this is only half of all-reduce.
    """
    return ring_phase(n_nodes, tensor_bytes, alpha, bandwidth_bytes_s,
                       n_steps=max(n_nodes - 1, 0))


def all_gather(n_nodes: int, tensor_bytes: float, alpha: float,
               bandwidth_bytes_s: float) -> CostReport:
    """Ring all-gather: N-1 steps, the broadcast outward.

    Same ring, same neighbour-only sends, but now no addition. Each core forwards
    a finished chunk and stores what arrives, circulating until every core holds
    all N chunks and the every-core postcondition is satisfied.
    """
    return ring_phase(n_nodes, tensor_bytes, alpha, bandwidth_bytes_s,
                       n_steps=max(n_nodes - 1, 0))


def all_reduce(n_nodes: int, tensor_bytes: float, alpha: float,
               bandwidth_bytes_s: float) -> CostReport:
    """Ring all-reduce = reduce-scatter then all-gather = 2(N-1) steps.

    Composed from the two phases so the decomposition
        all_reduce == reduce_scatter + all_gather
    holds by construction, not by a re-derived formula that could drift. The first N-1 collapses 
    the partials into a scattered sum (reduction inward), the second N-1 spreads that sum back out 
    so every core holds it (broadcast outward). The 2 is not a fudge factor and 
    not a redundant second lap: it is two structurally distinct jobs, summed.
    """
    if n_nodes < 2:
        return CostReport(total_time_s=0.0, bytes_per_link=0.0,
                          bandwidth_util=0.0, steps=0)

    rs = reduce_scatter(n_nodes, tensor_bytes, alpha, bandwidth_bytes_s)
    ag = all_gather(n_nodes, tensor_bytes, alpha, bandwidth_bytes_s)

    total_time_s = rs.total_time_s + ag.total_time_s
    bytes_per_link = rs.bytes_per_link + ag.bytes_per_link
    util = bandwidth_util(bytes_per_link, bandwidth_bytes_s, total_time_s)

    return CostReport(total_time_s=total_time_s, bytes_per_link=bytes_per_link,
                      bandwidth_util=util, steps=rs.steps + ag.steps)
