"""
ON A HETEROGENEOUS NETWORK OF CHIPS:

The N ranks are split across machines: inside a box, chips talk over NeuronLink (fast);
across boxes, instances talk over EFA (~8x slower at the aggregate level). A flat ring
treats both as equal links, which is where it goes wrong. A flat ring is a single cycle
through all N ranks, and in 'ring allreduce' every link carries ~2S bytes total
(S = tensor size) including the EFA links. So each inter-node link has to push ~2S
bytes at EFA's slow bandwidth, and since every step is paced by the slowest link, EFA
gates the whole collective while the fast NeuronLink sits idle.

So we don't run one collective flat over all ranks. We decompose by tier and match the
collective to each link's bandwidth:
  -> reduce-scatter + all-gather on fast NeuronLink (full tensor, cheap), and
  -> only an all-reduce of the 1/intra_size shard on slow EFA.
Identical result; the scarce inter-node link now moves ~2S/intra_size instead of ~2S.
That tier-matching is the entire reason hierarchical collectives exist.
"""

from simulator.cost_model import ring
from simulator.cost_model.alpha_beta import bandwidth_util
from simulator.report import CostReport


def all_reduce(intra_size: int, inter_size: int, tensor_bytes: float,
               intra_alpha: float, intra_bw: float,
               inter_alpha: float, inter_bw: float) -> CostReport:
    """Hierarchical all-reduce: reduce-scatter intra -> all-reduce inter -> all-gather intra.

    Phase 1  reduce-scatter across intra_size chips on the fast tier. Each chip
             ends up owning one summed shard of tensor_bytes/intra_size, reduced
             over its own box.
    Phase 2  all-reduce those shards across inter_size nodes on the slow tier.
             Only the shard crosses EFA, never the whole tensor. After this each
             chip's shard is reduced across *all* nodes.
    Phase 3  all-gather the finished shards back across the fast tier, so every
             chip holds the full, globally-reduced tensor.

    inter_size=1 (single instance) makes phase 2 free, so the op degenerates to a
    plain intra-node all-reduce (reduce-scatter + all-gather), which is correct.
    """
    shard_bytes = tensor_bytes / intra_size

    rs = ring.reduce_scatter(intra_size, tensor_bytes, intra_alpha, intra_bw)
    inter = ring.all_reduce(inter_size, shard_bytes, inter_alpha, inter_bw)
    ag = ring.all_gather(intra_size, tensor_bytes, intra_alpha, intra_bw)

    total_time_s = rs.total_time_s + inter.total_time_s + ag.total_time_s

    # The scarce resource is EFA, so we report what actually crossed it (phase 2 only).
    bytes_per_link = inter.bytes_per_link

    # EFA utilisation over the whole op. Low on purpose: the two NeuronLink phases
    # never touch EFA, and that low number is itself the signal the hierarchy did its
    # job, the op isn't inter-node-bound.
    util = bandwidth_util(bytes_per_link, inter_bw, total_time_s) if total_time_s > 0 else 0.0

    return CostReport(total_time_s=total_time_s, bytes_per_link=bytes_per_link,
                      bandwidth_util=util, steps=rs.steps + inter.steps + ag.steps)
