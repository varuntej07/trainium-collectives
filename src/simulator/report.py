"""The one record every cost model hands back.

Every algorithm in cost_model/ answers the same four questions, so they all return
the same record instead of a loose tuple or a dict whose keys quietly drift:

  total_time_s    how long the collective takes along its critical path
  bytes_per_link  bytes the busiest physical link has to carry
  bandwidth_util  bytes_per_link / (B * total_time_s), how much of the link we used
  steps           serialized communication steps on that critical path

Keeping the four together is the only reason ring, tree, and hierarchical are
comparable at all. The point of this project is not any single number but the
trade between them: a ring and a tree can report the same total_time_s for
completely different reasons, and you only see which is which by reading steps
(latency paid) against bytes_per_link (bandwidth paid).

Frozen, because a report is a decision already made. If you want a different
number, run a different model -> do not patch the record.

One caveat worth stating up front. Each model decides what "the busiest link"
means for its own topology, and says so in its own docstring. Ring counts the
chunks crossing one ring edge. Tree counts the root edge, which is a per-link
reading and deliberately not the steps*S that the critical path pays for.
Hierarchical reports only what crossed the slow inter-node tier, because that is
the scarce resource the whole algorithm exists to protect. So bytes_per_link is
comparable across models only once you have read those notes. total_time_s and
steps always are.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CostReport:
    """What one collective costs, under one set of (alpha, bandwidth) assumptions."""

    total_time_s: float
    bytes_per_link: float
    bandwidth_util: float
    steps: int

    def __str__(self) -> str:
        return (f"{self.total_time_s * 1e6:10.2f} us  "
                f"{self.bytes_per_link / 1024 ** 2:9.2f} MiB/link  "
                f"util {self.bandwidth_util:6.1%}  "
                f"{self.steps:4d} steps")
