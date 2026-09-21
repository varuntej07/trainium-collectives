# trainium-collectives

An explainable cost model for collective communication on Trainium-class, two-tier networks.

The question I care about is simple:

> When does a flat collective stop matching the machine, and when does a topology-aware collective win?

On Trn1, the links inside a box and the links between boxes are not interchangeable. NeuronLink is the fast intra-node tier; EFA is the scarcer inter-node tier. A flat ring ignores that distinction. A hierarchical all-reduce uses it: reduce-scatter inside each box, all-reduce only one shard across boxes, then all-gather inside each box.

The point is not to build another simulator for its own sake. The point is to predict the winning algorithm, measure it on real hardware, and explain every gap between the two.

## The model

Every communication step uses the alpha-beta model:

```text
step time = alpha + bytes / bandwidth
```

- `alpha` is the fixed latency paid per message.
- `bytes / bandwidth` is the time spent moving the payload.
- Algorithms compose those steps into a `CostReport`: total time, bytes on the busiest link, bandwidth utilization, and critical-path step count.

That makes the algorithm tradeoff visible instead of hiding it behind a benchmark number:

| Algorithm | Critical-path steps | Data movement | What it is good at |
| --- | ---: | --- | --- |
| Ring | `2(N - 1)` | Small `S/N` chunks | Large, bandwidth-bound tensors |
| Single tree | `2 ceil(log2 N)` | Full tensor per level | Small, latency-bound tensors |
| Hierarchical ring | Intra-node ring + inter-node shard ring | Only `S/intra_size` crosses the slow tier | Two-level networks such as NeuronLink + EFA |
| Recursive doubling | Planned | Planned | Small-message comparison |

## Current finding

The first result surprised me: with the current Trn1 assumptions, hierarchical ring does not merely cross over at a large message size. At eight instances (128 ranks), it beats a flat ring across the entire 4 KiB to 1 GiB sweep.

Why? Latency. The flat ring serializes `2(N - 1) = 254` EFA-gated steps. The hierarchical path pays only 14 inter-node ring steps and keeps the full-tensor traffic on NeuronLink.

That is a model result, not measured truth yet. The NeuronLink and EFA latency assumptions still need calibration, and the Trainium sweep is not implemented. I would rather label that boundary clearly than pretend the chart is validation.

## What works today

| Area | Status |
| --- | --- |
| Alpha-beta primitives | Implemented |
| Ring reduce-scatter, all-gather, and all-reduce | Implemented |
| Hierarchical ring all-reduce | Implemented |
| Single-tree reduce, broadcast, and all-reduce | Implemented |
| Recursive doubling | Not started |
| Trn1/Trn2 topology module | Not added yet |
| Trainium / Neuron Explorer measurements | Not started |
| Host-side C++ all-reduce and pybind11 module | Not started |
| Property tests for the claims above | Implemented |
| Model-vs-measurement validation | Not started |

## Quick start

Requires Python 3.10 or newer.

```bash
git clone https://github.com/varuntej07/trainium-collectives.git
cd trainium-collectives
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the package in editable mode:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Then call a cost model directly:

```python
from simulator.cost_model import hierarchical, ring, tree

tensor_bytes = 128 * 1024 * 1024

flat = ring.all_reduce(
    n_nodes=16,
    tensor_bytes=tensor_bytes,
    alpha=1e-6,
    bandwidth_bytes_s=100e9,
)

latency_first = tree.all_reduce(
    n_nodes=16,
    tensor_bytes=tensor_bytes,
    alpha=1e-6,
    bandwidth_bytes_s=100e9,
)

two_tier = hierarchical.all_reduce(
    intra_size=16,
    inter_size=8,
    tensor_bytes=tensor_bytes,
    intra_alpha=1e-6,
    intra_bw=768e9,
    inter_alpha=15e-6,
    inter_bw=100e9,
)

print(flat)
print(latency_first)
print(two_tier)
```

The hardware values above are deliberately plain inputs, not baked into the algorithms. The bandwidth figures come from the public Trn1 specifications; the latency values are estimates and calibration targets.

## Repository map

```text
src/simulator/cost_model/   Python analytical models
src/simulator/report.py     Shared CostReport returned by every model
tests/                      Property tests for the claims made above
pyproject.toml              Editable install, Python 3.10 or newer, no runtime deps
```

The docstrings are the lab notes. Each model file carries its own derivation and the limitation it is aware of, and `tests/` holds every claim this README makes to account.

## Validation plan

There are three sources I want to reconcile:

```text
analytical Python model <-> controlled host-side C++ all-reduce <-> real Trainium measurements
```

The project is done when those three agree closely enough to explain—and when they do not, the reason is written down. Until the measured Trainium data lands, treat this repository as a transparent analytical model and work in progress, not a performance claim.
