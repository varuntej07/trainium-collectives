"""Per-algorithm cost models. Each returns a CostReport from plain (alpha, bandwidth) inputs.

Deliberately no re-exports here. hierarchical.py imports ring from this package,
so eagerly importing the submodules would make `import hierarchical` re-enter a
half-initialised package. CPython recovers from that, but it is not a thing to
ship. Importing the submodules by name works without any of it.
"""
