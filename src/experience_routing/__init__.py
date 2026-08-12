"""Experience-routing MVP: receiver-aware donor->receiver experience routing
in robot policy populations.

See PLAN.md for the full research specification. This package implements the
scientific core (segmentation, experience grouping, competence estimation,
deficit/supply, greedy + unbalanced-OT routing) plus a lightweight synthetic
environment and a compact SAC backbone so the whole online loop runs on CPU.
"""

__version__ = "0.1.0"
