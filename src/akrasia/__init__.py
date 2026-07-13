"""akrasia — a diagnostic instrument for the reasoning-to-execution gap.

This package's Week-1 surface is the *ground-truth-side* measurement core:
``sim`` (the environment) and ``oracles`` (Oracle A + Oracle B). Both may see
the true simulator parameters. The agent-under-test harness (``agent/``) is
NOT part of this package yet and, per the DESIGN.md §2 firewall, must never
import :mod:`akrasia.sim.params`.
"""
