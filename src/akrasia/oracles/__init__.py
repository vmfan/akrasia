"""The two oracles (ground-truth side of the DESIGN.md §2 firewall).

Oracle A (:mod:`akrasia.oracles.oracle_a`) scores actions against the agent's
*stated* belief; Oracle B (:mod:`akrasia.oracles.oracle_b`) solves the true
model. Both are exact (``fractions.Fraction``), never Monte Carlo.
"""
