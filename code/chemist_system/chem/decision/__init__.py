"""Finite-pool reference implementation for the Chapter 5 research design.

This package does not call an LLM, operate equipment, or certify synthesis.
"""

from .domain_guard import Decision, NoAvailableCandidates, select_candidate

__all__ = ["Decision", "NoAvailableCandidates", "select_candidate"]
