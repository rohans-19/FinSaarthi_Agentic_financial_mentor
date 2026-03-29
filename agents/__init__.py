"""
FinSaarthi — Agents Package
=============================
LangGraph-powered multi-agent system for financial planning.
Each agent handles one module and writes to its own state namespace.

Imports are lazy for orchestrator/build_graph to avoid requiring
langgraph at import time (allows testing individual agents without it).
"""

from typing import Any, Dict, List, Optional
from agents.portfolio_agent import PortfolioAgent
from agents.fire_agent import FIREAgent
from agents.tax_agent import TaxAgent
from agents.couple_agent import CoupleAgent

class BaseFinSaarthiAgent:
    def __init__(self, llm: Any = None, kb: Any = None):
        self.llm = llm
        self.kb = kb

def build_graph(*args, **kwargs):
    """Lazy import to avoid requiring langgraph at package import time."""
    from agents.orchestrator import build_graph as _build_graph
    return _build_graph(*args, **kwargs)

def run_module(*args, **kwargs):
    """Lazy import to avoid requiring langgraph at package import time."""
    from agents.orchestrator import run_module as _run_module
    return _run_module(*args, **kwargs)

__all__ = [
    "BaseFinSaarthiAgent",
    "PortfolioAgent",
    "FIREAgent",
    "TaxAgent",
    "CoupleAgent",
    "build_graph",
    "run_module",
]
