"""Multi-agent ESG risk analysis system (v5.1).

Each agent lives in its own file, subclasses framework.BaseAgent, owns its tools,
completion criteria and planning policy, and talks to the others through the
shared Blackboard.
"""
from .framework import BaseAgent, Blackboard, LLMBrain, Tool  # noqa: F401
