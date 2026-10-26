"""Deprecated entry point. The one-line 'autonomous agents' layer
(ai_agents/agents.py) could not even be imported in v5 (it referenced a class that
no longer exists, and the error was swallowed). It is replaced by esg_agents/.
This wrapper forwards to run_agents.py so old batch files keep working."""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
runpy.run_path(str(Path(__file__).resolve().parent / "run_agents.py"), run_name="__main__")
