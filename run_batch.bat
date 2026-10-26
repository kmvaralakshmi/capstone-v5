@echo off
cd /d "%~dp0"
REM All 50 companies, bundled data, no LLM
python run_agents.py --offline
pause
