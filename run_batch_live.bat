@echo off
cd /d "%~dp0"
REM All 50 companies with LIVE market (Yahoo) + news (GDELT) data. GDELT is rate-limited: expect ~5-10 min.
python run_agents.py --dynamic
pause
