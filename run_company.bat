@echo off
cd /d "%~dp0"
set /p CO=Company name or NSE symbol: 
python run_agents.py --company "%CO%" --dynamic
pause
