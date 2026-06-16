@echo off
REM Switch to this script's own folder (%~dp0 = the .bat's drive+path) so that the
REM relative paths below resolve and .env is found, no matter where it was launched from.
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" app.py
