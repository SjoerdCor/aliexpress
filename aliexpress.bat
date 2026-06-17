@echo off
REM Switch to this script's own folder (%~dp0 = the .bat's drive+path) so that the
REM relative paths below resolve and .env is found, no matter where it was launched from.
cd /d "%~dp0"
REM Force production mode for end users: no debugger/reloader. load_dotenv won't override
REM an already-set variable, so SECRET_KEY still comes from .env.
set FLASK_ENV=production
start "" ".venv\Scripts\pythonw.exe" app.py
