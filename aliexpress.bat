@echo off
call conda activate aliexpress
start "" pythonw app.py
call conda deactivate
