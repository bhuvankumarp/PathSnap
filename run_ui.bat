@echo off
REM Laptop 2: serve the evaluator UI at http://localhost:5500
cd /d %~dp0ui
start "" http://localhost:5500
python -m http.server 5500
