@echo off
REM Laptop 1: start the PathSnap API on all interfaces, port 8000.
REM Serves data\evaluator_test if it exists, otherwise data\student_test.
cd /d %~dp0
echo This laptop's IPv4 addresses (give one to the UI laptop):
ipconfig | findstr /i "IPv4"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
