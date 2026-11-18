@echo off
REM Review 1: serve this folder and open the walkthrough page (maps, plots, model comparison).
cd /d %~dp0
python src\make_review_site.py
start "" http://localhost:5600/review1.html
python -m http.server 5600
