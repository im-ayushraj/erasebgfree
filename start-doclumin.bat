@echo off
REM Start Flask backend
start cmd /k py -3.13  app.py

REM Start Cloudflared tunnel
start cmd /k cloudflared tunnel run doclumin-tunnel