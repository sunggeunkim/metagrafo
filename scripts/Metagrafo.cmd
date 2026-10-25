@echo off
title Metagrafo
rem Sign-in launcher. Capture needs this user's audio session, so this
rem lives in the Startup folder rather than running as SYSTEM at boot.
cd /d "C:\Users\thewaychurch\metagrafo"
echo Starting Metagrafo in %CD%
rem Give the desktop and VB-Cable a moment after sign-in.
ping -n 16 127.0.0.1 >nul
uv run python -m main
echo.
echo Metagrafo stopped (exit %ERRORLEVEL%).
pause
