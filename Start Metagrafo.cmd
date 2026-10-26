@echo off
setlocal
cd /d "%~dp0"
title Metagrafo
echo Metagrafo
echo.

set "ROOT=%CD%"
set "PYDIR=%ROOT%\runtime\python"
set "PY=%PYDIR%\python.exe"
set "PYW=%PYDIR%\pythonw.exe"
set "REQS=%ROOT%\requirements-gemini.txt"
set "STAMP=%ROOT%\runtime\installed-requirements.txt"
rem tar and timeout come from System32, since Git's GNU copies on PATH cannot unzip or take /t.

rem A running copy holds its packages open, so only show it.
curl -sf -o NUL --max-time 2 http://127.0.0.1:8000/health
if not errorlevel 1 goto open

rem The stamp is written only after pip succeeds, so a new list or a broken install runs pip again.
if not exist "%PYW%" goto install
fc /b "%REQS%" "%STAMP%" >nul 2>&1
if errorlevel 1 goto install
goto launch

:install
if not exist "%PY%" echo First start installs Python and Metagrafo.
echo This needs the internet and can take a few minutes.
echo Leave this window open.
echo.
if exist "%PY%" goto pip
echo Downloading Python...
if not exist "%ROOT%\runtime" mkdir "%ROOT%\runtime"
if not exist "%PYDIR%" mkdir "%PYDIR%"
curl -L --fail --retry 3 -o "%TEMP%\metagrafo-python-embed.zip" https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip
if errorlevel 1 goto failed
"%SystemRoot%\System32\tar.exe" -xf "%TEMP%\metagrafo-python-embed.zip" -C "%PYDIR%"
if errorlevel 1 goto failed

:pip
"%PY%" "%ROOT%\scripts\prepare_embed_python.py"
if errorlevel 1 goto failed
if exist "%PYDIR%\Lib\site-packages\pip\__init__.py" goto packages
echo Installing pip...
curl -L --fail --retry 3 -o "%TEMP%\metagrafo-get-pip.py" https://bootstrap.pypa.io/get-pip.py
if errorlevel 1 goto failed
"%PY%" "%TEMP%\metagrafo-get-pip.py" --no-warn-script-location
if errorlevel 1 goto failed

:packages
echo Installing Metagrafo...
"%PY%" -m pip install --disable-pip-version-check --no-warn-script-location -r "%REQS%"
if errorlevel 1 goto stale
copy /y "%REQS%" "%STAMP%" >nul
goto launch

rem An offline booth still starts the copy it already has; the next start tries the update again.
:stale
if not exist "%PYDIR%\Lib\site-packages\fastapi\__init__.py" goto failed
echo.
echo Could not update Metagrafo. Starting the copy already installed.
echo.

:launch
echo Opening Metagrafo...
start "" /D "%ROOT%" "%PYW%" -m main
set /a TRIES=0
:wait
set /a TRIES+=1
if %TRIES% GTR 60 goto notstarted
curl -sf -o NUL --max-time 2 http://127.0.0.1:8000/health
if errorlevel 1 (
  ping -n 2 127.0.0.1 >nul
  goto wait
)

:open
start "" http://127.0.0.1:8000/control
echo.
echo Booth control is open in the browser.
echo Paste the Gemini key there and click Save key.
echo Then pick the sound input under Audio input.
echo You can close this window. Metagrafo keeps running.
"%SystemRoot%\System32\timeout.exe" /t 8 >nul
exit /b 0

:failed
echo.
echo Metagrafo could not install.
echo Check that this computer is online, then double-click this file again.
echo.
pause
exit /b 1

:notstarted
echo.
echo Metagrafo did not start.
echo The reason is at the end of metagrafo.log in this folder.
if exist "%ROOT%\metagrafo.log" start "" notepad "%ROOT%\metagrafo.log"
echo.
pause
exit /b 1
