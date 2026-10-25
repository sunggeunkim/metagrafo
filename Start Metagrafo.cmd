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

if not exist "%PYW%" goto install
if not exist "%PYDIR%\Lib\site-packages\fastapi\__init__.py" goto install
goto launch

:install
echo First start installs Python and Metagrafo.
echo This needs the internet and can take a few minutes.
echo Leave this window open.
echo.
if not exist "%PY%" (
  echo Downloading Python...
  if not exist "%ROOT%\runtime" mkdir "%ROOT%\runtime"
  if not exist "%PYDIR%" mkdir "%PYDIR%"
  curl -L --fail --retry 3 -o "%TEMP%\metagrafo-python-embed.zip" https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip
  if errorlevel 1 goto failed
  tar -xf "%TEMP%\metagrafo-python-embed.zip" -C "%PYDIR%"
  if errorlevel 1 goto failed
  "%PY%" "%ROOT%\scripts\prepare_embed_python.py"
  if errorlevel 1 goto failed
  echo Installing pip...
  curl -L --fail --retry 3 -o "%TEMP%\metagrafo-get-pip.py" https://bootstrap.pypa.io/get-pip.py
  if errorlevel 1 goto failed
  "%PY%" "%TEMP%\metagrafo-get-pip.py" --no-warn-script-location
  if errorlevel 1 goto failed
)
echo Installing Metagrafo...
"%PY%" -m pip install --disable-pip-version-check -r "%ROOT%\requirements-gemini.txt"
if errorlevel 1 goto failed

:launch
curl -sf -o NUL --max-time 2 http://127.0.0.1:8000/health
if not errorlevel 1 goto open
echo Opening Metagrafo...
start "" /D "%ROOT%" "%PYW%" -m main
set /a TRIES=0
:wait
set /a TRIES+=1
if %TRIES% GTR 60 goto failed
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
echo You can close this window. Metagrafo keeps running.
timeout /t 8 >nul
exit /b 0

:failed
echo.
echo Metagrafo did not start.
echo If this is the first start, check that this computer is online.
echo.
pause
exit /b 1
