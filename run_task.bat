@echo off
setlocal EnableExtensions

set "ROOT=C:\btc_bot"
set "PY=%ROOT%\.venv\Scripts\python.exe"
set "LOGDIR=%ROOT%\logs"

set "TASK=%~1"
set "ENVFILE=%~2"
set "LOG=%~3"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

REM --- debug vždy, aby sme vedeli čo prišlo ---
>> "%LOGDIR%\dbg_args.log" echo [%DATE% %TIME%] TASK=[%TASK%] ENVFILE=[%ENVFILE%] LOG=[%LOG%]

REM --- tvrdé kontroly s jasným logom ---
if "%TASK%"=="" (
  >> "%LOGDIR%\task_error.log" echo [%DATE% %TIME%] ERROR: missing TASK
  exit /b 2
)

if not exist "%PY%" (
  >> "%LOGDIR%\task_error.log" echo [%DATE% %TIME%] ERROR: python not found: %PY%
  exit /b 2
)

if not exist "%ROOT%\%TASK%" (
  >> "%LOGDIR%\task_error.log" echo [%DATE% %TIME%] ERROR: task file not found: %ROOT%\%TASK%
  exit /b 2
)

if "%LOG%"=="" set "LOG=%LOGDIR%\task_%TASK%.log"

REM --- vytvor log hneď teraz (aby sme ho vždy mali) ---
type nul >> "%LOG%"

cd /d "%ROOT%"

if not "%ENVFILE%"=="" (
  set "ENV_FILE=%ENVFILE%"
)

>> "%LOG%" echo.
>> "%LOG%" echo ===============================================
>> "%LOG%" echo START %DATE% %TIME%  TASK=%TASK%  ENVFILE=%ENVFILE%
>> "%LOG%" echo CWD=%CD%
>> "%LOG%" echo PY=%PY%
>> "%LOG%" echo --- RUN ---

"%PY%" "%ROOT%\%TASK%" >> "%LOG%" 2>&1
set "EC=%ERRORLEVEL%"

>> "%LOG%" echo --- END RUN ---
>> "%LOG%" echo END   %DATE% %TIME%  exit_code=%EC%

exit /b %EC%