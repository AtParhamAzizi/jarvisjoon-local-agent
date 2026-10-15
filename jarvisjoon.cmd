@echo off
setlocal
pushd "%~dp0" || exit /b 1

if not exist ".venv\Scripts\python.exe" (
    echo JarvisJoon needs the local Python environment. 1>&2
    echo From the project folder, run: 1>&2
    echo   python -m venv .venv 1>&2
    echo   .venv\Scripts\python.exe -m pip install -r CLI\requirements.txt 1>&2
    popd
    exit /b 1
)

".venv\Scripts\python.exe" -m Terminal %*
set "agent_exit_code=%errorlevel%"
popd
endlocal & exit /b %agent_exit_code%
