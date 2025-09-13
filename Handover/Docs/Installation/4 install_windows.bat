@echo off
setlocal

REM --- Check if Python is installed ---
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Python is not installed. Downloading and installing...

    REM --- Download the official Python installer silently ---
    powershell -Command "Invoke-WebRequest https://www.python.org/ftp/python/3.12.3/python-3.12.3-amd64.exe -OutFile python_installer.exe"

    REM --- Install Python silently with pip and add to PATH ---
    python_installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_pip=1

    IF %ERRORLEVEL% NEQ 0 (
        echo Python installation failed. Exiting.
        exit /b 1
    )
    
    echo Python installed successfully.
    del python_installer.exe
)

REM --- Run setup.py ---
echo Running setup script...
python setup.py

pause
