@echo off
REM AIOps Agentic Automation - Startup Script (Batch)
REM This script launches both backend and frontend servers
echo ========================================
echo AIOps Agentic Automation - Launcher
echo ========================================
echo.

REM Get the directory where the script is located
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%"

REM Check if .env file exists
if not exist "%PROJECT_ROOT%crewai_implementation\.env" (
    echo [ERROR] .env file not found in crewai_implementation directory.
    echo Please copy .env.example to .env and configure it first.
    echo.
    pause
    exit /b 1
)
echo [OK] .env file found
echo.

REM Check if conda environment exists
call conda env list | findstr "agentic_poc" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] 'agentic_poc' environment not found.
    echo Creating environment...
    call conda create -n agentic_poc python=3.12 -y
    call conda run -n agentic_poc pip install crewai crewai-tools[mcp] databricks-sdk fastapi uvicorn python-dotenv
    echo [OK] Environment created successfully!
    echo.
)

echo [INFO] Launching Backend Server...
REM Launch backend in a new window
start "AIOps Backend Server" cmd /k "cd /d "%PROJECT_ROOT%crewai_implementation\backend" && call conda activate agentic_poc && echo [INFO] Starting Backend Server... && echo [INFO] Backend will be available at: http://127.0.0.1:8000 && echo. && python app.py"

REM Wait for backend to start (increased wait time)
echo [INFO] Waiting for backend server to initialize...
timeout /t 8 /nobreak >nul

echo [INFO] Launching Frontend Server...
REM Launch frontend in a new window
start "AIOps Frontend Server" cmd /k "cd /d "%PROJECT_ROOT%crewai_implementation\frontend" && echo [INFO] Starting Frontend Server... && echo [INFO] Frontend will be available at: http://localhost:8080 && echo [INFO] Press Ctrl+C to stop the server && echo. && python -m http.server 8080"

REM Wait for frontend to start
echo [INFO] Waiting for frontend server to initialize...
timeout /t 3 /nobreak >nul

echo.
echo ========================================
echo Servers Started Successfully!
echo ========================================
echo Backend:  http://127.0.0.1:8000
echo Frontend: http://localhost:8080
echo.
echo Two command prompt windows have been opened.
echo Close those windows or press Ctrl+C in them to stop the servers.
echo.

REM Open web pages in default browser
echo [INFO] Opening web pages in browser...
start "" "http://127.0.0.1:8000/docs"
timeout /t 2 /nobreak >nul
start "" "http://localhost:8080"

echo.
echo [OK] Browser windows opened!
echo.
pause