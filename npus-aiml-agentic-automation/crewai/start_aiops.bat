@echo off
REM AIOps Agentic Automation - Startup Script (Batch with venv)
REM This script launches MCP server, backend and frontend servers using Python virtual environment
echo ========================================
echo AIOps Agentic Automation - Launcher
echo ========================================
echo.

REM Get the directory where the script is located
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%"
set "VENV_DIR=%PROJECT_ROOT%venv"

REM Check Python version
echo [INFO] Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.11 and add it to your PATH.
    echo.
    pause
    exit /b 1
)

REM Get Python version and check if it's 3.11
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [INFO] Found Python version: %PYTHON_VERSION%

REM Extract major and minor version (e.g., 3.11 from 3.11.x)
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

if not "%PYTHON_MAJOR%"=="3" (
    echo [ERROR] Python 3.x is required, but found Python %PYTHON_VERSION%
    echo Please install Python 3.11
    echo.
    pause
    exit /b 1
)

if not "%PYTHON_MINOR%"=="11" (
    echo [WARNING] Python 3.11 is recommended, but found Python %PYTHON_VERSION%
    echo The application may not work correctly with this version.
    echo.
    choice /C YN /M "Do you want to continue anyway?"
    if errorlevel 2 (
        echo [INFO] Installation cancelled.
        pause
        exit /b 1
    )
    echo.
) else (
    echo [OK] Python 3.11 detected
    echo.
)

REM Check if .env file exists in root or backend directory
if exist "%PROJECT_ROOT%.env" (
    echo [OK] .env file found in root directory
    echo.
) else if exist "%PROJECT_ROOT%backend\.env" (
    echo [OK] .env file found in backend directory
    echo.
) else (
    echo [ERROR] .env file not found.
    echo Please create a .env file in the root directory or backend directory.
    echo You can copy from .env.example if available.
    echo.
    pause
    exit /b 1
)

REM Check if backend directory exists
if not exist "%PROJECT_ROOT%backend" (
    echo [ERROR] Backend directory not found at: %PROJECT_ROOT%backend
    echo Please ensure the backend folder exists in the project root.
    echo.
    pause
    exit /b 1
)

REM Check if frontend directory exists
if not exist "%PROJECT_ROOT%frontend" (
    echo [ERROR] Frontend directory not found at: %PROJECT_ROOT%frontend
    echo Please ensure the frontend folder exists in the project root.
    echo.
    pause
    exit /b 1
)

REM Check if wiki_downloader script exists
if not exist "%PROJECT_ROOT%backend\docs\wiki_downloader.py" (
    echo [ERROR] Wiki downloader script not found at: %PROJECT_ROOT%backend\docs\wiki_downloader.py
    echo Please ensure the wiki downloader script exists.
    echo.
    pause
    exit /b 1
)

REM Check if knowledge_base_utility script exists
if not exist "%PROJECT_ROOT%backend\servers\knowledge_base_utility.py" (
    echo [ERROR] Knowledge base utility script not found at: %PROJECT_ROOT%backend\servers\knowledge_base_utility.py
    echo Please ensure the knowledge base utility script exists.
    echo.
    pause
    exit /b 1
)

REM Check if MCP server script exists
if not exist "%PROJECT_ROOT%backend\servers\aiops_automation_server.py" (
    echo [ERROR] MCP server script not found at: %PROJECT_ROOT%backend\servers\aiops_automation_server.py
    echo Please ensure the MCP server script exists.
    echo.
    pause
    exit /b 1
)

echo [OK] All required directories and scripts found
echo.

REM Check if requirements.txt exists
if not exist "%PROJECT_ROOT%requirements.txt" (
    echo [WARNING] requirements.txt not found in project root.
    echo Will attempt to use backend/requirements.txt if available.
    echo.
    set "REQUIREMENTS_FILE=%PROJECT_ROOT%backend\requirements.txt"
    if not exist "%REQUIREMENTS_FILE%" (
        echo [ERROR] No requirements.txt found in project root or backend directory.
        echo Please ensure requirements.txt exists.
        echo.
        pause
        exit /b 1
    )
) else (
    set "REQUIREMENTS_FILE=%PROJECT_ROOT%requirements.txt"
    echo [OK] requirements.txt found
    echo.
)

REM Check if virtual environment exists
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [WARNING] Virtual environment not found.
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        echo Please ensure Python 3.11 is installed and in PATH.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created!
    echo.
    
    echo [INFO] Installing dependencies from requirements.txt...
    call "%VENV_DIR%\Scripts\activate.bat"
    python -m pip install --upgrade pip
    pip install -r "%REQUIREMENTS_FILE%"
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
    call deactivate
    echo [OK] Dependencies installed successfully!
    echo.
) else (
    echo [OK] Virtual environment found
    echo.
)

echo ========================================
echo Step 1: Running Wiki Downloader
echo ========================================
call "%VENV_DIR%\Scripts\activate.bat"
cd /d "%PROJECT_ROOT%backend"
echo [INFO] Executing wiki_downloader.py...
python -m docs.wiki_downloader
if errorlevel 1 (
    echo [ERROR] Wiki downloader failed.
    call deactivate
    pause
    exit /b 1
)
echo [OK] Wiki downloader completed successfully!
echo.

echo ========================================
echo Step 2: Running Knowledge Base Utility
echo ========================================
echo [INFO] Executing knowledge_base_utility.py...
python -m servers.knowledge_base_utility
if errorlevel 1 (
    echo [ERROR] Knowledge base utility failed.
    call deactivate
    pause
    exit /b 1
)
echo [OK] Knowledge base utility completed successfully!
call deactivate
echo.

echo ========================================
echo Step 3: Launching MCP Server
echo ========================================
REM Launch MCP server in a new window
start "AIOps MCP Server" cmd /k "cd /d "%PROJECT_ROOT%backend" && call "%VENV_DIR%\Scripts\activate.bat" && echo [INFO] Starting MCP Server... && echo [INFO] MCP Server Location: %PROJECT_ROOT%backend\servers\aiops_automation_server.py && echo. && python -m servers.aiops_automation_server"

REM Wait for MCP server to initialize
echo [INFO] Waiting for MCP server to initialize...
timeout /t 5 /nobreak >nul
echo [OK] MCP server should be running now!
echo.

echo ========================================
echo Step 4: Launching Backend Server
echo ========================================
REM Launch backend in a new window
start "AIOps Backend Server" cmd /k "cd /d "%PROJECT_ROOT%backend" && call "%VENV_DIR%\Scripts\activate.bat" && echo [INFO] Starting Backend Server... && echo [INFO] Backend will be available at: http://127.0.0.1:8070 && echo. && python app.py"

REM Wait and check for backend to be fully ready
echo [INFO] Waiting for backend server to initialize...
:WAIT_BACKEND
timeout /t 2 /nobreak >nul
curl -s http://127.0.0.1:8070 >nul 2>&1
if errorlevel 1 (
    echo [INFO] Backend not ready yet, waiting...
    goto WAIT_BACKEND
)

echo [OK] Backend server is ready!
echo.

echo ========================================
echo Step 5: Launching Frontend Server
echo ========================================
REM Launch frontend in a new window
start "AIOps Frontend Server" cmd /k "cd /d "%PROJECT_ROOT%frontend" && call "%VENV_DIR%\Scripts\activate.bat" && echo [INFO] Starting Frontend Server... && echo [INFO] Frontend will be available at: http://localhost:8080 && echo [INFO] Press Ctrl+C to stop the server && echo. && python -m http.server 8080"

REM Wait for frontend to start
echo [INFO] Waiting for frontend server to initialize...
timeout /t 3 /nobreak >nul
echo.

echo ========================================
echo All Steps Completed Successfully!
echo ========================================
echo.
echo Startup Sequence:
echo 1. Wiki Downloader      - Completed
echo 2. Knowledge Base Setup - Completed
echo 3. MCP Server          - Running on http://127.0.0.1:8060
echo 4. Backend Server      - Running on http://127.0.0.1:8070
echo 5. Frontend Server     - Running on http://localhost:8080
echo.
echo Three command prompt windows have been opened:
echo 1. MCP Server
echo 2. Backend Server
echo 3. Frontend Server
echo.
echo Close those windows or press Ctrl+C in them to stop the servers.
echo.

REM Open web page in default browser
echo [INFO] Opening frontend in browser...
start "" "http://localhost:8080"
echo.

echo [OK] Browser window opened!
echo.
pause