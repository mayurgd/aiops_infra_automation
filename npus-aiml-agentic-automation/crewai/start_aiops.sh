#!/bin/bash

# AIOps Agentic Automation - Startup Script (macOS/Linux with venv)
# This script launches MCP server, backend and frontend servers using Python virtual environment

echo "========================================"
echo "AIOps Agentic Automation - Launcher"
echo "========================================"
echo

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
VENV_DIR="$PROJECT_ROOT/venv"

# Function to cleanup background processes on exit
cleanup() {
    echo
    echo "[INFO] Cleaning up background processes..."
    if [[ -n $MCP_PID ]]; then
        kill $MCP_PID 2>/dev/null
        echo "[INFO] MCP server stopped."
    fi
    if [[ -n $BACKEND_PID ]]; then
        kill $BACKEND_PID 2>/dev/null
        echo "[INFO] Backend server stopped."
    fi
    if [[ -n $FRONTEND_PID ]]; then
        kill $FRONTEND_PID 2>/dev/null
        echo "[INFO] Frontend server stopped."
    fi
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Check Python version
echo "[INFO] Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 is not installed or not in PATH."
    echo "Please install Python 3.11 using Homebrew:"
    echo "  brew install python@3.11"
    echo
    exit 1
fi

# Get Python version and check if it's 3.11
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "[INFO] Found Python version: $PYTHON_VERSION"

# Extract major and minor version (e.g., 3.11 from 3.11.x)
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [[ "$PYTHON_MAJOR" != "3" ]]; then
    echo "[ERROR] Python 3.x is required, but found Python $PYTHON_VERSION"
    echo "Please install Python 3.11"
    echo
    exit 1
fi

if [[ "$PYTHON_MINOR" != "11" ]]; then
    echo "[WARNING] Python 3.11 is recommended, but found Python $PYTHON_VERSION"
    echo "The application may not work correctly with this version."
    echo
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "[INFO] Installation cancelled."
        exit 1
    fi
    echo
else
    echo "[OK] Python 3.11 detected"
    echo
fi

# Check if .env file exists in root or backend directory
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    echo "[OK] .env file found in root directory"
    echo
elif [[ -f "$PROJECT_ROOT/backend/.env" ]]; then
    echo "[OK] .env file found in backend directory"
    echo
else
    echo "[ERROR] .env file not found."
    echo "Please create a .env file in the root directory or backend directory."
    echo "You can copy from .env.example if available."
    echo
    exit 1
fi

# Check if backend directory exists
if [[ ! -d "$PROJECT_ROOT/backend" ]]; then
    echo "[ERROR] Backend directory not found at: $PROJECT_ROOT/backend"
    echo "Please ensure the backend folder exists in the project root."
    echo
    exit 1
fi

# Check if frontend directory exists
if [[ ! -d "$PROJECT_ROOT/frontend" ]]; then
    echo "[ERROR] Frontend directory not found at: $PROJECT_ROOT/frontend"
    echo "Please ensure the frontend folder exists in the project root."
    echo
    exit 1
fi

# Check if MCP server script exists
if [[ ! -f "$PROJECT_ROOT/backend/servers/aiops_automation_server.py" ]]; then
    echo "[ERROR] MCP server script not found at: $PROJECT_ROOT/backend/servers/aiops_automation_server.py"
    echo "Please ensure the MCP server script exists."
    echo
    exit 1
fi

echo "[OK] All required directories and scripts found"
echo

# Check if requirements.txt exists
if [[ ! -f "$PROJECT_ROOT/requirements.txt" ]]; then
    echo "[WARNING] requirements.txt not found in project root."
    echo "Will attempt to use backend/requirements.txt if available."
    echo
    REQUIREMENTS_FILE="$PROJECT_ROOT/backend/requirements.txt"
    if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
        echo "[ERROR] No requirements.txt found in project root or backend directory."
        echo "Please ensure requirements.txt exists."
        echo
        exit 1
    fi
else
    REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"
    echo "[OK] requirements.txt found"
    echo
fi

# Check if virtual environment exists
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "[WARNING] Virtual environment not found."
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    if [[ $? -ne 0 ]]; then
        echo "[ERROR] Failed to create virtual environment."
        echo "Please ensure Python 3.11 is installed and in PATH."
        exit 1
    fi
    echo "[OK] Virtual environment created!"
    echo
    
    echo "[INFO] Installing dependencies from requirements.txt..."
    source "$VENV_DIR/bin/activate"
    python -m pip install --upgrade pip
    pip install -r "$REQUIREMENTS_FILE"
    if [[ $? -ne 0 ]]; then
        echo "[ERROR] Failed to install dependencies."
        exit 1
    fi
    deactivate
    echo "[OK] Dependencies installed successfully!"
    echo
else
    echo "[OK] Virtual environment found"
    echo
fi

echo "========================================"
echo "Step 1: Launching MCP Server"
echo "========================================"

# Launch MCP server in background
(
    cd "$PROJECT_ROOT/backend"
    source "$VENV_DIR/bin/activate"
    echo "[INFO] Starting MCP Server..."
    echo "[INFO] MCP Server Location: $PROJECT_ROOT/backend/servers/aiops_automation_server.py"
    echo
    python -m servers.aiops_automation_server
) &
MCP_PID=$!

# Wait for MCP server to initialize
echo "[INFO] Waiting for MCP server to initialize..."
sleep 5
echo "[OK] MCP server should be running now!"
echo

echo "========================================"
echo "Step 2: Launching Backend Server"
echo "========================================"

# Launch backend in background
(
    cd "$PROJECT_ROOT/backend"
    source "$VENV_DIR/bin/activate"
    echo "[INFO] Starting Backend Server..."
    echo "[INFO] Backend will be available at: http://127.0.0.1:8070"
    echo
    python app.py
) &
BACKEND_PID=$!

# Wait and check for backend to be fully ready
echo "[INFO] Waiting for backend server to initialize..."
for i in {1..30}; do
    sleep 2
    if curl -s http://127.0.0.1:8070 > /dev/null 2>&1; then
        break
    fi
    echo "[INFO] Backend not ready yet, waiting..."
    if [[ $i -eq 30 ]]; then
        echo "[ERROR] Backend server failed to start within timeout."
        exit 1
    fi
done

echo "[OK] Backend server is ready!"
echo

echo "========================================"
echo "Step 3: Launching Frontend Server"
echo "========================================"

# Launch frontend in background
(
    cd "$PROJECT_ROOT/frontend"
    source "$VENV_DIR/bin/activate"
    echo "[INFO] Starting Frontend Server..."
    echo "[INFO] Frontend will be available at: http://localhost:8080"
    echo "[INFO] Press Ctrl+C to stop the server"
    echo
    python -m http.server 8080
) &
FRONTEND_PID=$!

# Wait for frontend to start
echo "[INFO] Waiting for frontend server to initialize..."
sleep 3
echo

echo "========================================"
echo "All Steps Completed Successfully!"
echo "========================================"
echo
echo "Startup Sequence:"
echo "1. MCP Server          - Running on http://127.0.0.1:8060"
echo "2. Backend Server      - Running on http://127.0.0.1:8070"
echo "3. Frontend Server     - Running on http://localhost:8080"
echo
echo "All servers are running in the background."
echo
echo "Close this terminal or press Ctrl+C to stop all servers."
echo

# Open web page in default browser
echo "[INFO] Opening frontend in browser..."
if command -v open &> /dev/null; then
    # macOS
    open "http://localhost:8080"
    echo "[OK] Browser window opened!"
elif command -v xdg-open &> /dev/null; then
    # Linux
    xdg-open "http://localhost:8080"
    echo "[OK] Browser window opened!"
else
    echo "[INFO] Please open http://localhost:8080 in your browser manually."
fi
echo

# Keep the script running and wait for user to stop
echo "[INFO] All servers are running. Press Ctrl+C to stop all servers."
wait