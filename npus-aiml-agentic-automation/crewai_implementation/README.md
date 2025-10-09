# AIOps Agentic Automation - Setup Guide

## Overview
This guide will help you set up and run the AIOps Agentic Automation system, which provides an intelligent interface for automating GitHub repository creation, Databricks schema provisioning, and compute cluster setup.

---

## Prerequisites
- Anaconda or Miniconda installed
- GitHub account with access to `nestle-it` organization
- Databricks workspace access
- Azure Service Principal credentials

---

## Step 1: Environment Setup

### 1.1 Create Python Environment
Open Anaconda Prompt and run the following commands:

```bash
# Create a new conda environment with Python 3.12
conda create -n agentic_poc python=3.12

# Activate the environment
conda activate agentic_poc
```

### 1.2 Install Dependencies
```bash
pip install crewai crewai-tools[mcp] databricks-sdk fastapi uvicorn python-dotenv
```

---

## Step 2: GitHub Token Configuration

### 2.1 Generate Personal Access Token
1. Navigate to [GitHub Settings](https://github.com/settings/)
2. Click on **Developer settings** (left sidebar)
3. Select **Personal access tokens** → **Tokens (classic)**
4. Click **Generate new token** (classic)
5. Configure token settings:
   - **Note**: Add a descriptive name (e.g., "AIOps Automation")
   - **Expiration**: Select appropriate duration
   - **Scopes**: Check the following:
     - ✅ `repo` (selected by default)
     - ✅ `workflow`
6. Click **Generate token**
7. **Important**: Copy the token immediately (you won't see it again)

### 2.2 Authorize SSO
1. Refresh the tokens page
2. Locate your newly created token
3. Click **Configure SSO**
4. Click **Authorize** next to `nestle-it` organization
5. Complete the authorization process

---

## Step 3: Configuration File Setup

### 3.1 Locate Configuration Files
Navigate to the `crewai_implementation` directory:
```bash
cd crewai_implementation
```

### 3.2 Create Environment File
Copy the example environment file:
```bash
# On Windows
copy .env.example .env
```

### 3.3 Configure Environment Variables
Open the `.env` file and update the following values:

#### LLM Configuration
```bash
NESTLE_CLIENT_ID=CLIENT_ID
NESTLE_CLIENT_SECRET=CLIENT_SECRET
NESTLE_MODEL=gpt-4.1
```

#### GitHub Configuration
```bash
GITHUB_TOKEN=your-github-token-from-step-2
GITHUB_ORG=nestle-it
AUTOMATION_REPO=npus-aiml-utilities-create-repo
```

#### CrewAI Configuration
```bash
OTEL_SDK_DISABLED=true
CREWAI_DISABLE_TELEMETRY=true
CREWAI_DISABLE_TRACKING=true
SERVER_LOCATION=C:/path/to/your/crewai_implementation/backend/servers/aiops_automation_server.py
```

**Note**: Update `SERVER_LOCATION` with the absolute path to `aiops_automation_server.py` on your system.

**Example paths**:
- Windows: `C:/Users/YourName/Documents/project/crewai_implementation/backend/servers/aiops_automation_server.py`

#### Databricks Configuration
```bash
DATABRICKS_WORKSPACE_URL=https://adb-1125343200912494.14.azuredatabricks.net/
ARM_TENANT_ID=your-azure-tenant-id
ARM_CLIENT_ID=your-azure-client-id
ARM_CLIENT_SECRET=your-azure-client-secret
```

#### Terminal Mode
```bash
TERMINAL=false
```
- Set to `false` for web UI interaction (recommended)
- Set to `true` for terminal-based interaction

---

## Step 4: Running the Application

### 4.1 Start Backend Server
Navigate to the backend directory and start the FastAPI server:

```bash
cd backend
python app.py
```

**Expected output**:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 4.2 Access Frontend
1. Navigate to the `frontend` directory
2. Open `index.html` in a web browser:
   - Use a local server:
     ```bash
     cd frontend
     python -m http.server 8080
     ```
     Then navigate to `http://localhost:8080`

---

**Last Updated**: October 2025  
**Version**: 1.0