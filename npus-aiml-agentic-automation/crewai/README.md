# AIOps Agentic Automation - Setup Guide

## Overview
This guide will help you set up and run the AIOps Agentic Automation system, which provides an intelligent interface for automating GitHub repository creation, Databricks schema provisioning, and compute cluster setup.

---

## Prerequisites
- **Python 3.11** installed and available in PATH
- GitHub account with access to `nestle-it` organization
- Databricks workspace access
- Azure Service Principal credentials
- Access to Azure Key Vault (`npuspraimlkey`)
- Azure DevOps access for wiki repository
- Windows OS (for batch script launcher)

---

## Step 1: GitHub Token Configuration

### 1.1 Generate Personal Access Token
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

### 1.2 Authorize SSO
1. Refresh the tokens page
2. Locate your newly created token
3. Click **Configure SSO**
4. Click **Authorize** next to `nestle-it` organization
5. Complete the authorization process

---

## Step 2: Azure Databricks Secrets

Retrieve the following secrets from Azure Key Vault:

**Key Vault**: `npuspraimlkey` ([Azure Portal](https://portal.azure.com/#@nestle.onmicrosoft.com/resource/subscriptions/a6c9f13d-be73-4227-b485-c7a16289be08/resourceGroups/nppc-pr-pl4shared-a4adhpr-usw2-rg-001/providers/Microsoft.KeyVault/vaults/npuspraimlkey/secrets))

**Required Secrets**:
- `tenant-id`
- `npus-pr-aimlstage-dbr-spn-clientId`
- `npus-pr-aimlstage-dbr-spn-secret`

---

## Step 3: Configuration File Setup

### 3.1 Create Environment File
Copy the example environment file and place it in the root directory:

```bash
copy .env.example .env
```

### 3.2 Configure Environment Variables
Open the `.env` file and update the following values:

#### LLM Configuration
```bash
# Nestle Internal API (Active Configuration)
NESTLE_CLIENT_ID=your-nestle-oauth-client-id
NESTLE_CLIENT_SECRET=your-nestle-oauth-client-secret
NESTLE_MODEL=gpt-4.1
```

**Note**: The system is configured to use Nestle's internal LLM API by default.

#### GitHub Configuration
```bash
GITHUB_TOKEN=your-github-token-from-step-1
GITHUB_ORG=nestle-it
AUTOMATION_REPO=npus-aiml-utilities-create-repo
```

#### CrewAI Configuration
```bash
OTEL_SDK_DISABLED=true
CREWAI_DISABLE_TELEMETRY=true
CREWAI_DISABLE_TRACKING=true
SERVER_LOCATION=C:/path/to/npus-aiml-agentic-automation/backend/servers/aiops_automation_server.py
```

**Important**: Update `SERVER_LOCATION` with the absolute path to `aiops_automation_server.py` on your system. 
Use forward slashes (/) or escaped backslashes (\\\\) in the path.

#### Databricks Configuration
```bash
DATABRICKS_WORKSPACE_URL=https://adb-1125343200912494.14.azuredatabricks.net/
ARM_TENANT_ID=your-azure-tenant-id
ARM_CLIENT_ID=your-azure-client-id
ARM_CLIENT_SECRET=your-azure-client-secret
```
Use the values retrieved from Azure Key Vault in Step 2.

#### Knowledge Base Configuration
```bash
WIKI_REPO_URL=https://nestle-nppc@dev.azure.com/nestle-nppc/Purina%20AIML/_git/npus-aiml-wiki
WIKI_REPO_LOC=docs/npus-aiml-wiki
VECTOR_DB_LOC=docs/mlops_vector_db
```

**Note**: These settings control the wiki documentation source and vector database location used by the knowledge base system.

#### Runtime Configuration
```bash
TERMINAL=false
```
- Set to `false` for web UI interaction (recommended)
- Set to `true` for terminal/CLI mode

---

## Step 4: Running the Application

### 4.1 Automated Startup (Recommended)

Simply double-click the `startup.bat` file in the project root directory, or run it from command prompt:

```bash
startup.bat
```

**What it does**:
- Validates Python 3.11 installation (warns if different version detected)
- Checks that `.env` file exists (in root or backend directory)
- Verifies backend and frontend directories are present
- Validates all required scripts exist:
  - `wiki_downloader.py`
  - `knowledge_base_utility.py`
  - `aiops_automation_server.py`
- Automatically creates a Python virtual environment if it doesn't exist
- Installs all required dependencies automatically
- Executes the following startup sequence:
  1. **Wiki Downloader**: Downloads and updates wiki documentation from Azure DevOps
  2. **Knowledge Base Utility**: Prepares vector database for the system
  3. **MCP Server**: Starts on `http://127.0.0.1:8060`
  4. **Backend Server**: Starts on `http://127.0.0.1:8070`
  5. **Frontend Server**: Starts on `http://localhost:8080`
- Opens the frontend automatically in your default browser
- Displays detailed status updates throughout the process

**Expected output**:
```
========================================
AIOps Agentic Automation - Launcher
========================================

[INFO] Checking Python version...
[INFO] Found Python version: 3.11.x
[OK] Python 3.11 detected
[OK] .env file found in root directory
[OK] All required directories and scripts found
[OK] requirements.txt found
[OK] Virtual environment found

========================================
Step 1: Running Wiki Downloader
========================================
[INFO] Executing wiki_downloader.py...
[OK] Wiki downloader completed successfully!

========================================
Step 2: Running Knowledge Base Utility
========================================
[INFO] Executing knowledge_base_utility.py...
[OK] Knowledge base utility completed successfully!

========================================
Step 3: Launching MCP Server
========================================
[INFO] Waiting for MCP server to initialize...
[OK] MCP server should be running now!

========================================
Step 4: Launching Backend Server
========================================
[INFO] Waiting for backend server to initialize...
[OK] Backend server is ready!

========================================
Step 5: Launching Frontend Server
========================================
[INFO] Waiting for frontend server to initialize...

========================================
All Steps Completed Successfully!
========================================

Startup Sequence:
1. Wiki Downloader      - Completed
2. Knowledge Base Setup - Completed
3. MCP Server          - Running on http://127.0.0.1:8060
4. Backend Server      - Running on http://127.0.0.1:8070
5. Frontend Server     - Running on http://localhost:8080

Three command prompt windows have been opened:
1. MCP Server
2. Backend Server
3. Frontend Server

Close those windows or press Ctrl+C in them to stop the servers.
```

### 4.2 Manual Startup (Alternative)

If you prefer to start servers manually or are not on Windows:

#### Step 1: Set up virtual environment
From project root directory:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

#### Step 2: Run Wiki Downloader
```bash
cd backend
python -m docs.wiki_downloader
```

#### Step 3: Run Knowledge Base Utility
```bash
python -m servers.knowledge_base_utility
```

#### Step 4: Start MCP Server
In a new terminal:
```bash
cd backend
venv\Scripts\activate
python -m servers.aiops_automation_server
```

#### Step 5: Start Backend Server
In a new terminal:
```bash
cd backend
venv\Scripts\activate
python app.py
```

#### Step 6: Start Frontend Server
In a new terminal:
```bash
cd frontend
python -m http.server 8080
```

Then navigate to `http://localhost:8080` in your browser.

---

## Troubleshooting

### Python Version Issues
- If Python 3.11 is not detected, the script will warn you and ask for confirmation to continue
- While other Python 3.x versions may work, 3.11 is strongly recommended for compatibility

### Missing Dependencies
- The startup script will automatically install dependencies from `requirements.txt`
- If installation fails, try manually running: `pip install -r requirements.txt`

### Wiki Downloader Issues
- Ensure you have access to the Azure DevOps wiki repository
- Check that `WIKI_REPO_URL` in `.env` is correct
- Verify your Azure DevOps credentials are valid

### Knowledge Base Issues
- If vector database creation fails, check that `VECTOR_DB_LOC` path is writable
- Ensure wiki content was downloaded successfully in Step 1

### Server Health Checks
- The backend server uses `curl` to perform health checks
- If `curl` is not available, the script may not detect when the backend is ready
- Install `curl` or wait an additional 5-10 seconds before accessing the frontend

### Virtual Environment Issues
- Delete the `venv` folder and run `startup.bat` again to recreate it
- Ensure you have write permissions in the project directory

### Port Conflicts
If any of the default ports are in use, you'll need to:
- Stop the conflicting service, or
- Modify the port numbers in the respective server files

### Configuration Issues
- Verify all required fields in `.env` are populated (no placeholder values like TOKEN, CLIENT_ID, etc.)
- Check that file paths use forward slashes (/) or proper escaped backslashes (\\\\)
- Ensure there are no extra spaces or quotes around configuration values

---