# Project Backlog: Automated GitHub Repo, Databricks Schema & Compute Provisioning  

## Pre-Requirements   
- [X] Generate **GitHub CLI Token** (with repo and workflow creation permissions)  [USE PAT]
- [X] Generate **Databricks CLI Token** (with schema + compute provisioning permissions)  [USE THE STAGE SP]
- [X] Obtain **LLM API Model & Token** (for CrewAI agent integration) [CREATE SKELETON WITH GPT-4o]
- [X] Resolve issues regarding **Python environment setup**

---

## 1. Foundations  
### Tasks  
- [ ] Confirm and Finalize the onboarding form details

- [X] Set up Python development environment  
  - [X] Create isolated Python environment (`venv`/`conda`)  
  - [X] Install required libraries
  - [X] Validate environment setup with sample GitHub + Databricks SDK calls  

---

## 2. GitHub Repository Automation  
### Tasks  
- [X] Develop script to trigger `repo-utility.yml` GitHub Action via API/CLI  
  - [X] Add error handling and logging  
  - [X] Run end-to-end test with GitHub CLI  
  - [X] Confirm repo structure, permissions, and GitHub Action triggers  

---

## 3. Databricks Schema & Compute Provisioning  
### Tasks  
- [X] Design Databricks asset requirements  
  - [X] Identify catalog & schema for AI/ML assets  
  - [X] Define compute requirements (cluster type, node size, auto-scaling, libraries)  
  - [X] Draft questionnaire for schema + compute metadata  

- [X] Develop schema creation code  
  - [X] Implement script to create schema in Unity Catalog  
  - [X] Add validation for naming conventions & permissions  

- [X] Develop compute provisioning code  
  - [X] Implement script to spin up clusters/workflows  
  - [X] Add support for attaching extra libraries  

- [X] Test and validate Databricks assets  
  - [X] Test schema creation in dev environment  
  - [X] Validate compute provisioning (cluster start/stop, library installs)  
  - [X] Confirm access for specified team members  

---

## 4. Agentic Framework Integration  
### Tasks  
- [ ] Design UI + agent interaction flow  
  - [ ] Create mockup of input form → backend → asset creation  
  - [ ] Define required input fields:  
    - [ ] Use case name  
    - [ ] Business owner  
    - [ ] Schema name  
    - [ ] Repo name  
    - [ ] Compute config
    - [ ] ...TODO

- [ ] Build CrewAI agents  
  - [ ] Validation Agent → check required fields (team names, schema name, repo metadata)  
  - [ ] Creation Agent → trigger repo, schema, and compute creation scripts  
  - [ ] Add logging & error reporting  

- [ ] Run integration testing  
  - [ ] Execute full flow: user submits → validation agent → creation agent executes  
  - [ ] Validate GitHub repo, Databricks schema, and compute assets end-to-end  
