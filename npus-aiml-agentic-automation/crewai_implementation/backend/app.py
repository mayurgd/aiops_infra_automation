import os
from typing import Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from crew import AiopsAgenticAutomation, crew_state
from models import (
    CrewRequest,
    InputResponse,
    CreateRepoRequest,
    CreateSchemaRequest,
    CreateComputeRequest,
)

load_dotenv()

app = FastAPI()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development (including file://)
    allow_credentials=False,  # Set to False when using allow_origins=["*"]
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global crew instance
crew_instance = None


@app.post("/start_crew")
async def start_crew(request: CrewRequest, background_tasks: BackgroundTasks):
    """Start the AiopsAgenticAutomation crew with user query and inputs"""
    crew_state["status"] = "starting"
    crew_state["result"] = None
    crew_state["prompt"] = None
    background_tasks.add_task(run_crew_async, request.user_query, request.inputs)
    return {"status": "started", "message": "Crew execution started"}


@app.get("/status")
async def get_status():
    """Get current status of the crew execution"""
    return {
        "status": crew_state["status"],
        "prompt": crew_state["prompt"],
        "result": crew_state["result"],
    }


@app.post("/submit_input")
async def submit_input(response: InputResponse):
    """Submit user input when crew is waiting for human input"""
    crew_state["user_input"] = response.input
    if crew_state["input_event"]:
        crew_state["input_event"].set()
    return {"status": "received", "message": "Input received successfully"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "AiopsAgenticAutomation API is running"}


def run_crew_async(user_query: str, inputs: Dict[str, Any]):
    """Run crew in a separate thread"""
    global crew_instance

    try:
        crew_state["status"] = "running"

        # Create crew instance
        crew_instance = AiopsAgenticAutomation()

        # Prepare inputs with user query
        crew_inputs = inputs.copy()
        crew_inputs["user_query"] = user_query

        # Execute the crew
        result = crew_instance.kickoff(crew_inputs)

        crew_state["status"] = "complete"
        crew_state["result"] = str(result)

    except Exception as e:
        crew_state["status"] = "error"
        crew_state["result"] = f"Error: {str(e)}"
        import traceback

        print(f"Crew execution error: {traceback.format_exc()}")


@app.post("/create_repo")
async def create_repo_only(
    request: CreateRepoRequest, background_tasks: BackgroundTasks
):
    """Endpoint for creating GitHub repository"""
    inputs = {
        "use_case_name": request.use_case_name,
        "template": request.template,
        "internal_team": request.internal_team,
        "development_team": request.development_team,
        "additional_team": request.additional_team,
    }
    crew_state["status"] = "starting"
    crew_state["result"] = None
    crew_state["prompt"] = None
    background_tasks.add_task(run_crew_async, "Create a GitHub repository", inputs)
    return {"status": "started", "message": "GitHub repo creation started"}


@app.post("/create_schema")
async def create_databricks_schema(
    request: CreateSchemaRequest, background_tasks: BackgroundTasks
):
    """Endpoint for creating Databricks schema"""
    inputs = {
        "catalog": request.catalog,
        "schema": request.schema,
        "aiml_support_team": request.aiml_support_team,
        "aiml_use_case": request.aiml_use_case,
        "business_owner": request.business_owner,
        "internal_entra_id_group": request.internal_entra_id_group,
        "external_entra_id_group": request.external_entra_id_group,
    }
    crew_state["status"] = "starting"
    crew_state["result"] = None
    crew_state["prompt"] = None
    background_tasks.add_task(run_crew_async, "Create Databricks schema", inputs)
    return {"status": "started", "message": "Databricks schema creation started"}


@app.post("/create_compute")
async def create_databricks_compute(
    request: CreateComputeRequest, background_tasks: BackgroundTasks
):
    """Endpoint for creating Databricks compute cluster"""
    inputs = {
        "cluster_name": request.cluster_name,
        "spark_version": request.spark_version,
        "driver_node_type_id": request.driver_node_type_id,
        "node_type_id": request.node_type_id,
        "min_workers": request.min_workers,
        "max_workers": request.max_workers,
        "data_security_mode": request.data_security_mode,
        "aiml_use_case": request.aiml_use_case,
    }
    crew_state["status"] = "starting"
    crew_state["result"] = None
    crew_state["prompt"] = None
    background_tasks.add_task(
        run_crew_async, "Create Databricks compute cluster", inputs
    )
    return {
        "status": "started",
        "message": "Databricks compute cluster creation started",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000)
