from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv

# Import the updated crew class
from crew import AiopsAgenticAutomation, crew_state

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

# Set your OpenAI API key here or in environment
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"


class CrewRequest(BaseModel):
    user_query: str
    inputs: Optional[Dict[str, Any]] = {}


class InputResponse(BaseModel):
    input: str


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


# Example endpoints for testing different automation scenarios


@app.post("/create_repo")
async def create_repo_only(background_tasks: BackgroundTasks):
    """Example endpoint for creating GitHub repo only"""
    inputs = {
        "use_case_name": "test-repo",
        "template": "python-template",
        "internal_team": "data-team",
        "development_team": "dev-team",
    }

    crew_state["status"] = "starting"
    crew_state["result"] = None
    crew_state["prompt"] = None

    background_tasks.add_task(run_crew_async, "Create a GitHub repository", inputs)
    return {"status": "started", "message": "GitHub repo creation started"}


@app.post("/create_databricks")
async def create_databricks_only(background_tasks: BackgroundTasks):
    """Example endpoint for creating Databricks schema only"""
    inputs = {
        "catalog": "test_catalog",
        "schema": "test_schema",
        "aiml_support_team": "ml-team",
        "aiml_use_case": "data-analysis",
        "business_owner": "business-team",
    }

    crew_state["status"] = "starting"
    crew_state["result"] = None
    crew_state["prompt"] = None

    background_tasks.add_task(run_crew_async, "Create Databricks schema", inputs)
    return {"status": "started", "message": "Databricks schema creation started"}


@app.post("/create_all")
async def create_all_resources(background_tasks: BackgroundTasks):
    """Example endpoint for creating all resources"""
    inputs = {
        # GitHub repo inputs
        "use_case_name": "full-automation-test",
        "template": "ml-template",
        "internal_team": "data-team",
        "development_team": "dev-team",
        # Databricks schema inputs
        "catalog": "automation_catalog",
        "schema": "automation_schema",
        "aiml_support_team": "ml-support",
        "aiml_use_case": "automated-ml",
        "business_owner": "product-team",
        # Compute inputs
        "cluster_name": "automation-cluster",
        "spark_version": "11.3.x-scala2.12",
        "driver_node_type_id": "i3.xlarge",
        "node_type_id": "i3.xlarge",
    }

    crew_state["status"] = "starting"
    crew_state["result"] = None
    crew_state["prompt"] = None

    background_tasks.add_task(
        run_crew_async, "Create complete infrastructure setup", inputs
    )
    return {"status": "started", "message": "Full automation started"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
