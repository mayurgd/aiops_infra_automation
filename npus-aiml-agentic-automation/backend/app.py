import os
from typing import Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from crew import AiopsAgenticAutomation, crew_state
from models import (
    CrewRequest,
    InputResponse,
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
        result = crew_instance.crew().kickoff(crew_inputs)

        crew_state["status"] = "complete"
        crew_state["result"] = str(result)

    except Exception as e:
        crew_state["status"] = "error"
        crew_state["result"] = f"Error: {str(e)}"
        import traceback

        print(f"Crew execution error: {traceback.format_exc()}")


@app.post("/start_crew")
async def start_crew(request: CrewRequest, background_tasks: BackgroundTasks):
    """Start the AiopsAgenticAutomation crew with user query and inputs"""
    crew_state["status"] = "starting"
    crew_state["result"] = None
    crew_state["prompt"] = None
    background_tasks.add_task(run_crew_async, request.user_query, request.inputs)
    return {"status": "started", "message": "Crew execution started"}


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


@app.get("/status")
async def get_status():
    """Get current status of the crew execution"""
    return {
        "status": crew_state["status"],
        "prompt": crew_state["prompt"],
        "result": crew_state["result"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8070)
