from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import uuid
from typing import Optional, Dict, Any
import threading
import queue
from lg_agent import ModernizedSupervisorGraph
from langgraph.types import Command


# Pydantic models for API
class StartCrewRequest(BaseModel):
    user_query: str
    inputs: dict = {}


class SubmitInputRequest(BaseModel):
    input: str


class StatusResponse(BaseModel):
    status: str
    prompt: Optional[str] = None
    result: Optional[str] = None


# Global state management
class SessionManager:
    def __init__(self):
        self.sessions = {}
        self.current_session_id = None

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        supervisor = ModernizedSupervisorGraph()
        graph = supervisor.build_graph()

        self.sessions[session_id] = {
            "graph": graph,
            "config": {"configurable": {"thread_id": session_id}},
            "status": "idle",
            "result": None,
            "interrupt_queue": queue.Queue(),
            "current_prompt": None,
            "conversation_history": [],
        }
        self.current_session_id = session_id
        return session_id

    def get_current_session(self):
        if self.current_session_id and self.current_session_id in self.sessions:
            return self.sessions[self.current_session_id]
        return None


# Initialize FastAPI app
app = FastAPI(title="AIOps Agentic Automation API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global session manager
session_manager = SessionManager()


@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "🚀 AIOps Automation Server is running!"}


@app.post("/start_crew")
async def start_crew(request: StartCrewRequest):
    try:
        # Create new session
        session_id = session_manager.create_session()
        session = session_manager.get_current_session()

        # Set status to starting
        session["status"] = "starting"

        # Start the graph execution in background
        def run_graph():
            try:
                initial_state = {
                    "user_request": request.user_query,
                    "intent": "",
                    "next_action": "",
                    "conversation_history": [],
                    "current_step": "start",
                    "github_requirements": {},
                    "databricks_schema_requirements": {},
                    "databricks_compute_requirements": {},
                    "execution_result": {},
                    "error_message": "",
                }

                session["status"] = "running"
                result = session["graph"].invoke(
                    initial_state, config=session["config"]
                )

                # Handle interrupts
                while "__interrupt__" in result:
                    interrupt_data = result["__interrupt__"][0]
                    prompt_text = interrupt_data.value.get(
                        "prompt", "Please provide your input:"
                    )

                    # Store prompt and wait for user input
                    session["current_prompt"] = prompt_text
                    session["status"] = "waiting_input"

                    # Wait for user input
                    try:
                        user_response = session["interrupt_queue"].get(
                            timeout=300
                        )  # 5 minute timeout
                        result = session["graph"].invoke(
                            Command(resume=user_response), config=session["config"]
                        )
                        session["status"] = "running"
                        session["current_prompt"] = None
                    except queue.Empty:
                        session["status"] = "error"
                        session["result"] = "Session timed out waiting for user input"
                        return

                # Process final result
                if result.get("execution_result", {}).get("success"):
                    session["status"] = "complete"
                    session["result"] = "✅ Automation completed successfully!"
                elif result.get("execution_result", {}).get("error"):
                    session["status"] = "error"
                    session["result"] = (
                        f"❌ Automation failed: {result['execution_result']['error']}"
                    )
                else:
                    session["status"] = "complete"
                    session["result"] = "Process completed"

            except Exception as e:
                session["status"] = "error"
                session["result"] = f"❌ Error: {str(e)}"

        # Start in background thread
        thread = threading.Thread(target=run_graph)
        thread.daemon = True
        thread.start()

        return {"message": "🚀 Automation process started!", "session_id": session_id}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to start automation: {str(e)}"
        )


@app.post("/submit_input")
async def submit_input(request: SubmitInputRequest):
    try:
        session = session_manager.get_current_session()
        if not session:
            raise HTTPException(status_code=404, detail="No active session found")

        if session["status"] != "waiting_input":
            raise HTTPException(status_code=400, detail="Not waiting for input")

        # Submit user input to the queue
        session["interrupt_queue"].put(request.input)

        return {"message": "Input submitted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit input: {str(e)}")


@app.get("/status")
async def get_status():
    session = session_manager.get_current_session()
    if not session:
        return StatusResponse(status="idle")

    return StatusResponse(
        status=session["status"],
        prompt=session.get("current_prompt"),
        result=session.get("result"),
    )


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting AIOps Automation FastAPI Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
