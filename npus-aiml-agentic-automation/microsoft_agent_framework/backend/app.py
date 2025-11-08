"""
FastAPI server for MLOps Onboarding Assistant using Microsoft Agent Framework
Provides REST API endpoints for the onboarding agent
"""

import uuid
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from data_models.app_models import (
    StartConversationRequest,
    MessageRequest,
    MessageResponse,
    StatusResponse,
)
from contextlib import asynccontextmanager
from agent import MLOpsOnboardingAgent

load_dotenv()

# Store active agent instances and threads by session
active_sessions: Dict[str, Dict[str, Any]] = {}
# Global agent instance
agent_instance: Optional[MLOpsOnboardingAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global agent_instance
    # Startup
    try:
        agent_instance = MLOpsOnboardingAgent()
        print("MLOps Onboarding Agent initialized successfully")
    except Exception as e:
        print(f"Error initializing agent: {e}")
        raise

    yield  # Application runs here

    # Shutdown (cleanup code goes here if needed)
    print("Shutting down MLOps Onboarding Agent")
    agent_instance = None


# Create FastAPI app with lifespan
app = FastAPI(title="MLOps Onboarding Assistant API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.post("/start_conversation", response_model=MessageResponse)
async def start_conversation(request: StartConversationRequest):
    """Start a new conversation session"""
    global agent_instance

    if not agent_instance:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    session_id = request.session_id or str(uuid.uuid4())
    thread = agent_instance.custom_agent.get_new_thread()

    active_sessions[session_id] = {
        "thread": thread,
        "conversation_history": [],
        "is_first_message": True,
        "status": "active",
    }

    try:
        initial_message = request.initial_message or "Hello"

        messages = agent_instance._create_message_with_system_instructions(
            initial_message, is_first_message=True
        )

        from agent_framework import MCPStreamableHTTPTool

        async with MCPStreamableHTTPTool(
            name="AIOps servers",
            url=agent_instance.mcp_server_url,
        ) as mcp_server:
            response = await agent_instance.custom_agent.run(
                messages, thread=thread, tools=mcp_server
            )
            response_text = response.text

        active_sessions[session_id]["conversation_history"].extend(
            [
                {"role": "user", "content": initial_message},
                {"role": "assistant", "content": response_text},
            ]
        )
        active_sessions[session_id]["is_first_message"] = False

        return MessageResponse(
            session_id=session_id,
            response=response_text,
            status="active",
            conversation_history=active_sessions[session_id]["conversation_history"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error starting conversation: {str(e)}"
        )


@app.post("/send_message", response_model=MessageResponse)
async def send_message(request: MessageRequest):
    """Send a message in an existing conversation"""
    global agent_instance

    if not agent_instance:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    # Check if session exists
    if request.session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions[request.session_id]

    # Check for exit commands
    exit_commands = [
        "quit",
        "exit",
        "done",
        "goodbye",
        "bye",
        "stop",
        "that's all",
        "no more help needed",
    ]

    if request.message.lower().strip() in exit_commands:
        # Get closing message from agent
        try:
            closing_prompt = f"User said: '{request.message}'. Provide a warm, brief closing message."

            messages = agent_instance._create_message_with_system_instructions(
                closing_prompt, is_first_message=False
            )

            from agent_framework import MCPStreamableHTTPTool

            async with MCPStreamableHTTPTool(
                name="AIOps servers",
                url=agent_instance.mcp_server_url,
            ) as mcp_server:
                response = await agent_instance.custom_agent.run(
                    messages, thread=session["thread"], tools=mcp_server
                )
                response_text = response.text

            # Update conversation history
            session["conversation_history"].extend(
                [
                    {"role": "user", "content": request.message},
                    {"role": "assistant", "content": response_text},
                ]
            )

            # Mark session as complete
            session["status"] = "complete"

            return MessageResponse(
                session_id=request.session_id,
                response=response_text,
                status="complete",
                conversation_history=session["conversation_history"],
            )

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error ending conversation: {str(e)}"
            )

    # Process regular message
    try:
        messages = agent_instance._create_message_with_system_instructions(
            request.message, is_first_message=False
        )

        from agent_framework import MCPStreamableHTTPTool

        async with MCPStreamableHTTPTool(
            name="AIOps servers",
            url=agent_instance.mcp_server_url,
        ) as mcp_server:
            response = await agent_instance.custom_agent.run(
                messages, thread=session["thread"], tools=mcp_server
            )
            response_text = response.text

        # Update conversation history
        session["conversation_history"].extend(
            [
                {"role": "user", "content": request.message},
                {"role": "assistant", "content": response_text},
            ]
        )

        return MessageResponse(
            session_id=request.session_id,
            response=response_text,
            status=session["status"],
            conversation_history=session["conversation_history"],
        )

    except Exception as e:
        session["status"] = "error"
        raise HTTPException(
            status_code=500, detail=f"Error processing message: {str(e)}"
        )


@app.get("/status/{session_id}", response_model=StatusResponse)
async def get_status(session_id: str):
    """Get the current status of a conversation session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions[session_id]

    return StatusResponse(
        status=session["status"],
        session_id=session_id,
        conversation_history=session["conversation_history"],
        result=None,
        error=None,
    )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a conversation session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    del active_sessions[session_id]
    return {
        "status": "deleted",
        "message": f"Session {session_id} deleted successfully",
    }


@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    sessions = [
        {
            "session_id": sid,
            "status": session["status"],
            "message_count": len(session["conversation_history"]),
        }
        for sid, session in active_sessions.items()
    ]
    return {"sessions": sessions, "total": len(sessions)}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "MLOps Onboarding Assistant API is running",
        "agent_initialized": agent_instance is not None,
        "active_sessions": len(active_sessions),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8070)
