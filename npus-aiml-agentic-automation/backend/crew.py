import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.tools import tool
from typing import List, Dict, Any, Optional
from mcp import StdioServerParameters
from dotenv import load_dotenv
import threading
from custom_llm.nestle_llm import NestleLLM

load_dotenv()

# Global state management for human-in-the-loop
crew_state = {
    "status": "idle",
    "prompt": None,
    "result": None,
    "user_input": None,
    "input_event": None,
}


# Initialize the custom LLM
def initialize_llm():
    """Initialize the Nestle LLM with credentials from environment"""
    client_id = os.getenv("NESTLE_CLIENT_ID")
    client_secret = os.getenv("NESTLE_CLIENT_SECRET")
    model = os.getenv("NESTLE_MODEL", "gpt-4.1")

    # Create LLM for CrewAI
    llm = NestleLLM(
        model=model,
        client_id=client_id,
        client_secret=client_secret,
    )

    return llm


# Initialize the LLM once at module level
llm = initialize_llm()


@tool
def get_human_input(prompt: str, context: str = "") -> str:
    """Get input from human user through UI interface.

    Args:
        prompt: The question or prompt to show the human
        context: Additional context information to display

    Returns:
        Human response as string
    """
    # Handle case where agent might pass a dict instead of string
    if isinstance(prompt, dict):
        if "description" in prompt:
            prompt_text = prompt["description"]
        else:
            prompt_text = str(prompt)
    else:
        prompt_text = str(prompt)

    if os.environ.get("TERMINAL", "false").lower() == "true":
        user_response = input(f"AGENT: {prompt_text}\nUSER:")
    else:
        crew_state["status"] = "waiting_input"
        crew_state["prompt"] = prompt_text
        crew_state["input_event"] = threading.Event()

        # Wait for user input
        crew_state["input_event"].wait()

        user_response = crew_state["user_input"].strip()
        crew_state["status"] = "running"
        crew_state["prompt"] = None

    return user_response


@CrewBase
class AiopsAgenticAutomation:
    """AiopsAgenticAutomation crew with dynamic task selection and UI-based human input"""

    agents: List[BaseAgent]
    tasks: List[Task]

    mcp_server_params = [
        {"url": "http://127.0.0.1:8060/mcp", "transport": "streamable-http"}
    ]

    def __init__(self):
        self.user_context = {}
        self.llm = llm  # Use the Nestle LLM

    def set_user_context(self, context: Dict[str, Any]):
        """Set user context for personalized automation"""
        self.user_context = context

    @agent
    def supervisor_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["supervisor_agent"],
            tools=self.get_mcp_tools() + [get_human_input],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            memory=True,
        )

    @task
    def supervision_task(self) -> Task:
        return Task(
            config=self.tasks_config["supervision_task"],
            agent=self.supervisor_agent(),
            human_input=False,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
