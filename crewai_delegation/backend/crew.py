import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.tools import tool
from typing import List, Dict, Any, Optional
from mcp import StdioServerParameters
from dotenv import load_dotenv
import threading

crew_state = {
    "status": "idle",
    "prompt": None,
    "result": None,
    "user_input": None,
    "input_event": None,
}

LOCAL = True


@tool
def get_human_input(prompt: str, context: str = "") -> str:
    """Get input from human user through UI interface or local terminal.

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

    if LOCAL:
        # Local terminal execution
        print("\n" + "=" * 50)
        if context:
            print(f"Context: {context}")
            print("-" * 50)
        print(f"Agent: {prompt_text}")
        print("=" * 50)

        try:
            user_response = input("Your response: ").strip()
            return user_response
        except KeyboardInterrupt:
            print("\n\nExiting...")
            return "exit"
        except EOFError:
            return ""
    else:
        # Original UI interface execution
        crew_state["status"] = "waiting_input"
        crew_state["prompt"] = prompt_text
        crew_state["input_event"] = threading.Event()

        # Wait for user input
        crew_state["input_event"].wait()

        user_response = crew_state["user_input"].strip()
        crew_state["status"] = "running"
        crew_state["prompt"] = None

        return user_response


load_dotenv()


@CrewBase
class AiopsAgenticAutomation:
    """AiopsAgenticAutomation crew with proper agent delegation"""

    agents: List[BaseAgent]
    tasks: List[Task]

    mcp_server_params = [
        StdioServerParameters(
            command="python",
            args=[os.environ["SERVER_LOCATION"]],
        ),
    ]

    def __init__(self):
        super().__init__()
        self.user_context = {}

    def set_user_context(self, context: Dict[str, Any]):
        """Set user context for personalized automation"""
        self.user_context = context

    @agent
    def guidance_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["guidance_agent"],
            tools=[get_human_input],
            verbose=True,
        )

    @agent
    def github_requirements_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["github_requirements_agent"],
            tools=[get_human_input],
            verbose=True,
        )

    @task
    def intent_determination_task(self) -> Task:
        return Task(
            config=self.tasks_config["intent_determination_task"],
            agent=self.guidance_agent(),
        )

    @task
    def github_requirements_gathering_task(self) -> Task:
        return Task(
            config=self.tasks_config["github_requirements_gathering_task"],
            agent=self.github_requirements_agent(),
        )

    @crew
    def crew(self) -> Crew:
        """Creates crew with all agents and tasks available for delegation"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_llm="gpt-4o",
            verbose=True,
        )


# Example usage
def main():
    """Example of how to use the crew with proper delegation"""
    crew_instance = AiopsAgenticAutomation()

    print("🤖 Starting AI-Ops Service Selection...")
    print(
        "Available services: GitHub Repository Setup, Databricks Schema Creation, Databricks Compute Configuration"
    )

    # Just run the crew - the guidance agent will handle delegation internally
    result = crew_instance.crew().kickoff()

    print("\n" + "=" * 60)
    print("🎉 FINAL RESULT:")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
