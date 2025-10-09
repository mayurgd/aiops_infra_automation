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
        StdioServerParameters(
            command="python",
            args=[os.environ["SERVER_LOCATION"]],
        ),
    ]

    def __init__(self):
        super().__init__()
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

    @agent
    def git_repo_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["git_repo_agent"],
            tools=self.get_mcp_tools(),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def databricks_schema_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["databricks_schema_agent"],
            tools=self.get_mcp_tools(),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def databricks_compute_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["databricks_compute_agent"],
            tools=self.get_mcp_tools(),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    @task
    def supervision_task(self) -> Task:
        return Task(
            config=self.tasks_config["supervision_task"],
            agent=self.supervisor_agent(),
            human_input=False,
        )

    @task
    def create_repo_task(self) -> Task:
        return Task(
            config=self.tasks_config["create_repo_task"],
            agent=self.git_repo_agent(),
        )

    @task
    def create_databricks_schema_task(self) -> Task:
        return Task(
            config=self.tasks_config["create_databricks_schema_task"],
            agent=self.databricks_schema_agent(),
        )

    @task
    def create_databricks_compute_task(self) -> Task:
        return Task(
            config=self.tasks_config["create_databricks_compute_task"],
            agent=self.databricks_compute_agent(),
        )

    def _determine_required_tasks(
        self, user_query: str, inputs: Dict[str, Any]
    ) -> List[str]:
        """Use LLM to determine which tasks are needed based on user query and inputs"""
        # Check what resources are being requested based on input keys
        has_repo_inputs = any(
            key in inputs
            for key in [
                "use_case_name",
                "template",
                "internal_team",
                "development_team",
                "additional_team",
            ]
        )

        has_databricks_inputs = any(
            key in inputs
            for key in [
                "catalog",
                "schema",
                "aiml_support_team",
                "aiml_use_case",
                "business_owner",
            ]
        )

        has_compute_inputs = any(
            key in inputs
            for key in [
                "cluster_name",
                "spark_version",
                "driver_node_type_id",
                "node_type_id",
            ]
        )
        print(
            f"user_query:{user_query}\ninputs:{inputs}\nhas_repo_inputs:{has_repo_inputs}\nhas_databricks_inputs:{has_databricks_inputs}\nhas_compute_inputs:{has_compute_inputs}"
        )
        # Create a simple routing agent to make the determination
        router_agent = Agent(
            role="Task Router",
            goal="Determine which automation tasks are needed based on user query and inputs",
            backstory="You analyze user queries and input parameters to determine what automation tasks should be executed",
            llm=self.llm,
            verbose=False,
        )

        routing_task = Task(
            description=f"""
            Analyze this user query and input parameters to determine which automation tasks are needed.
            
            User Query: "{user_query}"
            
            Input Analysis:
            - Has GitHub repo inputs: {has_repo_inputs}
            - Has Databricks inputs: {has_databricks_inputs}
            - Has Compute inputs: {has_compute_inputs}
            - Available input keys: {list(inputs.keys())}
            
            Based on the inputs provided, determine what the user wants to automate.
            
            Return ONLY one of these exact responses:
            - "repo_only" - if user only wants GitHub repository creation and has provided all required GitHub repo inputs
            - "databricks_only" - if user only wants Databricks schema/catalog creation and has provided all required Databricks inputs
            - "compute_only" - if user only wants Databricks compute cluster creation and has provided all required compute inputs
            - "databricks_and_compute" - if user wants both Databricks schema and compute cluster and has provided all required inputs for both
            - "repo_and_databricks" - if user wants GitHub repo and Databricks schema and has provided all required inputs for both
            - "repo_and_compute" - if user wants GitHub repo and compute cluster and has provided all required inputs for both
            - "all" - if user wants GitHub repo, Databricks schema, and compute cluster and has provided all required inputs for all
            - "supervision_needed" - if inputs are missing, incomplete, contradictory, or if the request is ambiguous
            
            Rules:
            - A choice should only be made if ALL required inputs for that task (or combination of tasks) are present.
            - If any required inputs are missing, return "supervision_needed".
            - If the query is ambiguous or unclear about which automation tasks are needed, return "supervision_needed".
            
            Response:
        """,
            agent=router_agent,
            expected_output="One of: repo_only, databricks_only, compute_only, databricks_and_compute, repo_and_databricks, repo_and_compute, all, or route_needed",
        )

        # Create a minimal crew just for routing
        routing_crew = Crew(
            agents=[router_agent],
            tasks=[routing_task],
            process=Process.sequential,
            verbose=False,
        )

        # Get the routing decision
        routing_result = routing_crew.kickoff()
        decision = str(routing_result).strip().lower()

        # Map decision to required tasks
        if "repo_only" in decision:
            return ["create_repo_task"]
        elif "databricks_only" in decision:
            return ["create_databricks_schema_task"]
        elif "compute_only" in decision:
            return ["create_databricks_compute_task"]
        elif "databricks_and_compute" in decision:
            return ["create_databricks_schema_task", "create_databricks_compute_task"]
        elif "repo_and_databricks" in decision:
            return ["create_repo_task", "create_databricks_schema_task"]
        elif "repo_and_compute" in decision:
            return ["create_repo_task", "create_databricks_compute_task"]
        elif "all" in decision:
            return [
                "create_repo_task",
                "create_databricks_schema_task",
                "create_databricks_compute_task",
            ]
        elif "supervision_needed" in decision:
            return ["supervision_task"]
        else:
            # Default based on available inputs
            tasks = []
            if has_repo_inputs:
                tasks.append("create_repo_task")
            if has_databricks_inputs:
                tasks.append("create_databricks_schema_task")
            if has_compute_inputs:
                tasks.append("create_databricks_compute_task")
            return tasks if tasks else ["supervision_task"]

    def create_dynamic_crew(
        self,
        user_query: str,
        inputs: Dict[str, Any] = None,
    ) -> Crew:
        """Create a crew with only the required tasks based on LLM routing decision"""

        if inputs is None:
            inputs = {}

        # Determine required tasks using LLM
        required_tasks = self._determine_required_tasks(user_query, inputs)

        print(f"LLM determined required tasks: {required_tasks}")

        # Create agents and tasks lists based on requirements
        agents = []
        tasks = []

        # Keep references to tasks for context setting
        repo_task = None
        databricks_task = None
        compute_task = None

        # Add routing task if needed
        if "supervision_task" in required_tasks:
            agents.append(self.supervisor_agent())
            supervision_task = self.supervision_task()
            tasks.append(supervision_task)

        # Add repo creation task if needed
        if "create_repo_task" in required_tasks:
            agents.append(self.git_repo_agent())
            repo_task = self.create_repo_task()
            tasks.append(repo_task)

        # Add databricks schema task if needed
        if "create_databricks_schema_task" in required_tasks:
            agents.append(self.databricks_schema_agent())
            databricks_task = self.create_databricks_schema_task()

            # If repo task exists, databricks can reference it
            if repo_task is not None:
                databricks_task.context = [repo_task]

            tasks.append(databricks_task)

        # Add databricks compute task if needed
        if "create_databricks_compute_task" in required_tasks:
            agents.append(self.databricks_compute_agent())
            compute_task = self.create_databricks_compute_task()

            # Set context dependencies - compute should run after schema creation
            context_tasks = []
            if databricks_task is not None:
                context_tasks.append(databricks_task)
            if repo_task is not None:
                context_tasks.append(repo_task)

            if context_tasks:
                compute_task.context = context_tasks

            tasks.append(compute_task)

        # Ensure we have at least one task
        if not tasks:
            # Fallback to routing task
            agents.append(self.supervisor_agent())
            tasks.append(self.supervision_task())

        # Create and return the dynamic crew
        return Crew(
            agents=list(set(agents)),  # Remove duplicates
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )

    @crew
    def crew(self) -> Crew:
        """Default crew method - will be overridden by dynamic crew creation"""
        return Crew(
            agents=[
                self.git_repo_agent(),
                self.databricks_schema_agent(),
                self.databricks_compute_agent(),
            ],
            tasks=[
                self.create_repo_task(),
                self.create_databricks_schema_task(),
                self.create_databricks_compute_task(),
            ],
            process=Process.sequential,
            verbose=True,
        )

    def kickoff(self, inputs: Dict[str, Any]) -> Any:
        """Custom kickoff method that creates dynamic crew based on query and inputs"""

        # Extract user query if provided, otherwise create a default one
        user_query = inputs.get("user_query", "Automate infrastructure setup")

        # Set user context if provided
        if "user_context" in inputs:
            self.set_user_context(inputs["user_context"])

        # Create dynamic crew based on query and inputs
        dynamic_crew = self.create_dynamic_crew(user_query, inputs)

        # Run the dynamic crew
        return dynamic_crew.kickoff(inputs=inputs)
