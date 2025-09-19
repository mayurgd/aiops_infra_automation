from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.constants import START
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from litellm import completion
import uuid
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.types import interrupt, Command

load_dotenv()

# client = MultiServerMCPClient(
#     {
#         "command": "python",
#         "args": "/Users/mayurgd/Documents/CodingSpace/aiops_infra_automation/servers/aiops_automation_server.py",
#         "transport": "stdio",
#     }
# )
# tools = client.get_tools()

LOCAL = True


# Define the state structure
class SupervisorState(TypedDict):
    user_request: str
    intent: str
    messages: Annotated[list, add_messages]
    retry_count: int
    prompt: str
    current_step: str
    # New fields for GitHub requirements
    github_requirements: dict
    current_requirement: str
    validation_result: bool
    requirement_retry_count: int


class SupervisorGraph:
    def __init__(self):
        self.model = "gpt-4o-mini"
        self.max_retries = 3

    def human_node(self, state: SupervisorState) -> SupervisorState:
        """Handle human input using LangGraph's interrupt mechanism"""
        prompt_data = {
            "prompt": state.get("prompt", "Please provide your input:"),
            "step": state.get("current_step", "input"),
        }

        user_input = interrupt(prompt_data)
        return {"user_request": user_input}

    def greeter_node(self, state: SupervisorState) -> SupervisorState:
        """Greets user and gathers their request intelligently"""
        print("Starting supervisor graph")

        # Set the prompt for human input
        prompt = "Hello! I can help you\n1. Create a GitHub repository\n2. Set up a Databricks schema\n3. Create a Databricks compute cluster\n\nPlease choose an option:"

        return {"prompt": prompt, "current_step": "initial_request", "retry_count": 0}

    def intent_analyzer_node(self, state: SupervisorState) -> SupervisorState:
        """Analyze user intent with LLM"""
        user_request = state["user_request"]

        # Analyze user intent with improved prompt
        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Analyze this user request: "{user_request}"

                                IMPORTANT: Respond with EXACTLY one of these three options (no quotes, no extra text):
                                github_repo
                                databricks_schema
                                databricks_compute
                                not_specified

                                Classification rules:
                                - If they mention "1", "first", "option 1", GitHub, git, repository, repo, code -> respond with: github_repo
                                - If they mention "2", "second", "option 2", Databricks schema, database schema, data schema -> respond with: databricks_schema
                                - If they mention "3", "third", "option 3", Databricks compute, cluster, compute cluster -> respond with: databricks_compute
                                - else not_specified

                                Examples:
                                User: "1" -> github_repo
                                User: "I want option 1" -> github_repo
                                User: "I want to create a GitHub repo" -> github_repo
                                User: "first option" -> github_repo
                                User: "first" -> github_repo
                                If you cant choose any one from the above you can return not_specified
                                """,
                },
            ],
        )

        raw_intent = response["choices"][0]["message"]["content"]
        intent = raw_intent.strip().lower()

        # Debug output
        print(f"Raw LLM response: '{raw_intent}'")
        print(f"Processed intent: '{intent}'")
        print(f"User request: {user_request}")
        print(f"Retry count: {state.get('retry_count', 0)}")

        return {"intent": intent}

    def retry_handler_node(self, state: SupervisorState) -> SupervisorState:
        """Handle retries when intent is not clear"""
        retry_count = state.get("retry_count", 0) + 1

        prompt = f"I didn't understand your request '{state['user_request']}'.\nPlease specify if you want to:\n1. Create a GitHub repository\n2. Set up a Databricks schema\n3. Create a Databricks compute cluster\n\nPlease choose 1, 2, or 3:"

        return {
            "prompt": prompt,
            "current_step": "retry_request",
            "retry_count": retry_count,
        }

    def should_retry(self, state: SupervisorState) -> str:
        """Decide whether to retry based on intent and retry count"""
        intent = state.get("intent", "")
        retry_count = state.get("retry_count", 0)

        valid_intents = ["github_repo", "databricks_schema", "databricks_compute"]

        if intent == "github_repo":
            return "github_flow"
        elif intent in ["databricks_schema", "databricks_compute"]:
            return "complete"
        elif retry_count >= self.max_retries:
            return "complete"
        else:
            return "retry"

    def completion_node(self, state: SupervisorState) -> SupervisorState:
        """Final completion node"""
        intent = state.get("intent", "not_specified")
        print(f"Final detected intent: {intent}")
        print("Graph completed!")

        return {"current_step": "completed", "intent": intent}

    def github_requirements_node(self, state: SupervisorState) -> SupervisorState:
        """Gather GitHub repository requirements iteratively"""

        # Define requirements and their validation options
        requirements_config = {
            "use_case_name": {
                "prompt": "What is the name of the use case for the repository?",
                "validation": None,  # Any string is valid
            },
            "template": {
                "prompt": "Which template would you like to use?\nOptions: npus-aiml-mlops-stacks-template, npus-aiml-skinny-dab-template",
                "valid_options": [
                    "npus-aiml-mlops-stacks-template",
                    "npus-aiml-skinny-dab-template",
                ],
            },
            "internal_team": {
                "prompt": "What is the internal team name?\nOptions: eai, deloitte, sig, tredence, bora, genpact, tiger, srm, kroger",
                "valid_options": [
                    "eai",
                    "deloitte",
                    "sig",
                    "tredence",
                    "bora",
                    "genpact",
                    "tiger",
                    "srm",
                    "kroger",
                ],
            },
            "development_team": {
                "prompt": "What is the development team name?\nOptions: eai, deloitte, sig, tredence, bora, genpact, tiger, srm, kroger, none",
                "valid_options": [
                    "eai",
                    "deloitte",
                    "sig",
                    "tredence",
                    "bora",
                    "genpact",
                    "tiger",
                    "srm",
                    "kroger",
                    "none",
                ],
            },
            "additional_team": {
                "prompt": "What is the additional team?\nOptions: eai, deloitte, sig, tredence, bora, genpact, tiger, srm, kroger, none",
                "valid_options": [
                    "eai",
                    "deloitte",
                    "sig",
                    "tredence",
                    "bora",
                    "genpact",
                    "tiger",
                    "srm",
                    "kroger",
                    "none",
                ],
            },
        }

        github_requirements = state.get("github_requirements", {})
        current_requirement = state.get("current_requirement", "")

        # If no current requirement, start with the first one
        if not current_requirement:
            current_requirement = "use_case_name"
            prompt = f"Let's gather the requirements for your GitHub repository.\n\n{requirements_config[current_requirement]['prompt']}"
            return {
                "prompt": prompt,
                "current_step": "github_requirements",
                "current_requirement": current_requirement,
                "github_requirements": github_requirements,
                "requirement_retry_count": 0,
            }

        return {
            "current_step": "github_requirements",
            "current_requirement": current_requirement,
            "github_requirements": github_requirements,
        }

    def github_requirement_validator_node(
        self, state: SupervisorState
    ) -> SupervisorState:
        """Validate GitHub requirement input using LLM"""

        requirements_config = {
            "use_case_name": {
                "prompt": "What is the name of the use case for the repository?",
                "validation": None,
            },
            "template": {
                "prompt": "Which template would you like to use?\nOptions: npus-aiml-mlops-stacks-template, npus-aiml-skinny-dab-template",
                "valid_options": [
                    "npus-aiml-mlops-stacks-template",
                    "npus-aiml-skinny-dab-template",
                ],
            },
            "internal_team": {
                "prompt": "What is the internal team name?\nOptions: eai, deloitte, sig, tredence, bora, genpact, tiger, srm, kroger",
                "valid_options": [
                    "eai",
                    "deloitte",
                    "sig",
                    "tredence",
                    "bora",
                    "genpact",
                    "tiger",
                    "srm",
                    "kroger",
                ],
            },
            "development_team": {
                "prompt": "What is the development team name?\nOptions: eai, deloitte, sig, tredence, bora, genpact, tiger, srm, kroger, none",
                "valid_options": [
                    "eai",
                    "deloitte",
                    "sig",
                    "tredence",
                    "bora",
                    "genpact",
                    "tiger",
                    "srm",
                    "kroger",
                    "none",
                ],
            },
            "additional_team": {
                "prompt": "What is the additional team?\nOptions: eai, deloitte, sig, tredence, bora, genpact, tiger, srm, kroger, none",
                "valid_options": [
                    "eai",
                    "deloitte",
                    "sig",
                    "tredence",
                    "bora",
                    "genpact",
                    "tiger",
                    "srm",
                    "kroger",
                    "none",
                ],
            },
        }

        user_input = state["user_request"]
        current_requirement = state["current_requirement"]
        github_requirements = state.get("github_requirements", {}).copy()
        requirement_retry_count = state.get("requirement_retry_count", 0)

        # Get requirement config
        req_config = requirements_config[current_requirement]

        # For use_case_name, any non-empty string is valid
        if current_requirement == "use_case_name":
            if user_input.strip():
                github_requirements[current_requirement] = user_input.strip()
                is_valid = True
            else:
                is_valid = False
        else:
            # For other requirements, validate using LLM
            response = completion(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": f"""User input: "{user_input}"
                                    Valid options: {req_config['valid_options']}

                                    IMPORTANT: Respond with EXACTLY one of these options (no quotes, no extra text):
                                    - If the user input matches any of the valid options (case-insensitive), respond with the EXACT valid option
                                    - If the user input doesn't match any valid option, respond with: INVALID

                                    Examples:
                                    User: "eai" -> eai
                                    User: "EAI" -> eai  
                                    User: "Deloitte" -> deloitte
                                    User: "xyz" -> INVALID""",
                    }
                ],
            )

            validated_response = (
                response["choices"][0]["message"]["content"].strip().lower()
            )

            if (
                validated_response != "invalid"
                and validated_response in req_config["valid_options"]
            ):
                github_requirements[current_requirement] = validated_response
                is_valid = True
            else:
                is_valid = False

        return {
            "github_requirements": github_requirements,
            "requirement_retry_count": requirement_retry_count,
            "validation_result": is_valid,
        }

    def github_requirement_next_node(self, state: SupervisorState) -> SupervisorState:
        """Move to next requirement or complete"""

        requirements_order = [
            "use_case_name",
            "template",
            "internal_team",
            "development_team",
            "additional_team",
        ]
        requirements_config = {
            "use_case_name": {
                "prompt": "What is the name of the use case for the repository?"
            },
            "template": {
                "prompt": "Which template would you like to use?\nOptions: npus-aiml-mlops-stacks-template, npus-aiml-skinny-dab-template"
            },
            "internal_team": {
                "prompt": "What is the internal team name?\nOptions: eai, deloitte, sig, tredence, bora, genpact, tiger, srm, kroger"
            },
            "development_team": {
                "prompt": "What is the development team name?\nOptions: eai, deloitte, sig, tredence, bora, genpact, tiger, srm, kroger, none"
            },
            "additional_team": {
                "prompt": "What is the additional team?\nOptions: eai, deloitte, sig, tredence, bora, genpact, tiger, srm, kroger, none"
            },
        }

        current_requirement = state["current_requirement"]
        github_requirements = state["github_requirements"]

        # Find next requirement
        current_index = requirements_order.index(current_requirement)

        if current_index < len(requirements_order) - 1:
            next_requirement = requirements_order[current_index + 1]
            prompt = f"Great! Next requirement:\n\n{requirements_config[next_requirement]['prompt']}"

            return {
                "prompt": prompt,
                "current_requirement": next_requirement,
                "requirement_retry_count": 0,
            }
        else:
            # All requirements gathered
            summary = "All requirements gathered successfully!\n\nSummary:\n"
            for req, value in github_requirements.items():
                summary += f"- {req}: {value}\n"

            return {"current_step": "github_requirements_complete", "prompt": summary}

    def github_requirement_retry_node(self, state: SupervisorState) -> SupervisorState:
        """Handle retries for invalid requirement inputs"""

        requirements_config = {
            "use_case_name": {
                "prompt": "What is the name of the use case for the repository?"
            },
            "template": {
                "prompt": "Which template would you like to use?\nOptions: npus-aiml-mlops-stacks-template, npus-aiml-skinny-dab-template"
            },
            "internal_team": {
                "prompt": "What is the internal team name?\nOptions: eai, deloitte, sig, tredence, bora, genpact, tiger, srm, kroger"
            },
            "development_team": {
                "prompt": "What is the development team name?\nOptions: eai, deloitte, sig, tredence, bora, genpact, tiger, srm, kroger, none"
            },
            "additional_team": {
                "prompt": "What is the additional team?\nOptions: eai, deloitte, sig, tredence, bora, genpact, tiger, srm, kroger, none"
            },
        }

        current_requirement = state["current_requirement"]
        retry_count = state.get("requirement_retry_count", 0) + 1

        if retry_count >= 2:
            # After 2 retries, offer to go back to main menu
            prompt = f"I'm having trouble understanding your input for {current_requirement}. Would you like to go back to the main menu? (yes/no)"
            return {
                "prompt": prompt,
                "current_step": "confirm_back_to_menu",
                "requirement_retry_count": retry_count,
            }
        else:
            prompt = f"Invalid input for {current_requirement}. Please try again.\n\n{requirements_config[current_requirement]['prompt']}"
            return {"prompt": prompt, "requirement_retry_count": retry_count}

    def should_validate_requirement(self, state: SupervisorState) -> str:
        """Route based on validation result"""
        validation_result = state.get("validation_result", False)
        retry_count = state.get("requirement_retry_count", 0)

        if validation_result:
            return "next_requirement"
        elif retry_count >= 2:
            return "back_to_menu_check"
        else:
            return "retry_requirement"

    def should_go_back_to_menu(self, state: SupervisorState) -> str:
        """Check if user wants to go back to main menu"""
        user_input = state.get("user_request", "").strip().lower()

        if user_input in ["yes", "y", "yeah", "sure"]:
            return "back_to_greeter"
        else:
            return "continue_requirements"

    def route_from_human(self, state: SupervisorState) -> str:
        """Route from human node based on current step"""
        current_step = state.get("current_step", "")

        if current_step == "confirm_back_to_menu":
            return self.should_go_back_to_menu(state)
        elif current_step == "github_requirements":
            return "github_validator"
        else:
            return "intent_analyzer"

    def route_from_github_retry(self, state: SupervisorState) -> str:
        """Route from github retry based on current step"""
        current_step = state.get("current_step", "")

        if current_step == "confirm_back_to_menu":
            return "human_menu_confirm"
        else:
            return "human_requirements"

    def route_from_github_next(self, state: SupervisorState) -> str:
        """Route from github next based on completion status"""
        current_step = state.get("current_step", "")

        if current_step == "github_requirements_complete":
            return "completion"
        else:
            return "human_requirements"

    def build_graph(self) -> StateGraph:
        """Build and return the LangGraph"""
        workflow = StateGraph(SupervisorState)

        # Add existing nodes
        workflow.add_node("greeter", self.greeter_node)
        workflow.add_node("human", self.human_node)
        workflow.add_node("human_requirements", self.human_node)
        workflow.add_node("human_menu_confirm", self.human_node)
        workflow.add_node("intent_analyzer", self.intent_analyzer_node)
        workflow.add_node("retry_handler", self.retry_handler_node)
        workflow.add_node("completion", self.completion_node)

        # Add new GitHub requirement nodes
        workflow.add_node("github_requirements", self.github_requirements_node)
        workflow.add_node("github_validator", self.github_requirement_validator_node)
        workflow.add_node("github_next", self.github_requirement_next_node)
        workflow.add_node("github_retry", self.github_requirement_retry_node)

        # Add existing edges
        workflow.add_edge(START, "greeter")
        workflow.add_edge("greeter", "human")
        workflow.add_edge("human", "intent_analyzer")

        # Modified conditional edge for retry logic
        workflow.add_conditional_edges(
            "intent_analyzer",
            self.should_retry,
            {
                "retry": "retry_handler",
                "complete": "completion",
                "github_flow": "github_requirements",
            },
        )

        workflow.add_edge("retry_handler", "human")
        workflow.add_edge("completion", END)

        # New GitHub requirement flow edges
        workflow.add_edge("github_requirements", "human_requirements")
        workflow.add_edge("human_requirements", "github_validator")

        workflow.add_conditional_edges(
            "github_validator",
            self.should_validate_requirement,
            {
                "next_requirement": "github_next",
                "retry_requirement": "github_retry",
                "back_to_menu_check": "github_retry",
            },
        )

        workflow.add_conditional_edges(
            "github_retry",
            self.route_from_github_retry,
            {
                "human_menu_confirm": "human_menu_confirm",
                "human_requirements": "human_requirements",
            },
        )

        workflow.add_conditional_edges(
            "human_menu_confirm",
            self.should_go_back_to_menu,
            {
                "back_to_greeter": "greeter",
                "continue_requirements": "human_requirements",
            },
        )

        workflow.add_conditional_edges(
            "github_next",
            self.route_from_github_next,
            {"completion": "completion", "human_requirements": "human_requirements"},
        )

        # Add checkpointer for human-in-the-loop
        checkpointer = InMemorySaver()
        return workflow.compile(checkpointer=checkpointer)


# Example usage:
if __name__ == "__main__":
    supervisor = SupervisorGraph()
    graph = supervisor.build_graph()

    # Initialize state with ALL required variables
    initial_state = {
        "user_request": "",
        "intent": "",
        "messages": [],
        "retry_count": 0,
        "prompt": "",
        "current_step": "start",
        # GitHub requirements state variables
        "github_requirements": {},
        "current_requirement": "",
        "validation_result": False,
        "requirement_retry_count": 0,
    }

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Run the graph until the first interrupt
    print("Starting graph execution...")
    result = graph.invoke(initial_state, config=config)

    # Handle interrupts in a loop to support the full GitHub requirements flow
    while "__interrupt__" in result:
        print("Graph interrupted for human input:")
        interrupt_data = result["__interrupt__"][0]
        print(f"Interrupt data: {interrupt_data.value}")

        # Get user input
        prompt_text = interrupt_data.value.get("prompt", "Please provide your input:")
        user_response = input(f"{prompt_text}\nYour response: ")

        # Resume the graph with human input
        print(f"Resuming with user input: {user_response}")
        result = graph.invoke(Command(resume=user_response), config=config)

    print("Graph execution completed!")
    print(f"Final result: {result}")

    # Display final GitHub requirements if they were collected
    if result.get("github_requirements"):
        print("\n=== GitHub Repository Requirements ===")
        for req, value in result["github_requirements"].items():
            print(f"{req.replace('_', ' ').title()}: {value}")

    # Display the graph structure
    try:
        print("\nGraph structure:")
        print(graph.get_graph().draw_ascii())
    except Exception as e:
        print(f"Could not display graph structure: {e}")
