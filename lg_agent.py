import asyncio
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

client = MultiServerMCPClient(
    {
        "automation": {
            "command": "python",
            "args": [
                "/Users/mayurgd/Documents/CodingSpace/aiops_infra_automation/servers/aiops_automation_server.py"
            ],
            "transport": "stdio",
        }
    }
)


async def get_tools_async():
    tools = await client.get_tools()
    return tools


def get_tools():
    tools = asyncio.run(get_tools_async())
    return tools


LOCAL = True


# Define the state structure
class SupervisorState(TypedDict):
    user_request: str
    intent: str
    messages: Annotated[list, add_messages]
    retry_count: int
    prompt: str
    current_step: str
    # GitHub requirements
    github_requirements: dict
    current_requirement: str
    validation_result: bool
    requirement_retry_count: int
    confirmation_step: str


class SupervisorGraph:
    def __init__(self):
        self.model = "gpt-4o-mini"
        self.max_retries = 3
        self.tools = get_tools()

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

    def github_requirements_collector_node(
        self, state: SupervisorState
    ) -> SupervisorState:
        """Single node to collect all GitHub requirements with validation"""

        requirements_config = {
            "use_case_name": {
                "prompt": "What is the name of the use case for the repository?",
                "validation": "any_text",  # Any non-empty text
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

        requirements_order = [
            "use_case_name",
            "template",
            "internal_team",
            "development_team",
            "additional_team",
        ]

        github_requirements = state.get("github_requirements", {})
        current_requirement = state.get("current_requirement", "")
        user_input = state.get("user_request", "")
        retry_count = state.get("requirement_retry_count", 0)

        if state.get("current_step") == "github_confirmation":
            confirmation_step = state.get("confirmation_step", "")

            if confirmation_step == "awaiting_confirmation":
                user_response = user_input.strip().lower()

                if user_response in ["yes", "y", "yeah", "sure", "ok", "okay", "fine"]:
                    # User confirmed - complete the flow
                    final_summary = "🎉 GitHub repository requirements confirmed!\n\n📋 FINAL SUMMARY:\n"
                    for req, value in github_requirements.items():
                        final_summary += f"• {req.replace('_', ' ').title()}: {value}\n"
                    final_summary += "\n✅ Ready to proceed with repository creation!"

                    return {
                        "github_requirements": github_requirements,
                        "current_step": "github_requirements_complete",
                        "prompt": final_summary,
                    }

                elif user_response in ["no", "n", "nope", "not really"]:
                    # User wants to make changes
                    prompt = (
                        "Which field would you like to update?\n"
                        "Available fields: use_case_name, template, internal_team, development_team, additional_team\n"
                        "Please specify the field name:"
                    )

                    return {
                        "current_step": "github_confirmation",
                        "confirmation_step": "field_selection",
                        "prompt": prompt,
                    }
                else:
                    # Invalid confirmation response
                    prompt = "Please respond with 'yes' if you're satisfied with the requirements, or 'no' if you'd like to make changes:"
                    return {
                        "current_step": "github_confirmation",
                        "confirmation_step": "awaiting_confirmation",
                        "prompt": prompt,
                    }

            elif confirmation_step == "field_selection":
                # User specified which field to update
                field_to_update = user_input.strip().lower().replace(" ", "_")

                if field_to_update in requirements_order:
                    # Valid field - ask for new value
                    req_config = requirements_config[field_to_update]
                    prompt = f"Current value for {field_to_update.replace('_', ' ').title()}: {github_requirements.get(field_to_update, 'Not set')}\n\n"
                    prompt += f"Please provide the new value:\n{req_config['prompt']}"

                    return {
                        "current_step": "github_confirmation",
                        "confirmation_step": "updating_field",
                        "current_requirement": field_to_update,  # Reuse this field
                        "prompt": prompt,
                        "requirement_retry_count": 0,
                    }
                else:
                    # Invalid field
                    prompt = (
                        "Invalid field name. Please choose from:\n"
                        f"{', '.join([f.replace('_', ' ').title() for f in requirements_order])}"
                    )

                    return {
                        "current_step": "github_confirmation",
                        "confirmation_step": "field_selection",
                        "prompt": prompt,
                    }

            elif confirmation_step == "updating_field":
                # User provided new value for the field
                field_to_update = current_requirement
                req_config = requirements_config[field_to_update]

                # Validate the new value (reuse existing validation logic)
                is_valid = False
                validated_value = ""

                if req_config.get("validation") == "any_text":
                    if user_input.strip():
                        is_valid = True
                        validated_value = user_input.strip()
                else:
                    response = completion(
                        model=self.model,
                        messages=[
                            {
                                "role": "user",
                                "content": f"""
                            Question: {req_config['prompt']}
                            User Input: "{user_input}"
                            Valid Options: {req_config['valid_options']}
                            
                            Task: Validate if the user input matches any valid option (case-insensitive).
                            
                            Respond with EXACTLY one of:
                            - If valid: return the exact valid option (lowercase)
                            - If invalid: return "INVALID"
                            """,
                            }
                        ],
                    )

                    validation_result = (
                        response["choices"][0]["message"]["content"].strip().lower()
                    )

                    if (
                        validation_result != "invalid"
                        and validation_result in req_config["valid_options"]
                    ):
                        is_valid = True
                        validated_value = validation_result

                if is_valid:
                    # Update the requirement and show updated summary
                    github_requirements[field_to_update] = validated_value

                    summary = (
                        f"✅ Updated {field_to_update.replace('_', ' ').title()} to: {validated_value}\n\n"
                        "📋 UPDATED SUMMARY:\n"
                    )
                    for req, value in github_requirements.items():
                        summary += f"• {req.replace('_', ' ').title()}: {value}\n"

                    summary += (
                        "\n🔍 Are you satisfied with these requirements now? (yes/no)\n"
                    )
                    summary += "If no, please specify which field you'd like to update."

                    return {
                        "github_requirements": github_requirements,
                        "current_step": "github_confirmation",
                        "confirmation_step": "awaiting_confirmation",
                        "prompt": summary,
                    }
                else:
                    # Invalid new value
                    retry_count = state.get("requirement_retry_count", 0) + 1

                    if retry_count >= 2:
                        prompt = f"❌ Multiple invalid attempts. Going back to confirmation.\n\nCurrent requirements:\n"
                        for req, value in github_requirements.items():
                            prompt += f"• {req.replace('_', ' ').title()}: {value}\n"
                        prompt += (
                            "\nAre you satisfied with these requirements? (yes/no)"
                        )

                        return {
                            "current_step": "github_confirmation",
                            "confirmation_step": "awaiting_confirmation",
                            "prompt": prompt,
                            "requirement_retry_count": 0,
                        }
                    else:
                        prompt = f"❌ Invalid input: '{user_input}'\n\nPlease try again:\n{req_config['prompt']}"
                        return {
                            "current_step": "github_confirmation",
                            "confirmation_step": "updating_field",
                            "prompt": prompt,
                            "requirement_retry_count": retry_count,
                        }

        # If no current requirement, start with first
        if not current_requirement:
            current_requirement = requirements_order[0]
            prompt = f"Let's gather the GitHub repository requirements.\n\n{requirements_config[current_requirement]['prompt']}"
            return {
                "prompt": prompt,
                "current_step": "github_requirements",
                "current_requirement": current_requirement,
                "requirement_retry_count": 0,
            }

        # If we have user input, validate it
        if user_input:
            req_config = requirements_config[current_requirement]
            is_valid = False
            validated_value = ""

            if req_config.get("validation") == "any_text":
                # For use_case_name, any non-empty string is valid
                if user_input.strip():
                    is_valid = True
                    validated_value = user_input.strip()
            else:
                # Use LLM for validation with single call
                response = completion(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": f"""
                        Question: {req_config['prompt']}
                        User Input: "{user_input}"
                        Valid Options: {req_config['valid_options']}
                        
                        Task: Validate if the user input matches any valid option (case-insensitive).
                        
                        Respond with EXACTLY one of:
                        - If valid: return the exact valid option (lowercase)
                        - If invalid: return "INVALID"
                        
                        Examples:
                        Input "EAI" with options ["eai", "sig"] → "eai"
                        Input "xyz" with options ["eai", "sig"] → "INVALID"
                        """,
                        }
                    ],
                )

                validation_result = (
                    response["choices"][0]["message"]["content"].strip().lower()
                )

                if (
                    validation_result != "invalid"
                    and validation_result in req_config["valid_options"]
                ):
                    is_valid = True
                    validated_value = validation_result

            if is_valid:
                # Store the validated value and move to next requirement
                github_requirements[current_requirement] = validated_value

                # Find next requirement
                current_index = requirements_order.index(current_requirement)

                if current_index < len(requirements_order) - 1:
                    # Move to next requirement
                    next_requirement = requirements_order[current_index + 1]
                    prompt = f"✓ {current_requirement}: {validated_value}\n\nNext requirement:\n{requirements_config[next_requirement]['prompt']}"

                    return {
                        "prompt": prompt,
                        "current_requirement": next_requirement,
                        "github_requirements": github_requirements,
                        "requirement_retry_count": 0,
                        "current_step": "github_requirements",
                    }
                else:
                    # All requirements collected - SHOW CONFIRMATION INSTEAD OF COMPLETING
                    summary = "✅ All GitHub repository requirements collected!\n\n📋 SUMMARY:\n"
                    for req, value in github_requirements.items():
                        summary += f"• {req.replace('_', ' ').title()}: {value}\n"

                    summary += "\n🔍 Please review the above details. Are you satisfied with these requirements? (yes/no)\n"
                    summary += "If no, please specify which field you'd like to update."

                    return {
                        "github_requirements": github_requirements,
                        "current_step": "github_confirmation",  # NEW STEP
                        "confirmation_step": "awaiting_confirmation",
                        "prompt": summary,
                    }
            else:
                # Invalid input, retry
                retry_count += 1

                if retry_count >= 2:
                    prompt = f"❌ I'm having trouble with your input for {current_requirement}.\nWould you like to go back to the main menu? (yes/no)"
                    return {
                        "prompt": prompt,
                        "current_step": "confirm_back_to_menu",
                        "requirement_retry_count": retry_count,
                    }
                else:
                    prompt = f"❌ Invalid input: '{user_input}'\n\nPlease try again:\n{req_config['prompt']}"
                    return {
                        "prompt": prompt,
                        "requirement_retry_count": retry_count,
                        "current_step": "github_requirements",
                    }

        # Should not reach here, but return current state
        return state

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
        """Route from human node based on current step - UPDATED"""
        current_step = state.get("current_step", "")

        if current_step == "github_requirements":
            return "github_requirements"
        elif current_step == "github_confirmation":  # NEW: Handle confirmation step
            return "github_requirements"
        else:
            return "intent_analyzer"

    def route_github_requirements(self, state: SupervisorState) -> str:
        """Enhanced routing for GitHub requirements including confirmation"""
        current_step = state.get("current_step", "")

        if current_step == "github_requirements_complete":
            return "complete"
        elif current_step == "github_confirmation":
            return "continue"  # Stay in the loop for confirmation workflow
        elif current_step == "confirm_back_to_menu":
            user_input = state.get("user_request", "").strip().lower()
            if user_input in ["yes", "y", "yeah", "sure"]:
                return "back_to_menu"
            else:
                return "continue"
        else:
            return "continue"

    def build_graph(self) -> StateGraph:
        """Build and return the simplified LangGraph"""
        workflow = StateGraph(SupervisorState)

        # Add nodes
        workflow.add_node("greeter", self.greeter_node)
        workflow.add_node("human", self.human_node)
        workflow.add_node("intent_analyzer", self.intent_analyzer_node)
        workflow.add_node("retry_handler", self.retry_handler_node)
        workflow.add_node("completion", self.completion_node)

        # Single node for GitHub requirements
        workflow.add_node(
            "github_requirements", self.github_requirements_collector_node
        )

        # Add edges
        workflow.add_edge(START, "greeter")
        workflow.add_edge("greeter", "human")

        # Route from human based on current step
        workflow.add_conditional_edges(
            "human",
            self.route_from_human,
            {
                "intent_analyzer": "intent_analyzer",
                "github_requirements": "github_requirements",
            },
        )

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

        # Simplified GitHub flow - single loop
        workflow.add_conditional_edges(
            "github_requirements",
            self.route_github_requirements,
            {
                "continue": "human",  # Continue collecting requirements
                "complete": "completion",  # All done
                "back_to_menu": "greeter",  # User wants to go back
            },
        )

        workflow.add_edge("completion", END)

        checkpointer = InMemorySaver()
        return workflow.compile(checkpointer=checkpointer)


# Example usage:
if __name__ == "__main__":
    supervisor = SupervisorGraph()
    graph = supervisor.build_graph()

    # Simplified initial state - only core variables needed
    initial_state = {
        "user_request": "",
        "intent": "",
        "messages": [],
        "retry_count": 0,
        "prompt": "",
        "current_step": "start",
        # GitHub requirements handled internally, but initialize for safety
        "github_requirements": {},
        "current_requirement": "",
        "requirement_retry_count": 0,
    }

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("Starting graph execution...")
    result = graph.invoke(initial_state, config=config)

    # Handle interrupts - much simpler now
    while "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0]
        prompt_text = interrupt_data.value.get("prompt", "Please provide your input:")
        user_response = input(f"\n{prompt_text}\n> ")

        result = graph.invoke(Command(resume=user_response), config=config)

    print("\n✅ Graph execution completed!")

    # Display final results
    if result.get("github_requirements"):
        print("\n" + "=" * 50)
        print("🚀 GITHUB REPOSITORY REQUIREMENTS")
        print("=" * 50)
        for req, value in result["github_requirements"].items():
            print(f"📋 {req.replace('_', ' ').title()}: {value}")
        print("=" * 50)
    # Display the graph structure
    try:
        print("\nGraph structure:")
        print(graph.get_graph().draw_ascii())
    except Exception as e:
        print(f"Could not display graph structure: {e}")
