import re
import json
import uuid
import asyncio
from typing import Any, Dict, Union
from pydantic import BaseModel
from dotenv import load_dotenv
from langgraph.constants import START
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from typing import TypedDict, Annotated, Optional
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_litellm import ChatLiteLLM
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


# GitHub Requirements Structured Response
class GitHubRequirementsResponse(BaseModel):
    requirements: dict
    next_action: str
    response: str


# Intent Response (from previous step)
class IntentResponse(BaseModel):
    intent: str
    next_action: str
    response_message: str


# State structure
class SupervisorState(TypedDict):
    user_request: str
    intent: str
    next_action: str
    conversation_history: list
    current_step: str
    github_requirements: dict


def extract_llm_json(resp: Any, msg_index: int = 1) -> Dict[str, Any]:
    """
    Extract JSON content from an LLM response object that may wrap output in various formats.

    Args:
        resp (Any): The LLM response (from SDK).
        msg_index (int): Which message index to pull "content" from. Defaults to 1.

    Returns:
        Dict[str, Any]: Parsed JSON as a Python dictionary with attribute-style access.

    Raises:
        ValueError: If no valid JSON can be extracted from the response.
    """

    class AttrDict(dict):
        """Dict with attribute-style access"""

        def __getattr__(self, item):
            val = self.get(item)
            if isinstance(val, dict):
                return AttrDict(val)
            return val

        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    def clean_and_extract_json(text: str) -> str:
        """Extract and clean JSON from various text formats."""
        text = text.strip()

        # Try to extract JSON from markdown code fences
        # Pattern matches ```json...``` or ```...``` blocks
        code_fence_patterns = [
            r"```json\s*\n?(.*?)\n?```",
            r"```\s*\n?(.*?)\n?```",
        ]

        for pattern in code_fence_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # If no code fence found, try to find JSON-like content
        # Look for content between first { and last }
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            return brace_match.group(0)

        # Look for content between first [ and last ]
        bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
        if bracket_match:
            return bracket_match.group(0)

        # If still no match, return the original text
        return text

    def parse_key_value_pairs(text: str) -> Dict[str, Any]:
        """Parse simple key-value pair format into dictionary."""
        result = {}

        # Split by lines and process each
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):  # Skip empty lines and comments
                continue

            # Try different separators
            for separator in [":", "=", "->"]:
                if separator in line:
                    key, value = line.split(separator, 1)
                    key = key.strip().strip("\"'")
                    value = value.strip().strip("\"'")

                    # Try to convert value to appropriate type
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    elif value.lower() == "null" or value.lower() == "none":
                        value = None
                    elif value.isdigit():
                        value = int(value)
                    elif re.match(r"^\d+\.\d+$", value):
                        value = float(value)

                    result[key] = value
                    break

        return result

    try:
        # Get the raw content
        print(resp["messages"][msg_index].content)
        raw = resp["messages"][msg_index].content

        # Clean and extract JSON
        cleaned_json = clean_and_extract_json(raw)

        # Try to parse as JSON
        try:
            parsed = json.loads(cleaned_json)
            return AttrDict(parsed)
        except json.JSONDecodeError:
            # If JSON parsing fails, try key-value pair parsing
            kv_result = parse_key_value_pairs(cleaned_json)
            if kv_result:
                return AttrDict(kv_result)

            # If that also fails, try a more lenient JSON parsing
            # Remove common issues like trailing commas, single quotes, etc.
            cleaned_json = re.sub(
                r",(\s*[}\]])", r"\1", cleaned_json
            )  # Remove trailing commas
            cleaned_json = re.sub(
                r"'", '"', cleaned_json
            )  # Replace single quotes with double quotes

            try:
                parsed = json.loads(cleaned_json)
                return AttrDict(parsed)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Could not parse JSON from response: {e}\nContent: {cleaned_json[:200]}..."
                )

    except (KeyError, IndexError, AttributeError) as e:
        raise ValueError(f"Could not access response content at index {msg_index}: {e}")


class ModernizedSupervisorGraph:
    def __init__(self):
        self.valid_intents = ["github_repo", "databricks_schema", "databricks_compute"]

    def human_node(self, state: SupervisorState) -> SupervisorState:
        """Handle human input for any conversation step"""
        conversation_history = state.get("conversation_history", [])
        current_step = state.get("current_step", "intent_capture")

        if not conversation_history:
            # First interaction
            prompt = """Hello! I'm your AI Operations assistant. I can help you with:
1. Create a GitHub repository
2. Set up a Databricks schema  
3. Create a Databricks compute cluster
What would you like to do today?"""
            conversation_history.append({"role": "AI Assistant", "content": prompt})
        else:
            # Show the agent's last response
            last_agent_message = None
            for msg in reversed(conversation_history):
                if msg["role"] == "AI Assistant":
                    last_agent_message = msg["content"]
                    break

            if last_agent_message:
                prompt = f"🤖 Assistant: {last_agent_message}\n\nYour response:"
            else:
                prompt = "Please tell me what you'd like to do:"

        user_input = interrupt({"prompt": prompt, "step": current_step})

        return {
            "user_request": user_input,
            "conversation_history": conversation_history,
        }

    def intent_capture_agent_node(self, state: SupervisorState) -> SupervisorState:
        """Intent capture using react agent"""
        user_request = state["user_request"]
        conversation_history = state.get("conversation_history", [])
        conversation_context = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in conversation_history[-3:]]
        )
        self.intent_agent = create_react_agent(
            model="gpt-3.5-turbo",
            tools=[],
            prompt=f"""You are an AI Operations assistant. Analyze the user's request and classify it into one of the following services:

1. "github_repo" → Creating GitHub repositories
2. "databricks_schema" → Setting up Databricks schemas
3. "databricks_compute" → Creating Databricks compute clusters

Rules:
 -> Always return a JSON object with the keys: "intent", "next_action", "response_message".
 ->  "intent" must be one of ["github_repo", "databricks_schema", "databricks_compute", "unclear"].
 ->  "next_action" must be:
   - "proceed" if the intent is clear,
   - "clarify" if the intent is unclear.
 ->  "response_message" should be a natural, helpful response:
   - If intent is clear: acknowledge the specific service. Example:
       "intent": "github_repo",
       "next_action": "proceed",
       "response_message": "Great! You are looking to set up a GitHub repository."
   - If the user’s request is unrelated: acknowledge what they said and guide them back saying u cant help with that particular request but here is what u can do.
 -> Always enclose keys and values in double quotes, and wrap the output in curly braces.
 -> Do not add extra text outside the JSON.

Previous Conversation History
```
 {conversation_context}
```
""",
        )

        response = self.intent_agent.invoke(
            {"messages": [{"role": "user", "content": user_request}]}
        )
        intent_analysis = extract_llm_json(response)

        if intent_analysis.intent == "github_repo":
            conversation_history.clear()
            conversation_history.append(
                {
                    "role": "SUPERVISOR",
                    "content": "User want to create a git repository",
                }
            )
        elif intent_analysis.intent == "databricks_schema":
            conversation_history.clear()
            conversation_history.append(
                {
                    "role": "SUPERVISOR",
                    "content": "User want to create a databricks schema",
                }
            )
        elif intent_analysis.intent == "databricks_compute":
            conversation_history.clear()
            conversation_history.append(
                {
                    "role": "SUPERVISOR",
                    "content": "User want to create a databricks compute",
                }
            )
        else:
            conversation_history.append({"role": "user", "content": user_request})
            conversation_history.append(
                {"role": "AI Assistant", "content": intent_analysis.response_message}
            )
        return {
            "intent": intent_analysis.intent,
            "next_action": intent_analysis.next_action,
            "conversation_history": conversation_history,
            "current_step": (
                "intent_captured"
                if intent_analysis.next_action == "proceed"
                else "intent_clarification"
            ),
        }

    def route_from_human(self, state: SupervisorState) -> str:
        """Route from human node based on current step - UPDATED"""
        current_step = state.get("current_step", "")

        if current_step == "github_gathering":
            return "github_repo_requirements"
        elif current_step in "databricks_schema_gathering":
            return "databricks_schema_requirements"
        elif current_step in "databricks_compute_gathering":
            return "databricks_compute_requirements"
        else:
            return "intent_analyzer"

    def route_from_intent(self, state: SupervisorState) -> str:
        """Route after intent is captured"""
        intent = state.get("intent", "unclear")
        next_action = state.get("next_action", "clarify")

        if next_action == "proceed" and intent == "github_repo":
            return "github_flow"
        elif next_action == "proceed" and intent == "databricks_schema":
            return "databricks_schema_flow"
        elif next_action == "proceed" and intent == "databricks_compute":
            return "databricks_compute_flow"
        else:
            return "continue"

    def github_requirements_agent_node(self, state: SupervisorState) -> SupervisorState:
        """GitHub requirements gathering using intelligent agent"""
        user_request = state["user_request"]
        conversation_history = state.get("conversation_history", [])
        github_requirements = state.get("github_requirements", {})
        example_output_format = {
            "requirements_gathered": {...},
            "next_action": "...",
            "response_message": "...",
        }
        conversation_context = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in conversation_history[-10:]]
        )

        chat = ChatLiteLLM(model="gpt-3.5-turbo")
        system_message = f"""You are a GitHub repository Setup specialist. Your job is to gather ALL required information for creating a GitHub repository.

RULES:
    - Always respond ONLY with a JSON object. Do not include any text outside the JSON.
    - When a user wants to create a repo, introduce yourself and explicitly list all required fields with the available options inside the "response_message".
    - Ask the user to provide their inputs for the required fields. The user can provide one or multiple values at a time.
    - If the user provides a value for a specific field, attach it to that field in your requirements_gathered.
    - If the user provides multiple values and it is unclear which field each value corresponds to, ask a clarifying follow-up question to assign each value correctly.
    - The JSON response MUST follow this structure:
        {example_output_format}
    - "requirements_gathered": a dictionary containing the keys as the required fields and the values as the inputs provided by the user so far. Refer to previous requirements gathered so you do not miss what was gathered before. If no input gathered then set as empty dictionary {{}}.
    - "next_action": either "continue_gathering" if not all requirements are provided, "get_user_confirmation" if all are gathered, or "user_confirmed" if all are gathered and user has explicitly confirmed.
    - "response_message": a helpful natural language message guiding the user on next steps.
    - If user doesn’t provide confirmation at "get_user_confirmation", continue asking which field they want to update. Once they explicitly confirm that values are final, set next_action to "user_confirmed".

REQUIRED FIELDS:
    "use_case_name": Name for the repo (kebab-case, e.g., test-repo-name)
    "template": Choose from ["npus-aiml-mlops-stacks-template", "npus-aiml-skinny-dab-template"]
    "internal_team": Choose from ["eai", "deloitte", "sig", "tredence", "bora", "genpact", "tiger", "srm", "kroger"]
    "development_team": Choose from ["eai", "deloitte", "sig", "tredence", "bora", "genpact", "tiger", "srm", "kroger", "none"]
    "additional_team": Choose from ["eai", "deloitte", "sig", "tredence", "bora", "genpact", "tiger", "srm", "kroger", "none"]

Previous Requirements Gathered
{github_requirements}

Conversation History
{conversation_context}
"""
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_request),
        ]
        response = chat.invoke(messages)

        github_analysis = extract_llm_json({"messages": [0, response]})

        # Update conversation history
        conversation_history.append({"role": "user", "content": user_request})
        conversation_history.append(
            {"role": "AI Assistant", "content": github_analysis.response_message}
        )

        return {
            "github_requirements": github_analysis.requirements_gathered,
            "next_action": github_analysis.next_action,
            "conversation_history": conversation_history,
            "current_step": (
                "github_repo_completed"
                if github_analysis.next_action == "user_confirmed"
                else "github_gathering"
            ),
        }

    def databricks_schema_agent_node(self, state: SupervisorState) -> SupervisorState:
        return {"current_step": "databricks_schema_completed"}

    def databricks_compute_agent_node(self, state: SupervisorState) -> SupervisorState:
        return {"current_step": "databricks_compute_completed"}

    def route_worflows(self, state: SupervisorState) -> str:
        """Route within GitHub requirements flow"""
        current_step = state.get("current_step", "")

        if current_step in [
            "github_repo_completed",
            "databricks_schema_completed",
            "databricks_compute_completed",
        ]:
            return "complete"
        else:
            return "continue"

    def completion_node(self, state: SupervisorState) -> SupervisorState:
        """Final completion with results"""
        intent = state.get("intent", "unclear")
        github_requirements = state.get("github_requirements", {})

        if intent == "github_repo" and github_requirements:
            final_message = "🎉 GitHub repository requirements completed!"
            print(f"\n{final_message}")
            print("📋 FINAL REQUIREMENTS:")
            for field, value in github_requirements.items():
                print(f"  • {field.replace('_', ' ').title()}: {value}")
        else:
            final_message = f"✅ Session completed with intent: {intent}"
            print(f"\n{final_message}")

        return {
            "current_step": "completed",
            "intent": intent,
            "github_requirements": github_requirements,
        }

    def build_graph(self) -> StateGraph:
        """Build the complete modernized graph"""
        workflow = StateGraph(SupervisorState)

        # Add nodes
        workflow.add_node("human", self.human_node)
        workflow.add_node("intent_agent", self.intent_capture_agent_node)
        workflow.add_node("github_repo_agent", self.github_requirements_agent_node)
        workflow.add_node("databricks_schema_agent", self.databricks_schema_agent_node)
        workflow.add_node(
            "databricks_compute_agent", self.databricks_compute_agent_node
        )
        workflow.add_node("completion", self.completion_node)

        # Build the flow
        workflow.add_edge(START, "human")
        workflow.add_conditional_edges(
            "human",
            self.route_from_human,
            {
                "intent_analyzer": "intent_agent",
                "github_repo_requirements": "github_repo_agent",
                "databricks_schema_requirements": "databricks_schema_agent",
                "databricks_compute_requirements": "databricks_compute_agent",
            },
        )

        # Route after intent capture
        workflow.add_conditional_edges(
            "intent_agent",
            self.route_from_intent,
            {
                "continue": "human",
                "github_flow": "github_repo_agent",
                "databricks_schema_flow": "databricks_schema_agent",
                "databricks_compute_flow": "databricks_compute_agent",
            },
        )

        # GitHub flow routing
        workflow.add_conditional_edges(
            "github_repo_agent",
            self.route_worflows,
            {
                "continue": "human",
                "complete": "completion",
            },
        )

        # Databricks Schema routing
        workflow.add_conditional_edges(
            "databricks_schema_agent",
            self.route_worflows,
            {
                "continue": "human",
                "complete": "completion",
            },
        )

        workflow.add_conditional_edges(
            "databricks_compute_agent",
            self.route_worflows,
            {
                "continue": "human",
                "complete": "completion",
            },
        )

        workflow.add_edge("completion", END)

        checkpointer = InMemorySaver()
        return workflow.compile(checkpointer=checkpointer)


# Example usage:
if __name__ == "__main__":
    supervisor = ModernizedSupervisorGraph()
    graph = supervisor.build_graph()

    initial_state = {
        "user_request": "",
        "intent": "",
        "next_action": "",
        "conversation_history": [],
        "current_step": "start",
        "github_requirements": {},
    }

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("🚀 Starting AI Operations Assistant...")
    result = graph.invoke(initial_state, config=config)

    # Handle interrupts
    while "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0]
        prompt_text = interrupt_data.value.get("prompt", "Please provide your input:")
        user_response = input(f"\n{prompt_text}\n> ")

        result = graph.invoke(Command(resume=user_response), config=config)

    print("\n✅ Session completed!")

    # Show final results
    if result.get("github_requirements"):
        print("\n🎯 FINAL GITHUB REQUIREMENTS:")
        for field, value in result["github_requirements"].items():
            print(f"  {field}: {value}")

    try:
        print("\n📊 Graph structure:")
        print(graph.get_graph().draw_ascii())
    except Exception as e:
        print(f"Could not display graph structure: {e}")
