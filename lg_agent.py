import re
import json
import uuid
import asyncio
from typing import Any
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

load_dotenv()


# GitHub Requirements Structured Response
class GitHubRequirementsResponse(BaseModel):
    use_case_name: Optional[str] = None
    template: Optional[str] = None
    internal_team: Optional[str] = None
    development_team: Optional[str] = None
    additional_team: Optional[str] = None
    status: str  # "gathering", "confirming", "completed"
    next_action: str  # "ask_question", "confirm", "complete"
    response_message: str
    missing_fields: list[str] = []
    all_requirements_collected: bool = False


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


def extract_llm_json(resp: Any, msg_index: int = 1) -> dict:
    """
    Extract JSON content from an LLM response object that may wrap output in markdown code fences.

    Args:
        resp (Any): The LLM response (from SDK).
        msg_index (int): Which message index to pull "content" from. Defaults to 1.

    Returns:
        dict: Parsed JSON as a Python dictionary.
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

    print(resp["messages"][msg_index].content)
    raw = resp["messages"][msg_index].content
    return AttrDict(json.loads(raw))


class ModernizedSupervisorGraph:
    def __init__(self):
        self.valid_intents = ["github_repo", "databricks_schema", "databricks_compute"]

        # GitHub requirements gathering agent
        self.github_agent = create_react_agent(
            model="gpt-4o-mini",
            tools=[],
            response_format=GitHubRequirementsResponse,
        )

    def human_node(self, state: SupervisorState) -> SupervisorState:
        """Handle human input for any conversation step"""
        conversation_history = state.get("conversation_history", [])
        current_step = state.get("current_step", "intent_capture")

        if not conversation_history:
            # First interaction
            prompt = """
            Hello! I'm your AI Operations assistant. I can help you with:

            1. Create a GitHub repository
            2. Set up a Databricks schema  
            3. Create a Databricks compute cluster

            What would you like to do today?
            """
            conversation_history.append({"role": "assistant", "content": prompt})
        else:
            # Show the agent's last response
            last_agent_message = None
            for msg in reversed(conversation_history):
                if msg["role"] == "assistant":
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

        conversation_history.append({"role": "user", "content": user_request})
        conversation_history.append(
            {"role": "assistant", "content": intent_analysis.response_message}
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
        user_confirmed = state.get("user_confirmed", False)

        conversation_context = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in conversation_history[-10:]]
        )

        # Create comprehensive prompt for the GitHub agent
        agent_prompt = f"""You are a GitHub repository setup specialist. Your job is to gather ALL required information for creating a GitHub repository.

REQUIRED FIELDS:
1. use_case_name: Name of the use case (any text)
2. template: Must be exactly one of: "npus-aiml-mlops-stacks-template" or "npus-aiml-skinny-dab-template"
3. internal_team: Must be exactly one of: "eai", "deloitte", "sig", "tredence", "bora", "genpact", "tiger", "srm", "kroger"
4. development_team: Must be exactly one of: "eai", "deloitte", "sig", "tredence", "bora", "genpact", "tiger", "srm", "kroger", "none"
5. additional_team: Must be exactly one of: "eai", "deloitte", "sig", "tredence", "bora", "genpact", "tiger", "srm", "kroger", "none"

CURRENT STATE:
{github_requirements}

USER CONFIRMATION STATUS: {user_confirmed}

CONVERSATION CONTEXT:
{conversation_context}

LATEST USER INPUT: "{user_request}"

CRITICAL WORKFLOW RULES:
1. If fields are missing → ask for them (status: "gathering", next_action: "ask_question")
2. If user provides new info → validate and update fields, then ALWAYS go back to confirmation
3. If all fields collected BUT user_confirmed = False → ask for confirmation (status: "confirming", next_action: "confirm")
4. If user explicitly confirms (says yes/confirm/looks good/etc) → complete (status: "completed", next_action: "complete")
5. If user wants to update after confirmation → help them update, reset confirmation to False, then ASK FOR CONFIRMATION AGAIN

CRITICAL RULE: NEVER auto-complete even if all fields are collected. You MUST wait for explicit user confirmation before setting next_action to "complete". Only complete when the user clearly confirms they are satisfied with ALL the requirements.

Look for confirmation keywords like: "yes", "confirm", "looks good", "that's correct", "proceed", "ok", "fine", etc.

Respond with:
- All current field values (or None if not collected)
- status: "gathering" (collecting fields), "confirming" (asking for final approval), or "completed" (user explicitly confirmed)
- next_action: "ask_question", "confirm", or "complete"
- response_message: Natural, helpful message to user
- missing_fields: list of fields still needed
- all_requirements_collected: true if all fields have values
- never update the USER CONFIRMATION STATUS BY YOUSELF

Be conversational and guide the user through the process naturally."""

        response = self.github_agent.invoke(
            {"messages": [{"role": "user", "content": agent_prompt}]}
        )

        github_analysis = response["structured_response"]

        # Update conversation history
        conversation_history.append(
            {"role": "assistant", "content": github_analysis.response_message}
        )

        # Update github requirements from agent response
        updated_requirements = {}
        if github_analysis.use_case_name:
            updated_requirements["use_case_name"] = github_analysis.use_case_name
        if github_analysis.template:
            updated_requirements["template"] = github_analysis.template
        if github_analysis.internal_team:
            updated_requirements["internal_team"] = github_analysis.internal_team
        if github_analysis.development_team:
            updated_requirements["development_team"] = github_analysis.development_team
        if github_analysis.additional_team:
            updated_requirements["additional_team"] = github_analysis.additional_team

        # Determine if user confirmed based on agent's analysis
        new_user_confirmed = github_analysis.next_action == "complete"

        # Reset confirmation flag if user is updating something
        if (
            github_analysis.status == "gathering"
            and updated_requirements != github_requirements
        ):
            new_user_confirmed = False

        print(f"📋 GitHub Status: {github_analysis.status}")
        print(f"🔄 Next Action: {github_analysis.next_action}")
        print(f"✅ User Confirmed: {new_user_confirmed}")

        return {
            "github_requirements": updated_requirements,
            "structured_response": github_analysis.model_dump(),
            "conversation_history": conversation_history,
            "user_confirmed": new_user_confirmed,
            "current_step": (
                "github_completed"
                if github_analysis.next_action == "complete"
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
            "github_repo_complete",
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
        "structured_response": {},
        "github_requirements": {},
        "user_confirmed": False,  # NEW: Initialize confirmation flag
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
