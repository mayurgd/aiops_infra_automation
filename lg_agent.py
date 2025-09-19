import asyncio
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.constants import START
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv
import uuid
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.types import interrupt, Command
from pydantic import BaseModel


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
    confidence: str
    next_action: str
    response_message: str


# State structure
class SupervisorState(TypedDict):
    user_request: str
    intent: str
    messages: Annotated[list, add_messages]
    conversation_history: list
    current_step: str
    structured_response: dict
    github_requirements: dict
    user_confirmed: bool  # NEW: Track if user explicitly confirmed


class ModernizedSupervisorGraph:
    def __init__(self):
        self.valid_intents = ["github_repo", "databricks_schema", "databricks_compute"]

        # Intent capture agent
        self.intent_agent = create_react_agent(
            model="gpt-4o-mini",
            tools=[],
            response_format=IntentResponse,
        )

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
            prompt = """Hello! I'm your AI Operations assistant. I can help you with:

1. Create a GitHub repository
2. Set up a Databricks schema  
3. Create a Databricks compute cluster

What would you like to do today?"""
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

        conversation_history.append({"role": "user", "content": user_input})

        return {
            "user_request": user_input,
            "conversation_history": conversation_history,
        }

    def intent_capture_agent_node(self, state: SupervisorState) -> SupervisorState:
        """Intent capture using react agent"""
        user_request = state["user_request"]
        conversation_history = state.get("conversation_history", [])

        conversation_context = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in conversation_history[-5:]]
        )

        agent_prompt = f"""You are an AI Operations assistant. Analyze the user's intent for these services:

1. github_repo - Creating GitHub repositories
2. databricks_schema - Setting up Databricks schemas  
3. databricks_compute - Creating Databricks compute clusters

Conversation so far: {conversation_context}
Current request: "{user_request}"

Determine:
- intent: one of the three options above, or "unclear"
- confidence: "high", "medium", or "low"
- next_action: "proceed" if clear, "clarify" if unclear
- response_message: Natural, helpful response"""

        response = self.intent_agent.invoke(
            {"messages": [{"role": "user", "content": agent_prompt}]}
        )

        intent_analysis = response["structured_response"]

        conversation_history.append(
            {"role": "assistant", "content": intent_analysis.response_message}
        )

        return {
            "intent": intent_analysis.intent,
            "structured_response": intent_analysis.model_dump(),
            "conversation_history": conversation_history,
            "current_step": (
                "intent_captured"
                if intent_analysis.next_action == "proceed"
                else "intent_clarification"
            ),
        }

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

    def route_from_intent(self, state: SupervisorState) -> str:
        """Route after intent is captured"""
        structured_response = state.get("structured_response", {})
        next_action = structured_response.get("next_action", "clarify")
        intent = state.get("intent", "unclear")

        if next_action == "proceed" and intent == "github_repo":
            return "github_flow"
        elif next_action == "proceed" and intent in self.valid_intents:
            return "complete"
        elif next_action == "proceed":
            return "complete"
        else:
            return "continue"

    def route_github_flow(self, state: SupervisorState) -> str:
        """Route within GitHub requirements flow"""
        current_step = state.get("current_step", "")

        if current_step == "github_completed":
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
        workflow.add_node("github_agent", self.github_requirements_agent_node)
        workflow.add_node("completion", self.completion_node)

        # Build the flow
        workflow.add_edge(START, "human")
        workflow.add_edge("human", "intent_agent")

        # Route after intent capture
        workflow.add_conditional_edges(
            "intent_agent",
            self.route_from_intent,
            {
                "continue": "human",  # Keep clarifying intent
                "github_flow": "github_agent",  # Start GitHub requirements
                "complete": "completion",  # Other intents or done
            },
        )

        # GitHub flow routing
        workflow.add_conditional_edges(
            "github_agent",
            self.route_github_flow,
            {
                "continue": "human",  # Keep gathering requirements
                "complete": "completion",  # Requirements completed
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
        "messages": [],
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
