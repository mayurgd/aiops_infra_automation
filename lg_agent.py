import asyncio
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.constants import START
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import uuid
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.types import interrupt, Command
from pydantic import BaseModel


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


# Define structured response for intent analysis
class IntentResponse(BaseModel):
    intent: str
    confidence: str
    next_action: str
    response_message: str


# Define the state structure - simplified for STEP 1
class SupervisorState(TypedDict):
    user_request: str
    intent: str
    messages: Annotated[list, add_messages]
    conversation_history: list
    current_step: str
    structured_response: dict


class ModernizedSupervisorGraph:
    def __init__(self):
        self.valid_intents = ["github_repo", "databricks_schema", "databricks_compute"]

        # Create the modern react agent with structured response
        self.intent_agent = create_react_agent(
            model="gpt-4o-mini",
            tools=[],  # No external tools needed for intent analysis
            response_format=IntentResponse,
        )

    def human_node(self, state: SupervisorState) -> SupervisorState:
        """Handle human input using LangGraph's interrupt mechanism"""
        # Get conversation context for better prompting
        conversation_history = state.get("conversation_history", [])

        if not conversation_history:
            # First interaction - greeting
            prompt = """Hello! I'm your AI Operations assistant. I can help you with:

1. Create a GitHub repository
2. Set up a Databricks schema  
3. Create a Databricks compute cluster

What would you like to do today?"""
        else:
            # Show the agent's last response to continue the conversation
            last_agent_message = None
            for msg in reversed(conversation_history):
                if msg["role"] == "assistant":
                    last_agent_message = msg["content"]
                    break

            if last_agent_message:
                prompt = f"🤖 Assistant: {last_agent_message}\n\nYour response:"
            else:
                prompt = "Please tell me what you'd like to do:"

        user_input = interrupt(
            {"prompt": prompt, "step": state.get("current_step", "intent_capture")}
        )

        # Update conversation history
        conversation_history.append({"role": "user", "content": user_input})

        return {
            "user_request": user_input,
            "conversation_history": conversation_history,
        }

    def intent_capture_agent_node(self, state: SupervisorState) -> SupervisorState:
        """Modern react agent that handles greeting, intent analysis, and retry logic"""
        user_request = state["user_request"]
        conversation_history = state.get("conversation_history", [])

        # Build conversation context
        conversation_context = "\n".join(
            [
                f"{msg['role']}: {msg['content']}"
                for msg in conversation_history[-5:]  # Last 5 messages for context
            ]
        )

        # Create the message for the react agent
        agent_prompt = f"""You are an AI Operations assistant. Your job is to understand what the user wants to do and guide them to one of three specific services:

                        1. github_repo - Creating GitHub repositories
                        2. databricks_schema - Setting up Databricks schemas  
                        3. databricks_compute - Creating Databricks compute clusters

                        Analyze the user's request and determine:
                        - intent: one of the three options above, or "unclear" if you can't determine
                        - next_action: "proceed" if intent is clear, "clarify" if you need more information, or "redirect" if they want something else
                        - response_message: A natural, conversational response to the user

                        Conversation so far:
                        {conversation_context}

                        Current user request: "{user_request}"

                        Be conversational and helpful. If the intent is unclear, ask clarifying questions naturally."""

        # Invoke the react agent
        response = self.intent_agent.invoke(
            {"messages": [{"role": "user", "content": agent_prompt}]}
        )

        # Extract structured response
        intent_analysis = response["structured_response"]

        # Update conversation history with assistant response
        conversation_history.append(
            {"role": "assistant", "content": intent_analysis.response_message}
        )

        print(f"Intent Analysis: {intent_analysis.intent}")

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

    def route_conversation(self, state: SupervisorState) -> str:
        """Route based on whether intent is clear or needs more clarification"""
        structured_response = state.get("structured_response", {})
        next_action = structured_response.get("next_action", "clarify")
        intent = state.get("intent", "unclear")

        print(f"🔄 Routing decision: next_action={next_action}, intent={intent}")

        # If intent is clear and in our valid intents, proceed to completion
        if next_action == "proceed" and intent in self.valid_intents:
            return "complete"
        # If user wants something we don't support, redirect them
        elif next_action == "redirect":
            return "complete"  # For now, complete with explanation
        # Otherwise, continue the conversation - this is key for interactive chat
        else:
            return "continue"

    def completion_node(self, state: SupervisorState) -> SupervisorState:
        """Final completion node with structured response"""
        structured_response = state.get("structured_response", {})
        intent = state.get("intent", "unclear")

        # Show the final agent message to the user
        if intent in self.valid_intents:
            final_message = f"✅ Perfect! I understand you want to work with {intent.replace('_', ' ')}. Ready to proceed!"
        else:
            final_message = structured_response.get(
                "response_message", "Thank you for using the AI Operations assistant!"
            )

        # Display the final interaction to user
        print(f"\n🤖 Assistant: {final_message}")
        print(f"🎯 Final Intent: {intent}")

        return {"current_step": "completed", "intent": intent}

    def build_graph(self) -> StateGraph:
        """Build the modernized graph with recursive conversation flow"""
        workflow = StateGraph(SupervisorState)

        # Add nodes
        workflow.add_node("human", self.human_node)
        workflow.add_node("intent_agent", self.intent_capture_agent_node)
        workflow.add_node("completion", self.completion_node)

        # Flow: Start -> Human -> Intent Agent -> Route (Continue or Complete)
        workflow.add_edge(START, "human")
        workflow.add_edge("human", "intent_agent")

        # Conditional routing - either continue conversation or complete
        workflow.add_conditional_edges(
            "intent_agent",
            self.route_conversation,
            {
                "continue": "human",  # Keep the conversation going
                "complete": "completion",  # Intent captured, move to completion
            },
        )

        workflow.add_edge("completion", END)

        checkpointer = InMemorySaver()
        return workflow.compile(checkpointer=checkpointer)


# Example usage:
if __name__ == "__main__":
    supervisor = ModernizedSupervisorGraph()
    graph = supervisor.build_graph()

    # Simplified initial state
    initial_state = {
        "user_request": "",
        "intent": "",
        "messages": [],
        "conversation_history": [],
        "current_step": "start",
        "structured_response": {},
    }

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("🚀 Starting modernized intent capture agent...")
    result = graph.invoke(initial_state, config=config)

    # Handle interrupts with recursive conversation
    while "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0]
        prompt_text = interrupt_data.value.get("prompt", "Please provide your input:")
        user_response = input(f"\n{prompt_text}\n> ")

        result = graph.invoke(Command(resume=user_response), config=config)

    print("\n✅ Intent capture completed!")
    print(f"🎯 Final captured intent: {result.get('intent', 'Not captured')}")

    # Display the simplified graph structure
    try:
        print("\n📊 Graph structure:")
        print(graph.get_graph().draw_ascii())
    except Exception as e:
        print(f"Could not display graph structure: {e}")
