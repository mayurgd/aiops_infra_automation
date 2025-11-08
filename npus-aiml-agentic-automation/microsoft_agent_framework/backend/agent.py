"""
MLOps Onboarding Assistant using Microsoft Agent Framework
Provides continuous support for onboarding and platform queries
"""

import os
import asyncio
from dotenv import load_dotenv
from custom_llm.nestle_chat_client import NestleChatClient
from prompts import aiops_agent_prompt
from agent_framework import (
    ChatAgent,
    MCPStreamableHTTPTool,
    ChatMessage,
    TextContent,
    Role,
)

load_dotenv()


class MLOpsOnboardingAgent:
    """
    MLOps Onboarding Assistant that operates in continuous loop mode.
    Handles use case onboarding and platform questions until explicit exit.
    """

    def __init__(self):
        """Initialize the MLOps agent with Nestle LLM and MCP tools"""
        self.client_id = os.getenv("NESTLE_CLIENT_ID")
        self.client_secret = os.getenv("NESTLE_CLIENT_SECRET")
        self.model = os.getenv("NESTLE_MODEL", "gpt-4.1")
        self.mcp_server_url = "http://127.0.0.1:8060/mcp"

        # Define system instructions
        self.system_instructions = aiops_agent_prompt

        # Validate environment
        if not all([self.client_id, self.client_secret]):
            raise ValueError(
                "Missing required environment variables: NESTLE_CLIENT_ID, NESTLE_CLIENT_SECRET"
            )

        # Initialize the chat client
        self.chat_client = NestleChatClient(
            model=self.model,
            temperature=0.7,
            client_id=self.client_id,
            client_secret=self.client_secret,
        )

        # Create the agent with MCP tools
        self.custom_agent = ChatAgent(
            chat_client=self.chat_client,
            instructions=self.system_instructions,
            name="MLOps Onboarding Assistant",
        )

    def _create_message_with_system_instructions(
        self, user_input: str, is_first_message: bool = False
    ) -> list[ChatMessage]:
        """
        Create a list of ChatMessage objects including system instructions.

        Args:
            user_input: The user's input text
            is_first_message: Whether this is the first message in the conversation

        Returns:
            List of ChatMessage objects with system and user messages
        """
        messages = []

        # Always include system message at the start
        if is_first_message:
            system_message = ChatMessage(
                role=Role.SYSTEM, contents=[TextContent(text=self.system_instructions)]
            )
            messages.append(system_message)

        # Add user message
        user_message = ChatMessage(
            role=Role.USER, contents=[TextContent(text=user_input)]
        )
        messages.append(user_message)

        return messages

    async def start_conversation(self):
        """Start the continuous conversation loop"""
        # Create a persistent thread
        thread = self.custom_agent.get_new_thread()

        # Track first message to include system instructions
        is_first_message = True

        async with (
            MCPStreamableHTTPTool(
                name="AIOps servers",
                url="http://127.0.0.1:8060/mcp",
            ) as mcp_server,
        ):
            # Continuous conversation loop
            while True:
                try:
                    # Get user input
                    user_input = input("YOU: ").strip()

                    # Check for explicit exit commands
                    if user_input.lower() in [
                        "quit",
                        "exit",
                        "done",
                        "goodbye",
                        "bye",
                        "stop",
                        "that's all",
                        "no more help needed",
                    ]:
                        # Let agent provide closing message
                        closing_messages = self._create_message_with_system_instructions(
                            f"User said: '{user_input}'. Provide a warm, brief closing message.",
                            is_first_message=False,
                        )
                        response = await self.custom_agent.run(
                            closing_messages,
                            thread=thread,
                        )
                        print(f"ASSISTANT: {response.text}\n")
                        break

                    # Skip empty inputs
                    if not user_input:
                        continue

                    # Create messages with system instructions
                    messages = self._create_message_with_system_instructions(
                        user_input, is_first_message=is_first_message
                    )

                    # Mark that we've sent the first message
                    if is_first_message:
                        is_first_message = False

                    # Send messages to agent
                    response = await self.custom_agent.run(
                        messages, thread=thread, tools=mcp_server
                    )
                    print(f"\nASSISTANT: {response.text}\n")

                except KeyboardInterrupt:
                    print("\n\nInterrupted by user. Goodbye!")
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    print("Please try again or type 'quit' to exit.\n")
                    import traceback

                    traceback.print_exc()


async def main():
    """Main entry point for the MLOps Onboarding Assistant"""
    try:
        # Initialize the agent
        agent = MLOpsOnboardingAgent()

        # Debug: Print agent configuration
        print("\n" + "=" * 60)
        print("Agent Configuration:")
        print("=" * 60)
        agent_dict = agent.custom_agent.to_dict()
        print(f"Name: {agent_dict.get('name')}")
        print(f"Instructions Present: {'instructions' in agent_dict}")
        if "instructions" in agent_dict:
            instructions = agent_dict["instructions"]
            preview = (
                instructions[:100] + "..." if len(instructions) > 100 else instructions
            )
            print(f"Instructions Preview: {preview}")
        print("=" * 60 + "\n")

        # Start the continuous conversation
        await agent.start_conversation()

    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("Please ensure NESTLE_CLIENT_ID and NESTLE_CLIENT_SECRET are set.")
    except Exception as e:
        print(f"Unexpected Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
