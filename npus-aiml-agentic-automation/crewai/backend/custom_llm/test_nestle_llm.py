"""
Test script to verify Nestle LLM integration works correctly
Run this before using with CrewAI to ensure the connection is working
"""

import os
import json
from dotenv import load_dotenv
from nestle_llm import NestleLLM
from crewai import Agent, Task, Crew

load_dotenv()


def test_basic_completion():
    """Test basic completion with the Nestle LLM"""
    print("🔍 Testing Nestle LLM Integration...\n")

    # Get credentials from environment
    client_id = os.getenv("NESTLE_CLIENT_ID")
    client_secret = os.getenv("NESTLE_CLIENT_SECRET")
    model = os.getenv("NESTLE_MODEL", "gpt-4.1")

    if not client_id or not client_secret:
        print(
            "❌ Error: NESTLE_CLIENT_ID and NESTLE_CLIENT_SECRET must be set in .env file"
        )
        return False

    print(f"✓ Credentials loaded")
    print(f"✓ Using model: {model}\n")

    try:
        # Initialize the LLM
        llm = NestleLLM(
            model=model,
            client_id=client_id,
            client_secret=client_secret,
        )
        print("✓ LLM initialized\n")

        # Test with a simple question
        print("📤 Sending test request: 'What is MLOps?'\n")

        messages = [
            {
                "role": "user",
                "content": "What is MLOps? Give me a brief answer in 2-3 sentences.",
            }
        ]

        response = llm.call(messages=messages)

        print("✓ Response received!\n")
        print("=" * 60)
        print("RESPONSE:")
        print("=" * 60)
        print(response)
        print("=" * 60)
        print("\n✅ Test completed successfully!")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {str(e)}")
        import traceback

        print("\nFull traceback:")
        print(traceback.format_exc())
        return False


def test_multi_turn_conversation():
    """Test multi-turn conversation"""
    print("\n" + "=" * 60)
    print("🔍 Testing Multi-Turn Conversation...\n")

    client_id = os.getenv("NESTLE_CLIENT_ID")
    client_secret = os.getenv("NESTLE_CLIENT_SECRET")
    model = os.getenv("NESTLE_MODEL", "gpt-4.1")

    try:
        llm = NestleLLM(
            model=model,
            client_id=client_id,
            client_secret=client_secret,
        )

        # Multi-turn conversation
        messages = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "What is its population?"},
        ]

        print("📤 Sending multi-turn conversation...\n")

        response = llm.call(messages=messages, max_tokens=100)

        print("✓ Response received!\n")
        print("=" * 60)
        print("RESPONSE:")
        print("=" * 60)
        print(response)
        print("=" * 60)
        print("\n✅ Multi-turn test completed successfully!")

        return True

    except Exception as e:
        print(f"\n❌ Multi-turn test failed with error:")
        print(f"   {str(e)}")
        return False


def test_with_crewai():
    """Test integration with CrewAI Agent"""
    print("\n" + "=" * 60)
    print("🔍 Testing CrewAI Integration...\n")

    try:
        client_id = os.getenv("NESTLE_CLIENT_ID")
        client_secret = os.getenv("NESTLE_CLIENT_SECRET")
        model = os.getenv("NESTLE_MODEL", "gpt-4.1")

        # Create LLM for CrewAI
        llm = NestleLLM(
            model=model,
            client_id=client_id,
            client_secret=client_secret,
        )

        print("✓ Nestle LLM created\n")

        # Create a simple agent
        test_agent = Agent(
            role="Test Assistant",
            goal="Answer questions briefly and clearly",
            backstory="You are a helpful assistant that provides concise answers.",
            llm=llm,
            verbose=False,
        )

        print("✓ Test agent created\n")

        # Create a simple task
        test_task = Task(
            description="Explain what DevOps is in one sentence.",
            expected_output="A single sentence explaining DevOps",
            agent=test_agent,
        )

        print("✓ Test task created\n")
        print("📤 Running CrewAI task...\n")

        # Create and run crew
        crew = Crew(agents=[test_agent], tasks=[test_task], verbose=False)

        result = crew.kickoff()

        print("=" * 60)
        print("CREWAI RESULT:")
        print("=" * 60)
        print(result)
        print("=" * 60)
        print("\n✅ CrewAI integration test completed successfully!")

        return True

    except Exception as e:
        print(f"\n❌ CrewAI integration test failed with error:")
        print(f"   {str(e)}")
        import traceback

        print("\nFull traceback:")
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    test_basic_completion()
    test_multi_turn_conversation()
    test_with_crewai()
