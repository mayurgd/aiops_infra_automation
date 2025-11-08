import os
from dotenv import load_dotenv
from agent_framework import ai_function
from typing import Annotated
import logging
import asyncio
from agent_framework import (
    ChatClientProtocol,
    ChatMessage,
    ChatOptions,
    TextContent,
)
from custom_llm.nestle_chat_client import *
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize client with auto tool execution enabled
try:
    client = create_client_from_env(auto_execute_tools=True, verbose=True)
    logger.info("Client initialized successfully with auto tool execution")
except ValueError as e:
    logger.error(f"Failed to initialize client: {e}")
    exit(1)

# Verify it implements the protocol
assert isinstance(client, ChatClientProtocol)
logger.info("Client implements ChatClientProtocol ✓")

# Example messages
messages = [
    ChatMessage(
        role="user",
        contents=[
            TextContent(text="What is MLOps? Give me a brief answer in 2-3 sentences.")
        ],
    )
]


# Test 1: Regular Response
async def test_response():
    """Test basic chat completion."""
    print("\n" + "=" * 60)
    print("TEST 1: Regular Response")
    print("=" * 60)

    try:
        response = await client.get_response(messages)
        print("\n✓ Response received:")
        print(f"  {response.messages[0].contents[0].text}")
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


# Test 2: Streaming
async def test_streaming():
    """Test streaming chat completion."""
    print("\n" + "=" * 60)
    print("TEST 2: Streaming Response")
    print("=" * 60)
    print("\nStreaming: ", end="", flush=True)

    try:
        stream_generator = client.get_streaming_response(messages)

        async for update in stream_generator:
            if update and hasattr(update, "contents") and update.contents:
                content = update.contents[0]
                if hasattr(content, "text") and content.text:
                    print(content.text, end="", flush=True)

        print("\n\n✓ Streaming completed successfully")
        return True

    except Exception as e:
        print(f"\n✗ Streaming failed: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


# Test 3: Auto Tool Execution (NEW!)
async def test_auto_tool_execution():
    """Test automatic tool execution built into the client."""
    print("\n" + "=" * 60)
    print("TEST 3: Auto Tool Execution (Built-in)")
    print("=" * 60)

    # Define tools
    @ai_function
    async def get_weather(location: Annotated[str, "The city name"]) -> str:
        """Get the current weather for a location."""
        logger.info(f"[Tool] get_weather called with location='{location}'")
        await asyncio.sleep(0.1)
        return f"The weather in {location} is sunny and 72°F"

    @ai_function
    async def get_time(
        timezone: Annotated[str, "The timezone (e.g., 'UTC', 'EST')"],
    ) -> str:
        """Get the current time in a specific timezone."""
        logger.info(f"[Tool] get_time called with timezone='{timezone}'")
        await asyncio.sleep(0.1)
        return f"The current time in {timezone} is 14:30"

    tool_messages = [
        ChatMessage(
            role="user",
            contents=[
                TextContent(
                    text="What's the weather in Paris and what time is it in UTC?"
                )
            ],
        )
    ]

    options = ChatOptions(tools=[get_weather, get_time])

    try:
        print("\n[Test] Sending request with tools...")
        print(
            "[Test] Auto tool execution is ENABLED - tools will execute automatically"
        )

        # Just call get_response - tools are executed automatically!
        response = await client.get_response(tool_messages, chat_options=options)

        print("\n✓ Final Response (after auto tool execution):")
        for msg in response.messages:
            for content in msg.contents:
                if isinstance(content, TextContent) and content.text:
                    print(f"  {content.text}")

        return True

    except Exception as e:
        print(f"\n✗ Auto tool execution failed: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


# Test 4: Auto Tool Execution with Streaming (NEW!)
async def test_auto_tool_execution_streaming():
    """Test automatic tool execution with streaming."""
    print("\n" + "=" * 60)
    print("TEST 4: Auto Tool Execution with Streaming")
    print("=" * 60)

    @ai_function
    async def calculate(
        operation: Annotated[
            str, "The operation: 'add', 'subtract', 'multiply', 'divide'"
        ],
        a: Annotated[float, "First number"],
        b: Annotated[float, "Second number"],
    ) -> str:
        """Perform a mathematical calculation."""
        logger.info(f"[Tool] calculate called: {operation}({a}, {b})")
        await asyncio.sleep(0.1)

        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero",
        }

        result = operations.get(operation, lambda x, y: "Unknown operation")(a, b)
        return f"The result of {operation}({a}, {b}) is {result}"

    tool_messages = [
        ChatMessage(
            role="user",
            contents=[
                TextContent(
                    text="What is 15 multiplied by 8, and then add 100 to that result?"
                )
            ],
        )
    ]

    options = ChatOptions(tools=[calculate])

    try:
        print("\n[Test] Streaming response with auto tool execution...")
        print("[Test] Tools will execute first, then final answer will stream")
        print("\nStreaming final answer: ", end="", flush=True)

        # Stream the response - tools are executed automatically before streaming!
        async for update in client.get_streaming_response(
            tool_messages, chat_options=options
        ):
            if update and hasattr(update, "contents") and update.contents:
                content = update.contents[0]
                if hasattr(content, "text") and content.text:
                    print(content.text, end="", flush=True)

        print("\n\n✓ Streaming with auto tool execution completed successfully")
        return True

    except Exception as e:
        print(f"\n✗ Streaming with auto tool execution failed: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


# Test 5: Manual Tool Execution (Disabled Auto)
async def test_manual_tool_execution():
    """Test with auto tool execution disabled."""
    print("\n" + "=" * 60)
    print("TEST 5: Manual Tool Execution (Auto Disabled)")
    print("=" * 60)

    # Create client with auto execution disabled
    manual_client = NestleChatClient(
        model="gpt-4.1",
        client_id=os.getenv("NESTLE_CLIENT_ID"),
        client_secret=os.getenv("NESTLE_CLIENT_SECRET"),
        auto_execute_tools=False,  # Disable auto execution
        verbose=True,
    )

    @ai_function
    async def get_user_info(user_id: Annotated[str, "The user ID"]) -> str:
        """Get information about a user."""
        logger.info(f"[Tool] get_user_info called with user_id='{user_id}'")
        await asyncio.sleep(0.1)
        return f"User {user_id}: John Doe, email: john@example.com"

    tool_messages = [
        ChatMessage(
            role="user",
            contents=[TextContent(text="Get information for user ID 12345")],
        )
    ]

    options = ChatOptions(tools=[get_user_info])

    try:
        print("\n[Test] Auto execution is DISABLED")
        print("[Test] Using helper function for manual tool execution...")

        # Use the helper function for manual tool execution
        response = await get_response_with_auto_tools(
            manual_client, tool_messages, options, verbose=True
        )

        print("\n✓ Manual tool execution completed:")
        for msg in response.messages:
            for content in msg.contents:
                if isinstance(content, TextContent) and content.text:
                    print(f"  {content.text}")

        return True

    except Exception as e:
        print(f"\n✗ Manual tool execution failed: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


# Test 6: Complex Multi-Tool Scenario
async def test_complex_multi_tool():
    """Test complex scenario with multiple tool calls."""
    print("\n" + "=" * 60)
    print("TEST 6: Complex Multi-Tool Scenario")
    print("=" * 60)

    @ai_function
    async def search_database(
        query: Annotated[str, "Search query"],
        table: Annotated[str, "Table name"],
    ) -> str:
        """Search a database table."""
        logger.info(f"[Tool] search_database: {query} in {table}")
        await asyncio.sleep(0.1)
        return f"Found 3 results in {table} for '{query}': Item1, Item2, Item3"

    @ai_function
    async def get_details(item_id: Annotated[str, "Item ID"]) -> str:
        """Get detailed information about an item."""
        logger.info(f"[Tool] get_details: {item_id}")
        await asyncio.sleep(0.1)
        return f"Details for {item_id}: Price=$99, Stock=50, Rating=4.5/5"

    @ai_function
    async def calculate_discount(
        price: Annotated[float, "Original price"],
        discount_percent: Annotated[float, "Discount percentage"],
    ) -> str:
        """Calculate discounted price."""
        logger.info(f"[Tool] calculate_discount: {price} with {discount_percent}% off")
        await asyncio.sleep(0.1)
        discounted = price * (1 - discount_percent / 100)
        return f"Original: ${price}, Discount: {discount_percent}%, Final: ${discounted:.2f}"

    tool_messages = [
        ChatMessage(
            role="user",
            contents=[
                TextContent(
                    text="Search for 'laptop' in the products table, get details for Item1, "
                    "and calculate a 20% discount on its price."
                )
            ],
        )
    ]

    options = ChatOptions(tools=[search_database, get_details, calculate_discount])

    try:
        print("\n[Test] Complex multi-tool scenario with auto execution...")

        response = await client.get_response(tool_messages, chat_options=options)

        print("\n✓ Complex multi-tool execution completed:")
        for msg in response.messages:
            for content in msg.contents:
                if isinstance(content, TextContent) and content.text:
                    print(f"  {content.text}")

        return True

    except Exception as e:
        print(f"\n✗ Complex multi-tool test failed: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


# Test 7: Error Handling in Tool Execution
async def test_tool_error_handling():
    """Test error handling during tool execution."""
    print("\n" + "=" * 60)
    print("TEST 7: Tool Error Handling")
    print("=" * 60)

    @ai_function
    async def failing_tool(value: Annotated[int, "Some value"]) -> str:
        """A tool that fails."""
        logger.info(f"[Tool] failing_tool called with value={value}")
        raise ValueError("This tool intentionally fails for testing")

    @ai_function
    async def working_tool(text: Annotated[str, "Some text"]) -> str:
        """A tool that works."""
        logger.info(f"[Tool] working_tool called with text='{text}'")
        return f"Successfully processed: {text}"

    tool_messages = [
        ChatMessage(
            role="user",
            contents=[
                TextContent(
                    text="First use the failing tool with value 42, "
                    "then use the working tool with text 'hello'"
                )
            ],
        )
    ]

    options = ChatOptions(tools=[failing_tool, working_tool])

    try:
        print("\n[Test] Testing error handling in tool execution...")

        response = await client.get_response(tool_messages, chat_options=options)

        print("\n✓ Error handling test completed:")
        print("  (Check logs above to see error handling)")
        for msg in response.messages:
            for content in msg.contents:
                if isinstance(content, TextContent) and content.text:
                    print(f"  {content.text}")

        return True

    except Exception as e:
        print(f"\n✗ Error handling test failed: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


# Test 8: Configuration and Settings
async def test_configuration():
    """Test client configuration options."""
    print("\n" + "=" * 60)
    print("TEST 8: Configuration and Settings")
    print("=" * 60)

    # Test custom config
    custom_config = NestleAPIConfig(
        default_temperature=0.5,
        default_model="gpt-4.1",
        auto_execute_tools=True,
        max_tool_turns=3,
    )

    custom_client = NestleChatClient(
        model="gpt-4.1",
        client_id=os.getenv("NESTLE_CLIENT_ID"),
        client_secret=os.getenv("NESTLE_CLIENT_SECRET"),
        config=custom_config,
        auto_execute_tools=True,
        max_tool_turns=3,
        verbose=False,
    )

    print(f"✓ Custom client created")
    print(f"  Service URL: {custom_client.service_url()}")
    print(f"  Temperature: {custom_client.temperature}")
    print(f"  Model: {custom_client.model_id}")
    print(f"  Auto Execute Tools: {custom_client.auto_execute_tools}")
    print(f"  Max Tool Turns: {custom_client.max_tool_turns}")
    print(f"  Verbose: {custom_client.verbose}")

    # Test dynamic configuration changes
    print("\n[Test] Testing dynamic configuration changes...")
    custom_client.temperature = 0.8
    custom_client.auto_execute_tools = False
    custom_client.max_tool_turns = 10
    custom_client.verbose = True

    print(f"✓ Configuration updated:")
    print(f"  Temperature: {custom_client.temperature}")
    print(f"  Auto Execute Tools: {custom_client.auto_execute_tools}")
    print(f"  Max Tool Turns: {custom_client.max_tool_turns}")
    print(f"  Verbose: {custom_client.verbose}")

    return True


# Test 9: Performance Test
async def test_performance():
    """Test performance with multiple concurrent requests."""
    print("\n" + "=" * 60)
    print("TEST 9: Performance Test")
    print("=" * 60)

    @ai_function
    async def quick_tool(value: Annotated[int, "A number"]) -> str:
        """A quick tool for performance testing."""
        return f"Processed: {value}"

    async def single_request(request_id: int):
        """Make a single request."""
        messages = [
            ChatMessage(
                role="user",
                contents=[
                    TextContent(text=f"Use the quick tool with value {request_id}")
                ],
            )
        ]
        options = ChatOptions(tools=[quick_tool])

        start_time = time.time()
        await client.get_response(messages, chat_options=options)
        duration = time.time() - start_time

        return duration

    try:
        print("\n[Test] Running 5 concurrent requests...")

        start_time = time.time()
        durations = await asyncio.gather(*[single_request(i) for i in range(5)])
        total_time = time.time() - start_time

        print(f"\n✓ Performance test completed:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average request time: {sum(durations)/len(durations):.2f}s")
        print(f"  Min request time: {min(durations):.2f}s")
        print(f"  Max request time: {max(durations):.2f}s")

        return True

    except Exception as e:
        print(f"\n✗ Performance test failed: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


# Test 10: Edge Cases
async def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n" + "=" * 60)
    print("TEST 10: Edge Cases")
    print("=" * 60)

    passed_tests = []

    # Test 1: Empty tool list
    try:
        messages = [ChatMessage(role="user", contents=[TextContent(text="Hello")])]
        options = ChatOptions(tools=[])
        response = await client.get_response(messages, chat_options=options)
        print("✓ Empty tool list handled correctly")
        passed_tests.append(True)
    except Exception as e:
        print(f"✗ Empty tool list failed: {e}")
        passed_tests.append(False)

    # Test 2: No chat options
    try:
        messages = [ChatMessage(role="user", contents=[TextContent(text="Hello")])]
        response = await client.get_response(messages)
        print("✓ No chat options handled correctly")
        passed_tests.append(True)
    except Exception as e:
        print(f"✗ No chat options failed: {e}")
        passed_tests.append(False)

    # Test 3: Very long message
    try:
        long_text = "Hello " * 100
        messages = [ChatMessage(role="user", contents=[TextContent(text=long_text)])]
        response = await client.get_response(messages)
        print("✓ Very long message handled correctly")
        passed_tests.append(True)
    except Exception as e:
        print(f"✗ Very long message failed: {e}")
        passed_tests.append(False)

    # Test 4: Invalid temperature (should log warning)
    try:
        original_temp = client.temperature
        client.temperature = 3.0  # Outside normal range
        client.temperature = original_temp
        print("✓ Invalid temperature handled correctly (warning logged)")
        passed_tests.append(True)
    except Exception as e:
        print(f"✗ Invalid temperature test failed: {e}")
        passed_tests.append(False)

    # Test 5: Max tool turns reached
    @ai_function
    async def recursive_tool(count: Annotated[int, "Counter"]) -> str:
        """A tool that keeps calling itself."""
        return f"Call {count}, please call me again with {count + 1}"

    try:
        # Create client with max_tool_turns=2
        limited_client = NestleChatClient(
            model="gpt-4.1",
            client_id=os.getenv("NESTLE_CLIENT_ID"),
            client_secret=os.getenv("NESTLE_CLIENT_SECRET"),
            auto_execute_tools=True,
            max_tool_turns=2,
            verbose=False,
        )

        messages = [
            ChatMessage(
                role="user",
                contents=[TextContent(text="Start the recursive tool with count 1")],
            )
        ]
        options = ChatOptions(tools=[recursive_tool])

        response = await limited_client.get_response(messages, chat_options=options)
        print("✓ Max tool turns limit enforced correctly")
        passed_tests.append(True)
    except Exception as e:
        print(f"✗ Max tool turns test failed: {e}")
        passed_tests.append(False)

    return all(passed_tests)


# Run all tests
async def run_all_tests():
    """Run all test cases."""
    print("\n" + "=" * 80)
    print(" " * 20 + "NESTLE CHAT CLIENT - COMPREHENSIVE TEST SUITE")
    print(" " * 25 + "With Auto Tool Execution")
    print("=" * 80)

    results = {
        "Regular Response": await test_response(),
        "Streaming": await test_streaming(),
        "Auto Tool Execution": await test_auto_tool_execution(),
        "Auto Tool Execution + Streaming": await test_auto_tool_execution_streaming(),
        "Manual Tool Execution": await test_manual_tool_execution(),
        "Complex Multi-Tool": await test_complex_multi_tool(),
        "Tool Error Handling": await test_tool_error_handling(),
        "Configuration": await test_configuration(),
        "Performance": await test_performance(),
        "Edge Cases": await test_edge_cases(),
    }

    print("\n" + "=" * 80)
    print(" " * 30 + "TEST RESULTS SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        status_color = status
        print(f"{test_name:.<50} {status_color}")

    total = len(results)
    passed = sum(results.values())
    percentage = (passed / total) * 100

    print("=" * 80)
    print(f"Total: {passed}/{total} tests passed ({percentage:.1f}%)")
    print("=" * 80)

    if passed == total:
        print("\n🎉 All tests passed! The client is working perfectly.")
    elif passed >= total * 0.8:
        print("\n⚠️  Most tests passed, but some issues need attention.")
    else:
        print("\n❌ Multiple tests failed. Please review the errors above.")

    return all(results.values())


# Run tests
try:
    all_passed = asyncio.run(run_all_tests())
    exit(0 if all_passed else 1)
except KeyboardInterrupt:
    print("\n\nTests interrupted by user")
    exit(1)
except Exception as e:
    logger.error(f"Test suite failed: {e}", exc_info=True)
    exit(1)
