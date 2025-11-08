"""
Nestle Chat Client - Production Implementation

A custom ChatClient implementation for Nestle's internal OpenAI-compatible API.
This client extends OpenAIBaseChatClient to leverage built-in functionality for
tool calling, streaming, message formatting, etc.

Author: Mayur G D
Date: 2025
"""

import os
import json
import http.client
import logging
import asyncio
from typing import (
    Any,
    Dict,
    List,
    Optional,
    AsyncIterator,
    Union,
)
from dotenv import load_dotenv

from agent_framework import (
    ChatClientProtocol,
    ChatResponse,
    ChatMessage,
    ChatOptions,
    TextContent,
    FunctionCallContent,
    FunctionResultContent,
)
from agent_framework.openai._chat_client import OpenAIBaseChatClient
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types import CompletionUsage
import time
from data_models.custom_llm_models import NestleAPIConfig, NestleAPIError

load_dotenv()
# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NestleAsyncClient:
    """
    Adapter to make Nestle API compatible with OpenAI AsyncOpenAI interface.

    This adapter implements the necessary interface for OpenAIBaseChatClient
    to work with Nestle's internal API.
    """

    def __init__(
        self,
        host: str,
        base_path: str,
        model: str,
        client_id: str,
        client_secret: str,
        api_version: str,
    ):
        """
        Initialize the Nestle async client adapter.

        Args:
            host: API host
            base_path: Base path for API endpoints
            model: Model name/deployment
            client_id: API client ID for authentication
            client_secret: API client secret for authentication
            api_version: API version
        """
        self.host = host
        self.base_path = base_path
        self.model = model
        self.client_id = client_id
        self.client_secret = client_secret
        self.api_version = api_version
        self.base_url = f"https://{host}{base_path}"

        # Create chat namespace to match OpenAI structure
        self.chat = self._ChatNamespace(self)

    class _ChatNamespace:
        """Namespace for chat-related operations."""

        def __init__(self, parent: "NestleAsyncClient"):
            self.parent = parent
            self.completions = self._CompletionsNamespace(parent)

        class _CompletionsNamespace:
            """Namespace for chat completions operations."""

            def __init__(self, parent: "NestleAsyncClient"):
                self.parent = parent

            async def create(
                self,
                *,
                messages: List[Dict[str, Any]],
                model: str,
                stream: bool = False,
                temperature: Optional[float] = None,
                max_tokens: Optional[int] = None,
                stop: Optional[List[str]] = None,
                tools: Optional[List[Dict[str, Any]]] = None,
                stream_options: Optional[Dict[str, Any]] = None,
                **kwargs: Any,
            ):
                """
                Create a chat completion.

                Args:
                    messages: List of message dictionaries
                    model: Model name
                    stream: Whether to stream the response
                    temperature: Sampling temperature
                    max_tokens: Maximum tokens in response
                    stop: Stop sequences
                    tools: Available tools/functions
                    stream_options: Streaming options
                    **kwargs: Additional parameters

                Returns:
                    ChatCompletion or AsyncIterator[ChatCompletionChunk]
                """
                if stream:
                    return self._create_streaming(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stop=stop,
                        tools=tools,
                        **kwargs,
                    )
                else:
                    return await self._create_non_streaming(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stop=stop,
                        tools=tools,
                        **kwargs,
                    )

            async def _create_non_streaming(
                self,
                *,
                messages: List[Dict[str, Any]],
                model: str,
                temperature: Optional[float] = None,
                max_tokens: Optional[int] = None,
                stop: Optional[List[str]] = None,
                tools: Optional[List[Dict[str, Any]]] = None,
                **kwargs: Any,
            ) -> ChatCompletion:
                """Make non-streaming API call to Nestle LLM."""
                # Build payload
                payload_dict = {
                    "messages": messages,
                    "temperature": temperature if temperature is not None else 0.7,
                }

                if max_tokens is not None:
                    payload_dict["max_tokens"] = max_tokens

                if stop is not None:
                    payload_dict["stop"] = stop
                elif not tools:
                    payload_dict["stop"] = ["\nObservation:"]

                if tools:
                    payload_dict["tools"] = tools

                payload = json.dumps(payload_dict)

                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "client_id": self.parent.client_id,
                    "client_secret": self.parent.client_secret,
                }

                api_path = (
                    f"{self.parent.base_path}/{self.parent.model}/chat/completions"
                    f"?api-version={self.parent.api_version}&tes=null"
                )

                try:
                    conn = http.client.HTTPSConnection(self.parent.host, timeout=120)
                    conn.request("POST", api_path, payload, headers)
                    res = conn.getresponse()
                    data = json.loads(res.read().decode("utf-8"))
                    conn.close()

                    if res.status != 200:
                        error_msg = data.get("error", {}).get("message", str(data))
                        raise NestleAPIError(res.status, error_msg)

                    return self._convert_to_openai_format(data)

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse API response: {e}")
                    raise NestleAPIError(500, f"Invalid JSON response: {e}")
                except NestleAPIError:
                    raise
                except Exception as e:
                    logger.error(f"Nestle LLM API error: {str(e)}")
                    raise NestleAPIError(500, str(e))

            async def _create_streaming(
                self,
                *,
                messages: List[Dict[str, Any]],
                model: str,
                temperature: Optional[float] = None,
                max_tokens: Optional[int] = None,
                stop: Optional[List[str]] = None,
                tools: Optional[List[Dict[str, Any]]] = None,
                **kwargs: Any,
            ) -> AsyncIterator[ChatCompletionChunk]:
                """Simulate streaming by getting full response and yielding it word by word."""
                full_response = await self._create_non_streaming(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                    tools=tools,
                    **kwargs,
                )

                content = full_response.choices[0].message.content or ""
                response_id = full_response.id
                created = full_response.created

                words = content.split()

                for i, word in enumerate(words):
                    chunk_text = word + (" " if i < len(words) - 1 else "")

                    chunk = ChatCompletionChunk(
                        id=response_id,
                        choices=[
                            ChunkChoice(
                                delta=ChoiceDelta(
                                    content=chunk_text,
                                    role="assistant" if i == 0 else None,
                                ),
                                finish_reason=None,
                                index=0,
                            )
                        ],
                        created=created,
                        model=model,
                        object="chat.completion.chunk",
                    )

                    yield chunk
                    await asyncio.sleep(0.01)

                final_chunk = ChatCompletionChunk(
                    id=response_id,
                    choices=[
                        ChunkChoice(
                            delta=ChoiceDelta(),
                            finish_reason=full_response.choices[0].finish_reason,
                            index=0,
                        )
                    ],
                    created=created,
                    model=model,
                    object="chat.completion.chunk",
                    usage=full_response.usage,
                )

                yield final_chunk

            def _convert_to_openai_format(
                self, nestle_response: Dict[str, Any]
            ) -> ChatCompletion:
                """Convert Nestle API response to OpenAI ChatCompletion format."""
                choice_data = nestle_response["choices"][0]
                message_data = choice_data["message"]

                message_dict = {
                    "role": message_data.get("role", "assistant"),
                    "content": message_data.get("content"),
                }

                if "tool_calls" in message_data and message_data["tool_calls"]:
                    message_dict["tool_calls"] = message_data["tool_calls"]

                choice = Choice(
                    finish_reason=choice_data.get("finish_reason", "stop"),
                    index=choice_data.get("index", 0),
                    message=ChatCompletionMessage(**message_dict),
                    logprobs=choice_data.get("logprobs"),
                )

                usage_data = nestle_response.get("usage", {})
                usage = CompletionUsage(
                    completion_tokens=usage_data.get("completion_tokens", 0),
                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                    total_tokens=usage_data.get("total_tokens", 0),
                )

                return ChatCompletion(
                    id=nestle_response.get("id", f"nestle-{int(time.time())}"),
                    choices=[choice],
                    created=nestle_response.get("created", int(time.time())),
                    model=nestle_response.get("model", self.parent.model),
                    object="chat.completion",
                    usage=usage,
                    system_fingerprint=nestle_response.get("system_fingerprint"),
                )


class NestleChatClient(OpenAIBaseChatClient):
    """
    Custom ChatClient implementation for Nestle's internal OpenAI API.

    This client extends OpenAIBaseChatClient to leverage all the built-in
    functionality for tool calling, streaming, message formatting, etc.

    NEW FEATURE: Auto Tool Execution
    --------------------------------
    When auto_execute_tools=True (default), the client automatically:
    1. Detects when the LLM requests tool calls
    2. Executes the requested tools
    3. Sends results back to the LLM
    4. Repeats until the LLM provides a final answer

    Example:
        >>> client = NestleChatClient(
        ...     model="gpt-4.1",
        ...     temperature=0.7,
        ...     client_id="your-client-id",
        ...     client_secret="your-client-secret",
        ...     auto_execute_tools=True,  # Enable auto execution
        ... )
        >>>
        >>> # Define a tool
        >>> @ai_function
        >>> async def get_weather(location: str) -> str:
        ...     return f"Weather in {location}: Sunny"
        >>>
        >>> # Just call get_response - tools are executed automatically!
        >>> messages = [ChatMessage(role="user", contents=[TextContent(text="What's the weather in Paris?")])]
        >>> options = ChatOptions(tools=[get_weather])
        >>> response = await client.get_response(messages, chat_options=options)
        >>> # Response will contain the final answer after tool execution
    """

    def __init__(
        self,
        model: str = "gpt-4.1",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        client_id: str = "",
        client_secret: str = "",
        api_version: str = "2024-02-01",
        instruction_role: Optional[str] = None,
        config: Optional[NestleAPIConfig] = None,
        auto_execute_tools: bool = True,  # NEW: Enable auto tool execution
        max_tool_turns: int = 5,  # NEW: Maximum turns for tool execution
        verbose: bool = False,  # NEW: Enable verbose logging for tool execution
    ):
        """
        Initialize the Nestle Chat Client.

        Args:
            model: Model name (default: gpt-4.1)
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens in response
            client_id: API client ID
            client_secret: API client secret
            api_version: API version (default: 2024-02-01)
            instruction_role: The role to use for 'instruction' messages
            config: Optional NestleAPIConfig for advanced configuration
            auto_execute_tools: Enable automatic tool execution (default: True)
            max_tool_turns: Maximum conversation turns for tool execution (default: 5)
            verbose: Enable verbose logging for tool execution (default: False)

        Raises:
            ValueError: If client_id or client_secret is empty
        """
        if not client_id or not client_secret:
            raise ValueError("client_id and client_secret are required")

        self.config = config or NestleAPIConfig()

        self.client_id = client_id
        self.client_secret = client_secret
        self.api_version = api_version
        self.host = self.config.host
        self.base_path = self.config.base_path
        self._temperature = temperature
        self._max_tokens = max_tokens

        # NEW: Tool execution settings
        self._auto_execute_tools = auto_execute_tools
        self._max_tool_turns = max_tool_turns
        self._verbose = verbose

        nestle_client = NestleAsyncClient(
            host=self.host,
            base_path=self.base_path,
            model=model,
            client_id=client_id,
            client_secret=client_secret,
            api_version=api_version,
        )

        super().__init__(
            model_id=model,
            api_key=lambda: "dummy-key",
            client=nestle_client,  # type: ignore
            instruction_role=instruction_role,
        )

        logger.info(
            f"Initialized NestleChatClient with model: {model} "
            f"(auto_execute_tools={auto_execute_tools}, max_tool_turns={max_tool_turns})"
        )

    @property
    def temperature(self) -> float:
        """Get the temperature setting."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Set the temperature setting."""
        if not 0 <= value <= 2:
            logger.warning(f"Temperature {value} is outside typical range [0, 2]")
        self._temperature = value

    @property
    def max_tokens(self) -> Optional[int]:
        """Get the max_tokens setting."""
        return self._max_tokens

    @max_tokens.setter
    def max_tokens(self, value: Optional[int]) -> None:
        """Set the max_tokens setting."""
        if value is not None and value <= 0:
            raise ValueError("max_tokens must be positive")
        self._max_tokens = value

    @property
    def auto_execute_tools(self) -> bool:
        """Get the auto_execute_tools setting."""
        return self._auto_execute_tools

    @auto_execute_tools.setter
    def auto_execute_tools(self, value: bool) -> None:
        """Set the auto_execute_tools setting."""
        self._auto_execute_tools = value
        logger.info(f"Auto tool execution {'enabled' if value else 'disabled'}")

    @property
    def max_tool_turns(self) -> int:
        """Get the max_tool_turns setting."""
        return self._max_tool_turns

    @max_tool_turns.setter
    def max_tool_turns(self, value: int) -> None:
        """Set the max_tool_turns setting."""
        if value <= 0:
            raise ValueError("max_tool_turns must be positive")
        self._max_tool_turns = value

    @property
    def verbose(self) -> bool:
        """Get the verbose setting."""
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        """Set the verbose setting."""
        self._verbose = value

    def service_url(self) -> str:
        """Get the URL of the service."""
        return f"https://{self.host}{self.base_path}"

    # NEW: Override get_response to add auto tool execution
    async def get_response(
        self,
        messages: List[ChatMessage],
        *,  # Force keyword-only arguments
        chat_options: Optional[ChatOptions] = None,
    ) -> ChatResponse:
        """
        Get a response from the LLM with automatic tool execution.

        If auto_execute_tools is enabled and tools are provided in chat_options,
        this method will automatically execute any tool calls requested by the LLM
        and continue the conversation until a final answer is provided.

        Args:
            messages: List of chat messages
            chat_options: Optional chat options including tools

        Returns:
            ChatResponse with the final answer (after tool execution if applicable)

        Example:
            >>> @ai_function
            >>> async def get_weather(location: str) -> str:
            ...     return f"Weather in {location}: Sunny"
            >>>
            >>> messages = [ChatMessage(role="user", contents=[TextContent(text="What's the weather?")])]
            >>> options = ChatOptions(tools=[get_weather])
            >>> response = await client.get_response(messages, chat_options=options)
        """
        # If auto execution is disabled or no tools provided, use default behavior
        if not self._auto_execute_tools or not chat_options or not chat_options.tools:
            return await super().get_response(messages, chat_options=chat_options)

        # Auto execute tools
        return await self._get_response_with_auto_tools(messages, chat_options)

    async def _get_response_with_auto_tools(
        self,
        messages: List[ChatMessage],
        chat_options: ChatOptions,
    ) -> ChatResponse:
        """
        Internal method to handle automatic tool execution.

        This method implements the tool execution loop:
        1. Send messages to LLM
        2. Check if LLM requested tool calls
        3. Execute tools and add results to conversation
        4. Repeat until final answer or max turns reached

        Args:
            messages: Initial messages
            chat_options: Chat options with tools

        Returns:
            Final ChatResponse after tool execution
        """
        conversation = list(messages)

        for turn in range(self._max_tool_turns):
            if self._verbose:
                logger.info(
                    f"[Auto Tool Execution] Turn {turn + 1}/{self._max_tool_turns}"
                )

            # Get response from LLM - use keyword argument
            response = await super().get_response(
                conversation, chat_options=chat_options
            )
            conversation.extend(response.messages)

            # Check if there are function calls to execute
            tool_calls_found = await self._execute_tool_calls(
                response.messages, chat_options.tools, conversation
            )

            # If no tool calls were found, this is the final answer
            if not tool_calls_found:
                if self._verbose:
                    logger.info(
                        "[Auto Tool Execution] No more tool calls - returning final answer"
                    )
                return response

        # Max turns reached
        logger.warning(
            f"[Auto Tool Execution] Reached max turns ({self._max_tool_turns}) "
            "without completion"
        )
        return response

    async def _execute_tool_calls(
        self,
        messages: List[ChatMessage],
        tools: List[Any],
        conversation: List[ChatMessage],
    ) -> bool:
        """
        Execute any tool calls found in the messages.

        Args:
            messages: Messages to check for tool calls
            tools: Available tools
            conversation: Conversation history to append results to

        Returns:
            True if any tool calls were found and executed, False otherwise
        """
        has_tool_calls = False

        for msg in messages:
            for content in msg.contents:
                if isinstance(content, FunctionCallContent):
                    has_tool_calls = True
                    await self._execute_single_tool(content, tools, conversation)

        return has_tool_calls

    async def _execute_single_tool(
        self,
        function_call: FunctionCallContent,
        tools: List[Any],
        conversation: List[ChatMessage],
    ) -> None:
        """
        Execute a single tool call and add the result to the conversation.

        Args:
            function_call: The function call to execute
            tools: Available tools
            conversation: Conversation history to append result to
        """
        if self._verbose:
            logger.info(f"[Auto Tool Execution] Executing tool: {function_call.name}")
            logger.info(f"[Auto Tool Execution] Arguments: {function_call.arguments}")

        # Find the tool
        tool = next((t for t in tools if t.name == function_call.name), None)

        if not tool:
            error_msg = f"Tool '{function_call.name}' not found in available tools"
            logger.error(f"[Auto Tool Execution] {error_msg}")
            self._add_tool_error_to_conversation(conversation, function_call, error_msg)
            return

        try:
            # Parse arguments
            if isinstance(function_call.arguments, str):
                args = json.loads(function_call.arguments)
            else:
                args = function_call.arguments

            # Execute the tool
            result = await tool.invoke(arguments=tool.input_model(**args))

            if self._verbose:
                logger.info(f"[Auto Tool Execution] Tool result: {result}")

            # Add result to conversation
            conversation.append(
                ChatMessage(
                    role="tool",
                    contents=[
                        FunctionResultContent(
                            call_id=function_call.call_id,
                            name=function_call.name,
                            result=str(result[0].text),
                        )
                    ],
                )
            )

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse arguments: {e}"
            logger.error(f"[Auto Tool Execution] {error_msg}")
            self._add_tool_error_to_conversation(conversation, function_call, error_msg)

        except Exception as e:
            error_msg = f"Error executing tool: {str(e)}"
            logger.error(f"[Auto Tool Execution] {error_msg}", exc_info=True)
            self._add_tool_error_to_conversation(conversation, function_call, error_msg)

    def _add_tool_error_to_conversation(
        self,
        conversation: List[ChatMessage],
        function_call: FunctionCallContent,
        error_message: str,
    ) -> None:
        """
        Add a tool execution error to the conversation.

        Args:
            conversation: Conversation history
            function_call: The function call that failed
            error_message: Error message to add
        """
        conversation.append(
            ChatMessage(
                role="tool",
                contents=[
                    FunctionResultContent(
                        call_id=function_call.call_id,
                        name=function_call.name,
                        result=f"Error: {error_message}",
                    )
                ],
            )
        )

    # NEW: Override get_streaming_response to add auto tool execution
    async def get_streaming_response(
        self,
        messages: List[ChatMessage],
        *,  # Force keyword-only arguments
        chat_options: Optional[ChatOptions] = None,
    ) -> AsyncIterator[Union[ChatMessage, None]]:
        """
        Get a streaming response from the LLM with automatic tool execution.

        Note: When auto_execute_tools is enabled and tools are provided,
        streaming behavior is modified:
        - Tool execution happens non-streaming (tools are executed in full)
        - Only the final response after all tool calls is streamed

        Args:
            messages: List of chat messages
            chat_options: Optional chat options including tools

        Yields:
            ChatMessage updates for streaming response

        Example:
            >>> async for update in client.get_streaming_response(messages, chat_options=options):
            ...     if update and update.contents:
            ...         print(update.contents[0].text, end="", flush=True)
        """
        # If auto execution is disabled or no tools provided, use default streaming
        if not self._auto_execute_tools or not chat_options or not chat_options.tools:
            async for update in super().get_streaming_response(
                messages, chat_options=chat_options
            ):
                yield update
            return

        # With auto tool execution, we need to handle tools first
        # then stream the final response
        conversation = list(messages)

        for turn in range(self._max_tool_turns):
            if self._verbose:
                logger.info(
                    f"[Auto Tool Execution - Streaming] Turn {turn + 1}/{self._max_tool_turns}"
                )

            # Get non-streaming response to check for tool calls
            response = await super().get_response(
                conversation, chat_options=chat_options
            )
            conversation.extend(response.messages)

            # Check if there are function calls to execute
            tool_calls_found = await self._execute_tool_calls(
                response.messages, chat_options.tools, conversation
            )

            # If no tool calls, stream the final response
            if not tool_calls_found:
                if self._verbose:
                    logger.info(
                        "[Auto Tool Execution - Streaming] No more tool calls - "
                        "streaming final answer"
                    )

                # Stream the final response
                async for update in super().get_streaming_response(
                    conversation, chat_options=chat_options
                ):
                    yield update
                return

        # Max turns reached - stream what we have
        logger.warning(
            f"[Auto Tool Execution - Streaming] Reached max turns ({self._max_tool_turns})"
        )
        async for update in super().get_streaming_response(
            conversation, chat_options=chat_options
        ):
            yield update


# Utility Functions


async def get_response_with_auto_tools(
    client: ChatClientProtocol,
    messages: List[ChatMessage],
    options: ChatOptions,
    max_turns: int = 5,
    verbose: bool = False,
) -> ChatResponse:
    """
    Helper that automatically executes tools in a loop.

    NOTE: If you're using NestleChatClient with auto_execute_tools=True,
    you don't need this function - just call client.get_response() directly!

    This function is provided for backwards compatibility and for use with
    other chat clients that don't have built-in auto tool execution.

    Args:
        client: Chat client to use
        messages: Initial messages
        options: Chat options including tools
        max_turns: Maximum conversation turns (default: 5)
        verbose: Whether to print debug information

    Returns:
        Final ChatResponse with the answer

    Example:
        >>> # With NestleChatClient (auto execution built-in):
        >>> response = await client.get_response(messages, options)
        >>>
        >>> # With other clients (need helper function):
        >>> response = await get_response_with_auto_tools(client, messages, options)
    """
    conversation = list(messages)

    for turn in range(max_turns):
        if verbose:
            logger.info(f"Turn {turn + 1}/{max_turns}")

        response = await client.get_response(conversation, chat_options=options)
        conversation.extend(response.messages)

        # Check if there are function calls to execute
        has_function_calls = False
        for msg in response.messages:
            for content in msg.contents:
                if isinstance(content, FunctionCallContent):
                    has_function_calls = True

                    if verbose:
                        logger.info(f"Executing tool: {content.name}")

                    # Find and execute the tool
                    tool = next(
                        (t for t in options.tools if t.name == content.name), None
                    )
                    if tool:
                        try:
                            # Parse arguments
                            args = (
                                json.loads(content.arguments)
                                if isinstance(content.arguments, str)
                                else content.arguments
                            )

                            # Execute the tool
                            result = await tool.invoke(
                                arguments=tool.input_model(**args)
                            )

                            if verbose:
                                logger.info(f"Tool result: {result}")

                            # Add result to conversation
                            conversation.append(
                                ChatMessage(
                                    role="tool",
                                    contents=[
                                        FunctionResultContent(
                                            call_id=content.call_id,
                                            name=content.name,
                                            result=str(result),
                                        )
                                    ],
                                )
                            )
                        except Exception as e:
                            logger.error(f"Error executing tool {content.name}: {e}")
                            # Add error result to conversation
                            conversation.append(
                                ChatMessage(
                                    role="tool",
                                    contents=[
                                        FunctionResultContent(
                                            call_id=content.call_id,
                                            name=content.name,
                                            result=f"Error: {str(e)}",
                                        )
                                    ],
                                )
                            )
                    else:
                        logger.warning(
                            f"Tool {content.name} not found in options.tools"
                        )

        # If no more function calls, return the response
        if not has_function_calls:
            if verbose:
                logger.info("Conversation completed")
            return response

    logger.warning(f"Reached max turns ({max_turns}) without completion")
    return response


def create_client_from_env(
    auto_execute_tools: bool = True,
    max_tool_turns: int = 5,
    verbose: bool = False,
) -> NestleChatClient:
    """
    Create a NestleChatClient from environment variables.

    Expected environment variables:
        - NESTLE_CLIENT_ID: API client ID
        - NESTLE_CLIENT_SECRET: API client secret
        - NESTLE_MODEL: Model name (optional, default: gpt-4.1)
        - NESTLE_TEMPERATURE: Temperature (optional, default: 0.7)
        - NESTLE_MAX_TOKENS: Max tokens (optional)

    Args:
        auto_execute_tools: Enable automatic tool execution (default: True)
        max_tool_turns: Maximum turns for tool execution (default: 5)
        verbose: Enable verbose logging (default: False)

    Returns:
        Configured NestleChatClient

    Raises:
        ValueError: If required environment variables are missing

    Example:
        >>> from dotenv import load_dotenv
        >>> load_dotenv()
        >>> client = create_client_from_env(auto_execute_tools=True, verbose=True)
    """
    import os

    client_id = os.getenv("NESTLE_CLIENT_ID")
    client_secret = os.getenv("NESTLE_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError(
            "NESTLE_CLIENT_ID and NESTLE_CLIENT_SECRET environment variables are required"
        )

    model = os.getenv("NESTLE_MODEL", "gpt-4.1")
    temperature = float(os.getenv("NESTLE_TEMPERATURE", "0.7"))
    max_tokens_str = os.getenv("NESTLE_MAX_TOKENS")
    max_tokens = int(max_tokens_str) if max_tokens_str else None

    return NestleChatClient(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        client_id=client_id,
        client_secret=client_secret,
        auto_execute_tools=auto_execute_tools,
        max_tool_turns=max_tool_turns,
        verbose=verbose,
    )
