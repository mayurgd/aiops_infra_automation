import json
import http.client
import logging
from typing import Any, Dict, List, Optional, Union

from crewai.llm import BaseLLM

logger = logging.getLogger(__name__)


class NestleLLM(BaseLLM):
    """Custom LLM implementation for Nestle's internal OpenAI API."""

    def __init__(
        self,
        model: str = "gpt-4.1",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        client_id: str = "",
        client_secret: str = "",
        api_version: str = "2024-02-01",
    ):
        """
        Initialize the Nestle LLM.

        Args:
            model: Model name (default: gpt-4.1)
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens in response
            client_id: API client ID
            client_secret: API client secret
            api_version: API version (default: 2024-02-01)
        """
        super().__init__(model=model, temperature=temperature)
        self.max_tokens = max_tokens
        self.client_id = client_id
        self.client_secret = client_secret
        self.api_version = api_version
        self.host = "int-eur-sdr-int-pub.nestle.com"
        self.base_path = "/api/dv-exp-accelerator-openai-api/1/openai/deployments"

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Make a call to the Nestle LLM API.

        Args:
            messages: String or list of message dicts with 'role' and 'content'
            stop: Stop sequences
            **kwargs: Additional parameters (temperature, max_tokens override)

        Returns:
            String response from the LLM
        """
        # Convert string to message format
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Use provided values or defaults
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        # Add default stop sequence
        if stop is None:
            stop = ["\nObservation:"]
        elif "\nObservation:" not in stop:
            stop.append("\nObservation:")

        payload = json.dumps(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop": stop,
            }
        )

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        api_path = (
            f"{self.base_path}/{self.model}/chat/completions"
            f"?api-version={self.api_version}&tes=null"
        )

        try:
            conn = http.client.HTTPSConnection(self.host)
            conn.request("POST", api_path, payload, headers)
            res = conn.getresponse()
            data = json.loads(res.read().decode("utf-8"))
            conn.close()

            if res.status != 200:
                error_msg = data.get("error", {}).get("message", "Unknown error")
                raise Exception(f"API error {res.status}: {error_msg}")

            return data["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"Nestle LLM error: {str(e)}")
            raise

    def supports_function_calling(self) -> bool:
        """Function calling not supported for this LLM."""
        return False
