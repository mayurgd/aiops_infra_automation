"""Nestle LLM API client module."""

import json
import os
from http.client import HTTPSConnection
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class NestleLLM:
    """Client for interacting with the Nestle LLM API.

    This class provides a simple interface to call Nestle's LLM service
    with configurable parameters.

    Attributes:
        client_id: API client identifier.
        client_secret: API client secret for authentication.
        model: The model to use for completions (default: gpt-4.1).
    """

    API_HOST = "int-eur-sdr-int-pub.nestle.com"
    API_BASE_PATH = "/api/dv-exp-accelerator-openai-api/1/openai/deployments"
    API_VERSION = "2024-02-01"
    DEFAULT_MODEL = "gpt-4.1"

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """Initialize the NestleLLM client.

        Args:
            client_id: API client ID. If None, reads from NESTLE_CLIENT_ID env var.
            client_secret: API secret. If None, reads from NESTLE_CLIENT_SECRET env var.
            model: Model name. If None, reads from NESTLE_MODEL env var or uses default.

        Raises:
            ValueError: If client_id or client_secret are not provided.
        """
        self.client_id = client_id or os.getenv("NESTLE_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("NESTLE_CLIENT_SECRET")
        self.model = model or os.getenv("NESTLE_MODEL", self.DEFAULT_MODEL)

        if not self.client_id or not self.client_secret:
            raise ValueError("Missing NESTLE_CLIENT_ID or NESTLE_CLIENT_SECRET")

    def call(
        self,
        messages: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Call the Nestle LLM API with the given messages.

        Args:
            messages: The user message to send to the API.
            temperature: Sampling temperature (0.0 to 1.0). Default is 0.7.
            max_tokens: Maximum tokens in the response. Default is 1000.

        Returns:
            The content of the API response.

        Raises:
            Exception: If the API request fails with a non-200 status code.
        """
        conn = HTTPSConnection(self.API_HOST)

        payload = json.dumps(
            {
                "messages": [{"role": "user", "content": messages}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        endpoint = (
            f"{self.API_BASE_PATH}/{self.model}/chat/completions"
            f"?api-version={self.API_VERSION}&tes=null"
        )

        try:
            conn.request("POST", endpoint, payload, headers)
            res = conn.getresponse()
            data = res.read()

            if res.status != 200:
                raise Exception(
                    f"API failed with status {res.status}: {data.decode('utf-8')}"
                )

            response_json = json.loads(data.decode("utf-8"))
            return response_json["choices"][0]["message"]["content"]
        finally:
            conn.close()
