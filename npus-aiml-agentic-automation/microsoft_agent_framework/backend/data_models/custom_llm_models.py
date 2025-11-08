from dataclasses import dataclass
from typing import List


@dataclass
class NestleAPIConfig:
    """Configuration for Nestle API."""

    host: str = "int-eur-sdr-int-pub.nestle.com"
    base_path: str = "/api/dv-exp-accelerator-openai-api/1/openai/deployments"
    api_version: str = "2024-02-01"
    default_model: str = "gpt-4.1"
    default_temperature: float = 0.7
    default_stop_sequences: List[str] = None
    auto_execute_tools: bool = True  # NEW: Enable auto tool execution by default
    max_tool_turns: int = 5  # NEW: Maximum turns for tool execution loop

    def __post_init__(self):
        if self.default_stop_sequences is None:
            self.default_stop_sequences = ["\nObservation:"]


class NestleAPIError(Exception):
    """Custom exception for Nestle API errors."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API error {status_code}: {message}")
