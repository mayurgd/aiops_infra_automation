#!/usr/bin/env python
import sys
from crew import AiopsAgenticAutomation

from dotenv import load_dotenv

load_dotenv()


def run():
    """
    Run the crew.
    """
    inputs = {"topic": "Hi"}
    AiopsAgenticAutomation().crew().kickoff(inputs=inputs)


if __name__ == "__main__":
    run()
