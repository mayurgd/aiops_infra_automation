from crew import AiopsAgenticAutomation
from dotenv import load_dotenv

load_dotenv()


def run_supervision_example():
    """
    Run the crew with ambiguous inputs to demonstrate supervising functionality.
    """
    inputs = {
        "user_query": "I need to set up infrastructure for my new project",
    }

    print("=== Running Supervising Example (Ambiguous Inputs) ===")
    try:
        result = AiopsAgenticAutomation().crew().kickoff(inputs=inputs)
        print("Supervising completed successfully!")
        return result
    except Exception as e:
        raise Exception(f"An error occurred during supervision: {e}")


if __name__ == "__main__":
    run_supervision_example()
