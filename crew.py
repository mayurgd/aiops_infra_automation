from crewai.flow.flow import Flow, listen, start
from dotenv import load_dotenv
from litellm import completion
import threading
import mlflow

load_dotenv()
mlflow.openai.autolog()
mlflow.litellm.autolog()
mlflow.crewai.autolog()
mlflow.set_experiment("CrewAI")

# Global state management for human-in-the-loop
crew_state = {
    "status": "idle",
    "prompt": None,
    "result": None,
    "user_input": None,
    "input_event": None,
}

LOCAL = True


def get_human_input(prompt: str) -> str:
    """Get input from human user through UI interface."""
    if LOCAL:
        return input(f"ASSISTANT: {prompt}\nUSER:")
    else:
        crew_state["status"] = "waiting_input"
        crew_state["prompt"] = prompt
        crew_state["input_event"] = threading.Event()

        # Wait for user input
        crew_state["input_event"].wait()

        user_response = crew_state["user_input"].strip()
        crew_state["status"] = "running"
        crew_state["prompt"] = None

        return user_response


class SupervisorFlow(Flow):
    model = "gpt-4o-mini"

    @start()
    def greeter(self):
        """Greets user and gathers their request intelligently"""
        print("Starting supervisor flow")
        print(f"Flow State ID: {self.state['id']}")
        prompt = "Hello! I can help you\n1. Create a GitHub repository\n2. Set up a Databricks schema\n3. Create a Databricks compute cluster"
        # Get initial user request
        user_request = get_human_input(prompt)

        cnt = 0
        max_retries = 3
        self.state["intent"] = ""

        while (
            self.state["intent"]
            not in ["github_repo", "databricks_schema", "databricks_compute"]
            and cnt < max_retries
        ):

            if cnt >= 1:
                user_request = get_human_input(
                    f"I didn't understand your request `{user_request}`.\nPlease specify if you want to {prompt.replace('Hello! I can help you','')}"
                )

            # Analyze user intent with improved prompt
            response = completion(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Analyze this user request: "{user_request}"

                                    IMPORTANT: Respond with EXACTLY one of these three options (no quotes, no extra text):
                                    github_repo
                                    databricks_schema
                                    databricks_compute
                                    not_specified

                                    Classification rules:
                                    - If they mention "1", "first", "option 1", GitHub, git, repository, repo, code -> respond with: github_repo
                                    - If they mention "2", "second", "option 2", Databricks schema, database schema, data schema -> respond with: databricks_schema
                                    - If they mention "3", "third", "option 3", Databricks compute, cluster, compute cluster -> respond with: databricks_compute
                                    - else not_specified

                                    Examples:
                                    User: "1" -> github_repo
                                    User: "I want option 1" -> github_repo
                                    User: "I want to create a GitHub repo" -> github_repo
                                    User: "first option" -> github_repo
                                    User: "first" -> github_repo
                                    If you cant choose any one from the above you can return not_specified
                                    """,
                    },
                ],
            )

            raw_intent = response["choices"][0]["message"]["content"]
            intent = raw_intent.strip().lower()

            # Store user request and intent in state
            self.state["user_request"] = user_request
            self.state["intent"] = intent
            cnt += 1

        # Debug output
        print(f"Raw LLM response: '{raw_intent}'")
        print(f"Processed intent: '{intent}'")
        print(f"User request: {user_request}")
        print(f"Final intent: {intent}")
        print(f"Attempt {cnt} of {max_retries}")

        print(f"Final detected intent: {self.state['intent']}")
        return {"user_request": user_request, "intent": self.state["intent"]}


# Example usage:
if __name__ == "__main__":
    flow = SupervisorFlow()
    flow.plot()
    result = flow.kickoff()

    print("Flow completed!")
    print(f"Final state: {flow.state}")
