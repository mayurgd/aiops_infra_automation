import requests
import time
import json
from typing import Dict, Any


class GitHubRepoAutomation:
    def __init__(self, github_token: str, org: str, automation_repo: str):
        self.github_token = github_token
        self.org = org
        self.automation_repo = automation_repo
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def validate_repository_inputs(
        self,
        use_case_name: str,
        template: str,
        internal_team: str,
        development_team: str,
        additional_team: str,
    ) -> Dict[str, Any]:
        """
        Validate all inputs for repository creation.

        Returns:
            Dict with validation results
        """
        errors = []

        # Validate template
        valid_templates = [
            "npus-aiml-mlops-stacks-template",
            "npus-aiml-skinny-dab-template",
        ]
        if template not in valid_templates:
            errors.append(f"Invalid template. Must be one of: {valid_templates}")

        # Validate teams
        valid_teams = [
            "eai",
            "deloitte",
            "sig",
            "tredence",
            "bora",
            "genpact",
            "tiger",
            "srm",
            "kroger",
        ]

        if internal_team not in valid_teams:
            errors.append(f"Invalid internal team. Must be one of: {valid_teams}")

        if development_team not in valid_teams + ["none"]:
            errors.append(
                f"Invalid development team. Must be one of: {valid_teams + ['none']}"
            )

        if additional_team not in valid_teams + ["none"]:
            errors.append(
                f"Invalid additional team. Must be one of: {valid_teams + ['none']}"
            )

        # Validate use case name
        if not use_case_name or not use_case_name.strip():
            errors.append("Use case name cannot be empty")
        elif not use_case_name.replace("-", "").replace("_", "").isalnum():
            errors.append(
                "Use case name should only contain alphanumeric character and hyphens. eg: example-repo"
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "expected_repo_name": (
                f"npus-aiml-{internal_team}-{use_case_name}"
                if len(errors) == 0
                else None
            ),
        }

    def create_repository(
        self,
        use_case_name: str,
        template: str,
        internal_team: str,
        development_team: str,
        additional_team: str = "none",
    ) -> Dict[str, Any]:
        """
        Create a repository using workflow_dispatch
        """

        # Use the centralized validation function
        validation_result = self.validate_repository_inputs(
            use_case_name, template, internal_team, development_team, additional_team
        )

        if not validation_result["valid"]:
            return {"success": False, "errors": validation_result["errors"]}

        # Trigger workflow_dispatch
        url = f"{self.base_url}/repos/{self.org}/{self.automation_repo}/actions/workflows/repo-utility.yml/dispatches"

        data = {
            "ref": "main",
            "inputs": {
                "use_case_name": use_case_name,
                "template": template,
                "internal_team": internal_team,
                "development_team": development_team,
                "additional_team": additional_team,
            },
        }

        response = requests.post(url, json=data, headers=self.headers, verify=False)

        if response.status_code == 204:
            return {
                "success": True,
                "message": f"Repository creation triggered for {use_case_name}",
                "expected_repo_name": f"npus-aiml-{internal_team}-{use_case_name}",
            }
        else:
            return {
                "success": False,
                "error": f"Failed to trigger repository creation: {response.status_code}",
            }

    def get_workflow_runs(
        self, workflow_file: str = "repo-utility.yml", per_page: int = 10
    ) -> Dict[str, Any]:
        """
        Get the status of recent workflow runs
        """
        url = f"{self.base_url}/repos/{self.org}/{self.automation_repo}/actions/workflows/{workflow_file}/runs"
        params = {"per_page": per_page}

        response = requests.get(url, headers=self.headers, params=params, verify=False)

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to get workflow runs: {response.status_code}"}

    def wait_for_completion(
        self, max_wait_minutes: int = 10, run_id: str = None
    ) -> Dict[str, Any]:
        """
        Wait for the most recent workflow run to complete
        """
        max_wait_seconds = max_wait_minutes * 60
        start_time = time.time()

        # Store initial run ID to track the correct workflow
        target_run_id = run_id

        while time.time() - start_time < max_wait_seconds:
            if target_run_id:
                # Get specific run by ID
                url = f"{self.base_url}/repos/{self.org}/{self.automation_repo}/actions/runs/{target_run_id}"
                response = requests.get(url, headers=self.headers, verify=False)

                if response.status_code == 200:
                    latest_run = response.json()
                else:
                    return {
                        "completed": False,
                        "error": f"Failed to get workflow run {target_run_id}: {response.status_code}",
                    }
            else:
                # Get latest run
                runs = self.get_workflow_runs(per_page=1)
                if "workflow_runs" not in runs or not runs["workflow_runs"]:
                    time.sleep(30)
                    continue

                latest_run = runs["workflow_runs"][0]
                # Store the run ID for subsequent checks
                if not target_run_id:
                    target_run_id = latest_run["id"]

            status = latest_run["status"]
            conclusion = latest_run.get("conclusion")

            print(
                f"Workflow run {latest_run['id']}: status={status}, conclusion={conclusion}"
            )

            # Check if workflow is truly completed
            if status == "completed" and conclusion is not None:
                return {
                    "completed": True,
                    "success": conclusion == "success",
                    "conclusion": conclusion,
                    "run_id": latest_run["id"],
                    "html_url": latest_run["html_url"],
                    "status": status,
                }
            elif status in ["queued", "in_progress"]:
                # Workflow is still running
                print(f"Workflow still running... Status: {status}")
                time.sleep(30)
                continue
            elif status == "completed" and conclusion is None:
                # This shouldn't happen, but handle it just in case
                print("Workflow marked as completed but conclusion is None, waiting...")
                time.sleep(30)
                continue
            else:
                # Handle other statuses
                print(f"Unexpected workflow status: {status}")
                time.sleep(30)
                continue

        return {
            "completed": False,
            "timeout": True,
            "message": f"Workflow did not complete within {max_wait_minutes} minutes",
            "last_status": (
                latest_run.get("status") if "latest_run" in locals() else "unknown"
            ),
        }

    def get_latest_workflow_run_id(self) -> str:
        """
        Get the ID of the most recent workflow run
        """
        runs = self.get_workflow_runs(per_page=1)
        if "workflow_runs" in runs and runs["workflow_runs"]:
            return runs["workflow_runs"][0]["id"]
        return None


# Simplified interface for agentic AI
def create_repo_for_agent(
    github_token: str,
    org: str,
    automation_repo: str,
    use_case_name: str,
    template: str = "npus-aiml-mlops-stacks-template",
    internal_team: str = "eai",
    development_team: str = "none",
    additional_team: str = "none",
    wait_for_completion: bool = True,
) -> Dict[str, Any]:
    """
    Simplified interface for agentic AI to create repositories
    """

    automation = GitHubRepoAutomation(github_token, org, automation_repo)

    # Get current latest run ID before triggering new one
    pre_trigger_run_id = automation.get_latest_workflow_run_id()

    # Trigger the creation
    result = automation.create_repository(
        use_case_name=use_case_name,
        template=template,
        internal_team=internal_team,
        development_team=development_team,
        additional_team=additional_team,
    )

    if not result["success"]:
        return result

    if wait_for_completion:
        # Wait a bit for the new workflow to appear
        time.sleep(10)

        # Get the new workflow run ID
        new_run_id = automation.get_latest_workflow_run_id()

        # If we have a new run ID that's different from before, track that specific run
        if new_run_id and new_run_id != pre_trigger_run_id:
            completion_result = automation.wait_for_completion(run_id=new_run_id)
        else:
            completion_result = automation.wait_for_completion()

        result.update(completion_result)

    return result


# Standalone validation function for external use
def validate_repository_inputs(
    use_case_name: str,
    template: str,
    internal_team: str,
    development_team: str,
    additional_team: str,
) -> Dict[str, Any]:
    """
    Standalone validation function that can be used without instantiating the class.

    Args:
        use_case_name: Name of the use case for the repository
        template: Template to use
        internal_team: Internal team
        development_team: Development team
        additional_team: Additional team

    Returns:
        Dict with validation results
    """
    # Create a temporary instance just for validation (no API calls needed)
    temp_automation = GitHubRepoAutomation("dummy_token", "dummy_org", "dummy_repo")
    return temp_automation.validate_repository_inputs(
        use_case_name, template, internal_team, development_team, additional_team
    )


# Example usage
if __name__ == "__main__":
    # Configure these variables
    GITHUB_TOKEN = "TOKEN"
    ORG = "nestle-it"
    AUTOMATION_REPO = "npus-aiml-utilities-create-repo"

    # Create a repository
    result = create_repo_for_agent(
        github_token=GITHUB_TOKEN,
        org=ORG,
        automation_repo=AUTOMATION_REPO,
        use_case_name="test-agent-run",
        template="npus-aiml-mlops-stacks-template",
        internal_team="eai",
        development_team="deloitte",
        additional_team="none",
    )

    print(json.dumps(result, indent=2))
