import time
import json
import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class WorkflowResult:
    """Result of workflow execution"""

    success: bool
    message: Optional[str] = None
    error: Optional[str] = None
    completed: Optional[bool] = None
    conclusion: Optional[str] = None
    run_id: Optional[str] = None
    html_url: Optional[str] = None
    status: Optional[str] = None
    timeout: Optional[bool] = None
    last_status: Optional[str] = None
    repository_exists: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}


class GitHubRepoAutomation:
    """GitHub repository automation via workflow dispatch"""

    def __init__(self, github_token: str, org: str, automation_repo: str):
        """
        Initialize GitHub automation client

        Args:
            github_token: GitHub personal access token
            org: GitHub organization name
            automation_repo: Repository containing automation workflows
        """
        self.github_token = github_token
        self.org = org
        self.automation_repo = automation_repo
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def check_repository_exists(self, repo_name: str) -> tuple[bool, Optional[str]]:
        """
        Check if a repository already exists in the organization

        Args:
            repo_name: Repository name to check

        Returns:
            Tuple of (exists: bool, error_message: Optional[str])
            - (True, None) if repository exists
            - (False, None) if repository doesn't exist
            - (False, error_message) if there was an error checking
        """
        url = f"{self.base_url}/repos/{self.org}/{repo_name}"

        try:
            response = requests.get(url, headers=self.headers, verify=False)

            if response.status_code == 200:
                # Repository exists
                return True, None
            elif response.status_code == 404:
                # Repository doesn't exist
                return False, None
            else:
                # Unexpected status code
                error_message = f"Unexpected status code {response.status_code} while checking repository existence"
                try:
                    error_data = response.json()
                    error_message += (
                        f": {error_data.get('message', 'No message provided')}"
                    )
                except Exception:
                    pass
                return False, error_message

        except requests.exceptions.RequestException as e:
            return False, f"Network error while checking repository existence: {str(e)}"

    def trigger_repository_creation(
        self,
        use_case_name: str,
        internal_team: str,
        external_team: str,
        additional_team: str,
        template: str,
        skip_existence_check: bool = False,
    ) -> WorkflowResult:
        """
        Trigger repository creation workflow

        Args:
            use_case_name: Name of the use case (already validated)
            internal_team: Internal AI/ML team (already normalized)
            external_team: External AI/ML team (already normalized)
            additional_team: Additional internal team (already normalized)
            template: Template to use (already validated)
            skip_existence_check: Skip the repository existence check (default: False)

        Returns:
            WorkflowResult with trigger status
        """
        # Construct the full repository name
        repo_name = f"npus-aiml-{use_case_name}"

        # Check if repository already exists (unless skipped)
        if not skip_existence_check:
            print(f"Checking if repository '{repo_name}' already exists...")
            exists, error = self.check_repository_exists(repo_name)

            if error:
                return WorkflowResult(
                    success=False,
                    error=f"Failed to check repository existence: {error}",
                )

            if exists:
                return WorkflowResult(
                    success=False,
                    repository_exists=True,
                    error=f"Repository '{repo_name}' already exists in organization '{self.org}'. "
                    f"Please use a different use case name or delete the existing repository first.",
                )

            print(
                f"Repository '{repo_name}' does not exist. Proceeding with creation..."
            )

        url = f"{self.base_url}/repos/{self.org}/{self.automation_repo}/actions/workflows/repo-utility.yml/dispatches"

        data = {
            "ref": "main",
            "inputs": {
                "use_case_name": use_case_name,
                "template": template,
                "internal_team": internal_team,
                "development_team": external_team,
                "additional_team": additional_team,
            },
        }

        try:
            response = requests.post(url, json=data, headers=self.headers, verify=False)

            if response.status_code == 204:
                return WorkflowResult(
                    success=True,
                    repository_exists=False,
                    message=f"Repository creation workflow triggered for '{repo_name}'",
                )
            else:
                error_message = "Unknown error"
                try:
                    error_data = response.json()
                    error_message = error_data.get("message", "No message provided")
                except Exception:
                    error_message = response.text or "No error details"

                error = (
                    f"Failed to trigger repository creation workflow.\n"
                    f"Status Code: {response.status_code}\n"
                    f"Error: {error_message}"
                )
                return WorkflowResult(success=False, error=error)

        except requests.exceptions.RequestException as e:
            return WorkflowResult(
                success=False,
                error=f"Network error while triggering workflow: {str(e)}",
            )

    def get_latest_workflow_run_id(
        self, workflow_file: str = "repo-utility.yml"
    ) -> Optional[str]:
        """
        Get the ID of the most recent workflow run

        Args:
            workflow_file: Workflow filename

        Returns:
            Run ID or None if not found
        """
        url = f"{self.base_url}/repos/{self.org}/{self.automation_repo}/actions/workflows/{workflow_file}/runs"
        params = {"per_page": 1}

        try:
            response = requests.get(
                url, headers=self.headers, params=params, verify=False
            )

            if response.status_code == 200:
                data = response.json()
                if "workflow_runs" in data and data["workflow_runs"]:
                    return data["workflow_runs"][0]["id"]
            return None

        except Exception:
            return None

    def get_workflow_run_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific workflow run

        Args:
            run_id: Workflow run ID

        Returns:
            Run details or None if error
        """
        url = f"{self.base_url}/repos/{self.org}/{self.automation_repo}/actions/runs/{run_id}"

        try:
            response = requests.get(url, headers=self.headers, verify=False)

            if response.status_code == 200:
                return response.json()
            return None

        except Exception:
            return None

    def wait_for_completion(
        self,
        run_id: Optional[str] = None,
        max_wait_minutes: int = 10,
        poll_interval_seconds: int = 30,
    ) -> WorkflowResult:
        """
        Wait for a workflow run to complete

        Args:
            run_id: Specific run ID to track (optional)
            max_wait_minutes: Maximum time to wait
            poll_interval_seconds: Time between status checks

        Returns:
            WorkflowResult with completion status
        """
        max_wait_seconds = max_wait_minutes * 60
        start_time = time.time()
        target_run_id = run_id

        # If no run_id provided, get the latest
        if not target_run_id:
            target_run_id = self.get_latest_workflow_run_id()
            if not target_run_id:
                return WorkflowResult(
                    success=False,
                    completed=False,
                    error="Could not find workflow run to track",
                )

        print(f"Tracking workflow run: {target_run_id}")

        while time.time() - start_time < max_wait_seconds:
            run_data = self.get_workflow_run_status(target_run_id)

            if not run_data:
                return WorkflowResult(
                    success=False,
                    completed=False,
                    error=f"Failed to get status for workflow run {target_run_id}",
                )

            status = run_data["status"]
            conclusion = run_data.get("conclusion")

            print(
                f"Workflow run {target_run_id}: status={status}, conclusion={conclusion}"
            )

            # Check if workflow completed
            if status == "completed" and conclusion is not None:
                return WorkflowResult(
                    success=conclusion == "success",
                    completed=True,
                    conclusion=conclusion,
                    run_id=target_run_id,
                    html_url=run_data["html_url"],
                    status=status,
                    message=(
                        f"Workflow completed with conclusion: {conclusion}"
                        if conclusion == "success"
                        else f"Workflow failed with conclusion: {conclusion}"
                    ),
                )

            # Workflow still in progress
            if status in ["queued", "in_progress"]:
                print(f"Workflow still running... Status: {status}")
                time.sleep(poll_interval_seconds)
                continue

            # Unexpected status
            print(f"Unexpected workflow status: {status}, waiting...")
            time.sleep(poll_interval_seconds)

        # Timeout
        return WorkflowResult(
            success=False,
            completed=False,
            timeout=True,
            message=f"Workflow did not complete within {max_wait_minutes} minutes",
            last_status=run_data.get("status") if run_data else "unknown",
            run_id=target_run_id,
        )


def create_repo_for_agent(
    github_token: str,
    org: str,
    automation_repo: str,
    use_case_name: str,
    internal_team: str,
    external_team: str = "none",
    additional_team: str = "none",
    template: str = "npus-aiml-skinny-dab-template",
    wait_for_completion: bool = True,
    max_wait_minutes: int = 10,
    skip_existence_check: bool = False,
) -> Dict[str, Any]:
    """
    Create GitHub repository via workflow dispatch

    This function is called by the server after validation is complete.
    All inputs are assumed to be already validated and normalized.

    Args:
        github_token: GitHub personal access token
        org: GitHub organization name
        automation_repo: Repository containing automation workflows
        use_case_name: Name of the use case (validated, normalized)
        internal_team: Internal AI/ML team (validated, normalized)
        external_team: External AI/ML team (validated, normalized)
        additional_team: Additional internal team (validated, normalized)
        template: Template to use (validated)
        wait_for_completion: Whether to wait for workflow completion
        max_wait_minutes: Maximum time to wait for completion
        skip_existence_check: Skip the repository existence check (default: False)

    Returns:
        Dictionary with operation results
    """
    automation = GitHubRepoAutomation(github_token, org, automation_repo)

    # Get current latest run ID before triggering (for tracking)
    pre_trigger_run_id = automation.get_latest_workflow_run_id()

    # Trigger the repository creation workflow (includes existence check)
    result = automation.trigger_repository_creation(
        use_case_name=use_case_name,
        internal_team=internal_team,
        external_team=external_team,
        additional_team=additional_team,
        template=template,
        skip_existence_check=skip_existence_check,
    )

    if not result.success:
        return result.to_dict()

    # Wait for completion if requested
    if wait_for_completion:
        print("Waiting for workflow to start...")
        time.sleep(10)  # Give workflow time to appear

        # Get the new workflow run ID
        new_run_id = automation.get_latest_workflow_run_id()

        # Track the new run if we found it
        target_run_id = new_run_id if new_run_id != pre_trigger_run_id else None

        print(f"Waiting for workflow completion (max {max_wait_minutes} minutes)...")
        completion_result = automation.wait_for_completion(
            run_id=target_run_id, max_wait_minutes=max_wait_minutes
        )

        # Merge results
        result_dict = result.to_dict()
        result_dict.update(completion_result.to_dict())
        return result_dict

    return result.to_dict()


# Example usage
if __name__ == "__main__":
    # Test configuration
    GITHUB_TOKEN = "your_token_here"
    ORG = "nestle-it"
    AUTOMATION_REPO = "npus-aiml-utilities-create-repo"

    # Create repository (assumes inputs are already validated)
    result = create_repo_for_agent(
        github_token=GITHUB_TOKEN,
        org=ORG,
        automation_repo=AUTOMATION_REPO,
        use_case_name="test-repo",
        internal_team="sig",
        external_team="none",
        additional_team="none",
        template="npus-aiml-skinny-dab-template",
        wait_for_completion=True,
    )

    print(json.dumps(result, indent=2))
