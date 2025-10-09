import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import sys

sys.path.append("/servers")
from create_repo_utility import create_repo_for_agent
from create_catalog_schema_utility import create_schema_for_agent
from create_compute_utility import create_cluster_for_agent

load_dotenv()

mcp = FastMCP("AIOps-Automation-Server")


@mcp.tool()
def create_github_repository(
    use_case_name: str,
    template: str = "npus-aiml-mlops-stacks-template",
    internal_team: str = "eai",
    development_team: str = "none",
    additional_team: str = "none",
) -> Dict[str, Any]:
    """
    Create a GitHub repository using the automation workflow.

    Args:
        use_case_name: Name of the use case for the repository
        template: Template to use (npus-aiml-mlops-stacks-template, npus-aiml-skinny-dab-template)
        internal_team: Internal team (eai, deloitte, sig, tredence, bora, genpact, tiger, srm, kroger)
        development_team: Development team (eai, deloitte, sig, tredence, bora, genpact, tiger, srm, kroger, 'none')
        additional_team: Additional team (eai, deloitte, sig, tredence, bora, genpact, tiger, srm, kroger, 'none')

    Returns:
        Dict with creation result and status
    """
    try:
        # Get credentials from environment
        GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
        ORG = os.getenv("GITHUB_ORG", "nestle-it")
        AUTOMATION_REPO = os.getenv(
            "AUTOMATION_REPO", "npus-aiml-utilities-create-repo"
        )

        if not GITHUB_TOKEN:
            return {"success": False, "error": "GitHub token not configured"}

        # Call the repository creation utility
        result = create_repo_for_agent(
            github_token=GITHUB_TOKEN,
            org=ORG,
            automation_repo=AUTOMATION_REPO,
            use_case_name=use_case_name,
            template=template,
            internal_team=internal_team,
            development_team=development_team,
            additional_team=additional_team,
            wait_for_completion=True,
        )

        return result

    except Exception as e:
        return {"success": False, "error": f"Failed to create repository: {str(e)}"}


@mcp.tool()
def create_databricks_schema(
    catalog: str,
    schema: str,
    aiml_support_team: str,
    aiml_use_case: str,
    business_owner: str,
    internal_entra_id_group: str = "AAD-SG-NPUS-aiml-internal-contributors",
    external_entra_id_group: str = "none",
) -> Dict[str, Any]:
    """
    Create a Databricks schema and volume with appropriate permissions.

    Args:
        catalog: Databricks catalog name (npus_aiml_workbench, npus_aiml_stage)
        schema: Schema name (e.g., category_forecast, pricing_ppa_tool, etc.)
        aiml_support_team: AIML support team (Digital Manufacturing, EAI, SIG, SRM, AIOps)
        aiml_use_case: AIML use case (retailer_pos_kroger_forecast, Category Forecast, etc.)
        business_owner: Business owner (kroger, SRM, Digital Manufacturing, Enterprise-wide, MDO, ORM, Transportation, AIOps)
        internal_entra_id_group: Internal Entra ID group (default: AAD-SG-NPUS-aiml-internal-contributors)
        external_entra_id_group: External Entra ID group (optional)

    Returns:
        Dict with creation result and status
    """
    try:
        # Get Databricks credentials from environment
        WORKSPACE_URL = os.getenv("DATABRICKS_WORKSPACE_URL")
        TENANT_ID = os.getenv("ARM_TENANT_ID")
        CLIENT_ID = os.getenv("ARM_CLIENT_ID")
        CLIENT_SECRET = os.getenv("ARM_CLIENT_SECRET")

        if not WORKSPACE_URL:
            return {
                "success": False,
                "error": "Databricks workspace URL not configured",
            }

        if not all([TENANT_ID, CLIENT_ID, CLIENT_SECRET]):
            return {
                "success": False,
                "error": "Azure credentials not configured (ARM_TENANT_ID, ARM_CLIENT_ID, ARM_CLIENT_SECRET)",
            }

        # Call the schema creation utility
        result = create_schema_for_agent(
            workspace_url=WORKSPACE_URL,
            catalog=catalog,
            schema=schema,
            aiml_support_team=aiml_support_team,
            aiml_use_case=aiml_use_case,
            business_owner=business_owner,
            internal_entra_id_group=internal_entra_id_group,
            external_entra_id_group=external_entra_id_group,
            tenant_id=TENANT_ID,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
        )

        return result

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to create Databricks schema: {str(e)}",
        }


@mcp.tool()
def create_databricks_compute(
    cluster_name: str,
    spark_version: str = "16.4.x-scala2.12",
    driver_node_type_id: str = "Standard_D4a_v4",
    node_type_id: str = "Standard_D4a_v4",
    min_workers: int = 1,
    max_workers: int = 4,
    data_security_mode: str = "SINGLE_USER",
    aiml_use_case: str = None,
) -> Dict[str, Any]:
    """
    Create a Databricks compute cluster with specified configuration.

    Args:
        cluster_name: Name of the cluster
        spark_version: Spark runtime version (16.4.x-scala2.13, 16.4.x-scala2.12, 15.4.x-scala2.12)
        driver_node_type_id: Driver node type (Standard_D4a_v4, Standard_D8a_v4, Standard_D16a_v4)
        node_type_id: Worker node type (Standard_D4a_v4, Standard_D8a_v4, Standard_D16a_v4)
        min_workers: Minimum number of workers (default: 1)
        max_workers: Maximum number of workers (default: 4)
        data_security_mode: Data security mode (SINGLE_USER)
        aiml_use_case: AIML use case (defaults to cluster_name if not provided)

    Returns:
        Dict with creation result and status
    """
    try:
        # Get Databricks credentials from environment
        WORKSPACE_URL = os.getenv("DATABRICKS_WORKSPACE_URL")
        TENANT_ID = os.getenv("ARM_TENANT_ID")
        CLIENT_ID = os.getenv("ARM_CLIENT_ID")
        CLIENT_SECRET = os.getenv("ARM_CLIENT_SECRET")

        if not WORKSPACE_URL:
            return {
                "success": False,
                "error": "Databricks workspace URL not configured",
            }

        if not all([TENANT_ID, CLIENT_ID, CLIENT_SECRET]):
            return {
                "success": False,
                "error": "Azure credentials not configured (ARM_TENANT_ID, ARM_CLIENT_ID, ARM_CLIENT_SECRET)",
            }

        # Call the compute creation utility
        result = create_cluster_for_agent(
            workspace_url=WORKSPACE_URL,
            cluster_name=cluster_name,
            spark_version=spark_version,
            driver_node_type_id=driver_node_type_id,
            node_type_id=node_type_id,
            min_workers=min_workers,
            max_workers=max_workers,
            data_security_mode=data_security_mode,
            aiml_use_case=aiml_use_case,
            tenant_id=TENANT_ID,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            wait_for_completion=False,
        )

        return result

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to create Databricks compute cluster: {str(e)}",
        }


if __name__ == "__main__":
    mcp.run()
