import os
import time
import logging
from typing import Dict, Any, Optional
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv

from servers.config import ClusterConfig, DEFAULT_CLUSTER_CONFIG

load_dotenv()


class DatabricksComputeManager:
    """
    Simplified class to manage Databricks compute clusters with standardized configuration.
    """

    def __init__(
        self,
        workspace_url: str,
        tenant_id: str = None,
        client_id: str = None,
        client_secret: str = None,
        cluster_config: Optional[ClusterConfig] = None,
    ):
        """
        Initialize the Databricks Compute Manager.

        Args:
            workspace_url: Databricks workspace URL
            tenant_id: Azure tenant ID (can be provided via environment variable ARM_TENANT_ID)
            client_id: Azure client ID (can be provided via environment variable ARM_CLIENT_ID)
            client_secret: Azure client secret (can be provided via environment variable ARM_CLIENT_SECRET)
            cluster_config: ClusterConfig instance (uses default if not provided)
        """
        self.workspace_url = workspace_url
        self.cluster_config = cluster_config or DEFAULT_CLUSTER_CONFIG

        # Setup authentication
        self._setup_auth(tenant_id, client_id, client_secret)

        # Initialize Databricks client
        self.client = WorkspaceClient(
            host=self.workspace_url,
            azure_tenant_id=self.tenant_id,
            azure_client_id=self.client_id,
            azure_client_secret=self.client_secret,
        )

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _setup_auth(self, tenant_id: str, client_id: str, client_secret: str):
        """Setup Azure authentication credentials."""
        self.tenant_id = tenant_id or os.getenv("ARM_TENANT_ID")
        self.client_id = client_id or os.getenv("ARM_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("ARM_CLIENT_SECRET")

        if not all([self.tenant_id, self.client_id, self.client_secret]):
            raise ValueError(
                "Azure credentials must be provided either as parameters or environment variables"
            )

    def create_cluster(
        self,
        cluster_name: str,
        wait_for_completion: bool = True,
    ) -> Dict[str, Any]:
        """
        Create a Databricks compute cluster with standardized configuration.

        Args:
            cluster_name: Name of the cluster (already normalized by ResourceNameGenerator)
            wait_for_completion: Whether to wait for cluster to be ready (default: True)

        Returns:
            Dict containing cluster creation results
        """
        self.logger.info(f"Creating cluster: {cluster_name}")

        try:
            # Check if cluster already exists
            existing_clusters = self.list_clusters()
            for cluster in existing_clusters:
                if cluster["cluster_name"] == cluster_name:
                    return {
                        "success": False,
                        "error": f"Cluster '{cluster_name}' already exists",
                        "existing_cluster": cluster,
                    }

            # Create the cluster using centralized configuration
            response = self.client.clusters.create(
                cluster_name=cluster_name,
                spark_version=self.cluster_config.spark_version,
                node_type_id=self.cluster_config.node_type_id,
                driver_node_type_id=self.cluster_config.driver_node_type_id,
                autoscale=self.cluster_config.get_autoscale(),
                data_security_mode=self.cluster_config.data_security_mode,
                runtime_engine=self.cluster_config.runtime_engine,
                autotermination_minutes=self.cluster_config.autotermination_minutes,
                enable_local_disk_encryption=self.cluster_config.enable_local_disk_encryption,
                custom_tags=self.cluster_config.custom_tags,
            )
            cluster_id = response.cluster_id

            self.logger.info(f"Cluster creation initiated. Cluster ID: {cluster_id}")

            if wait_for_completion:
                cluster_info = self._wait_for_cluster_ready(
                    cluster_id, timeout_minutes=10
                )
            else:
                cluster_info = {}

            result = {
                "success": True,
                "message": f"Successfully created cluster '{cluster_name}'",
                "cluster_id": cluster_id,
                "cluster_name": cluster_name,
                "cluster_state": (
                    cluster_info.get("state", "PENDING") if cluster_info else "PENDING"
                ),
                "configuration": self.cluster_config.to_dict(),
            }

            self.logger.info(f"Successfully created cluster: {result}")
            return result

        except Exception as e:
            error_msg = f"Failed to create cluster: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
            }

    def _wait_for_cluster_ready(
        self, cluster_id: str, timeout_minutes: int = 10
    ) -> Optional[Dict[str, Any]]:
        """Wait for cluster to be in RUNNING state."""
        timeout_seconds = timeout_minutes * 60
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            try:
                cluster_info = self.client.clusters.get(cluster_id)
                state = cluster_info.state.value if cluster_info.state else "UNKNOWN"

                self.logger.info(f"Cluster {cluster_id} state: {state}")

                if state == "RUNNING":
                    return {"state": state, "cluster_info": cluster_info}
                elif state in ["ERROR", "TERMINATED", "TERMINATING"]:
                    self.logger.error(
                        f"Cluster {cluster_id} failed to start. State: {state}"
                    )
                    return {"state": state, "cluster_info": cluster_info}

                time.sleep(30)

            except Exception as e:
                self.logger.warning(f"Error checking cluster status: {str(e)}")
                time.sleep(30)

        self.logger.warning(f"Timeout waiting for cluster {cluster_id} to be ready")
        return None

    def list_clusters(self) -> list[Dict[str, Any]]:
        """List all clusters in the workspace."""
        try:
            clusters = self.client.clusters.list()
            result = []

            for cluster in clusters:
                cluster_dict = {
                    "cluster_id": cluster.cluster_id,
                    "cluster_name": cluster.cluster_name,
                    "state": cluster.state.value if cluster.state else "UNKNOWN",
                }
                result.append(cluster_dict)

            return result

        except Exception as e:
            self.logger.error(f"Failed to list clusters: {str(e)}")
            return []


# Simplified interface for agentic AI
def create_cluster_for_agent(
    workspace_url: str,
    cluster_name: str,
    tenant_id: str = None,
    client_id: str = None,
    client_secret: str = None,
    wait_for_completion: bool = True,
    cluster_config: Optional[ClusterConfig] = None,
) -> Dict[str, Any]:
    """
    Simplified interface for agentic AI to create Databricks compute clusters.

    Args:
        workspace_url: Databricks workspace URL
        cluster_name: Name of the cluster (should be pre-normalized by ResourceNameGenerator)
        tenant_id: Azure tenant ID (optional, can use env var ARM_TENANT_ID)
        client_id: Azure client ID (optional, can use env var ARM_CLIENT_ID)
        client_secret: Azure client secret (optional, can use env var ARM_CLIENT_SECRET)
        wait_for_completion: Whether to wait for cluster to be ready (default: True)
        cluster_config: ClusterConfig instance for custom configuration (uses default if not provided)

    Returns:
        Dict containing creation results
    """
    try:
        compute_manager = DatabricksComputeManager(
            workspace_url=workspace_url,
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
            cluster_config=cluster_config,
        )

        result = compute_manager.create_cluster(
            cluster_name=cluster_name,
            wait_for_completion=wait_for_completion,
        )

        return result

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to initialize Databricks client or create cluster: {str(e)}",
        }


if __name__ == "__main__":
    import json
    from backend.servers.config import ResourceNameGenerator

    # Example usage with proper resource name generation
    print("\nCreating cluster with normalized name...")
    use_case_name = "agent-test-run"
    cluster_name = ResourceNameGenerator.normalize_cluster_name(use_case_name)

    result = create_cluster_for_agent(
        workspace_url="https://adb-1125343200912494.14.azuredatabricks.net/",
        cluster_name=cluster_name,
        wait_for_completion=True,
    )
    print("Result:", json.dumps(result, indent=2))

    # Example: Create cluster with custom configuration
    # print("\nCreating cluster with custom configuration...")
    # custom_config = ClusterConfig(
    #     spark_version="15.4.x-scala2.12",
    #     min_workers=2,
    #     max_workers=10,
    #     autotermination_minutes=120,
    #     custom_tags={"aiml_use_case": "custom_project", "team": "data_science"},
    # )
    #
    # cluster_name = ResourceNameGenerator.normalize_cluster_name("my-custom-cluster")
    # result = create_cluster_for_agent(
    #     workspace_url="https://adb-1125343200912494.14.azuredatabricks.net/",
    #     cluster_name=cluster_name,
    #     wait_for_completion=True,
    #     cluster_config=custom_config,
    # )
    # print("Result:", json.dumps(result, indent=2))
