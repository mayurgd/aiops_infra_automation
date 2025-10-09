import os
from typing import Optional, List, Dict, Any
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.compute import (
    AutoScale,
    DataSecurityMode,
)
import logging
from dotenv import load_dotenv
import time

load_dotenv()


class DatabricksComputeManager:
    """
    A class to manage Databricks compute clusters.
    Provides functionality to create, delete, and manage compute clusters with standardized configurations.
    """

    # Predefined options for validation
    SPARK_VERSIONS = ["16.4.x-scala2.13", "16.4.x-scala2.12", "15.4.x-scala2.12"]

    NODE_TYPES = [
        "Standard_D4a_v4",
        "Standard_D8a_v4",
        "Standard_D16a_v4",
    ]

    DATA_SECURITY_MODES = ["SINGLE_USER"]

    AIML_USE_CASES = ["agent_test_run"]

    def __init__(
        self,
        workspace_url: str,
        tenant_id: str = None,
        client_id: str = None,
        client_secret: str = None,
    ):
        """
        Initialize the Databricks Compute Manager.

        Args:
            workspace_url: Databricks workspace URL
            tenant_id: Azure tenant ID (can be provided via environment variable ARM_TENANT_ID)
            client_id: Azure client ID (can be provided via environment variable ARM_CLIENT_ID)
            client_secret: Azure client secret (can be provided via environment variable ARM_CLIENT_SECRET)
        """
        self.workspace_url = workspace_url

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

    def validate_cluster_inputs(
        self,
        cluster_name: str,
        spark_version: str = "16.4.x-scala2.12",
        driver_node_type_id: str = "Standard_D4a_v4",
        node_type_id: str = "Standard_D4a_v4",
        min_workers: int = 1,
        max_workers: int = 4,
        data_security_mode: str = "SINGLE_USER",
        aiml_use_case: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate all inputs for cluster creation.

        Returns:
            Dict with validation results
        """
        errors = []
        warnings = []

        # Validate required fields
        if not cluster_name or not str(cluster_name).strip():
            errors.append("Parameter 'cluster_name' cannot be empty")

        # Validate cluster name format
        if (
            cluster_name
            and not cluster_name.replace("_", "").replace("-", "").isalnum()
        ):
            errors.append(
                "Cluster name should only contain alphanumeric characters, hyphens, and underscores"
            )

        # Validate spark version
        if spark_version and spark_version not in self.SPARK_VERSIONS:
            errors.append(
                f"Spark version '{spark_version}' is not in predefined options: {self.SPARK_VERSIONS}"
            )

        # Validate node types
        if driver_node_type_id and driver_node_type_id not in self.NODE_TYPES:
            warnings.append(
                f"Driver node type '{driver_node_type_id}' is not in predefined options: {self.NODE_TYPES}"
            )

        if node_type_id and node_type_id not in self.NODE_TYPES:
            warnings.append(
                f"Worker node type '{node_type_id}' is not in predefined options: {self.NODE_TYPES}"
            )

        # Validate autoscale settings
        if min_workers < 0 or max_workers < 0:
            errors.append("min_workers and max_workers must be non-negative integers")

        if min_workers > max_workers:
            errors.append("min_workers cannot be greater than max_workers")

        if max_workers > 100:
            warnings.append(
                "max_workers is set to a high value. Consider if this is necessary for cost optimization"
            )

        # Validate data security mode
        if data_security_mode and data_security_mode not in self.DATA_SECURITY_MODES:
            errors.append(
                f"Data security mode '{data_security_mode}' is not valid. Options: {self.DATA_SECURITY_MODES}"
            )

        # Validate aiml_use_case if provided
        if aiml_use_case and aiml_use_case not in self.AIML_USE_CASES:
            warnings.append(
                f"AIML use case '{aiml_use_case}' is not in predefined options: {self.AIML_USE_CASES}"
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "cluster_name": cluster_name if len(errors) == 0 else None,
        }

    def create_cluster(
        self,
        cluster_name: str,
        spark_version: str = "16.4.x-scala2.12",
        driver_node_type_id: str = "Standard_D4a_v4",
        node_type_id: str = "Standard_D4a_v4",
        min_workers: int = 1,
        max_workers: int = 4,
        data_security_mode: str = "SINGLE_USER",
        aiml_use_case: Optional[str] = None,
        additional_tags: Optional[Dict[str, str]] = None,
        wait_for_completion: bool = True,
    ) -> Dict[str, Any]:
        """
        Create a Databricks compute cluster with the specified configuration.

        Args:
            cluster_name: Name of the cluster
            spark_version: Spark runtime version
            driver_node_type_id: Driver node type
            node_type_id: Worker node type
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
            data_security_mode: Data security mode
            aiml_use_case: AIML use case (defaults to cluster_name if not provided)
            additional_tags: Additional custom tags

        Returns:
            Dict containing cluster creation results
        """
        # Use cluster_name as aiml_use_case if not provided
        if not aiml_use_case:
            aiml_use_case = cluster_name

        # Validate inputs
        validation_result = self.validate_cluster_inputs(
            cluster_name,
            spark_version,
            driver_node_type_id,
            node_type_id,
            min_workers,
            max_workers,
            data_security_mode,
            aiml_use_case,
        )

        if not validation_result["valid"]:
            return {
                "success": False,
                "errors": validation_result["errors"],
                "warnings": validation_result.get("warnings", []),
            }

        # Log warnings if any
        for warning in validation_result.get("warnings", []):
            self.logger.warning(warning)

        # Prepare custom tags
        custom_tags = {"aiml_use_case": aiml_use_case}
        if additional_tags:
            custom_tags.update(additional_tags)

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
                        "warnings": validation_result.get("warnings", []),
                    }

            # Create autoscale configuration
            autoscale = AutoScale(min_workers=min_workers, max_workers=max_workers)

            # Map data security mode string to enum
            security_mode_mapping = {
                "SINGLE_USER": DataSecurityMode.SINGLE_USER,
            }

            # Create the cluster
            response = self.client.clusters.create(
                cluster_name=cluster_name,
                spark_version=spark_version,
                driver_node_type_id=driver_node_type_id,
                node_type_id=node_type_id,
                autoscale=autoscale,
                data_security_mode=security_mode_mapping.get(
                    data_security_mode, DataSecurityMode.SINGLE_USER
                ),
                custom_tags=custom_tags,
            )
            cluster_id = response.cluster_id

            self.logger.info(f"Cluster creation initiated. Cluster ID: {cluster_id}")

            if wait_for_completion:
                # Wait for cluster to be running (optional, can be removed for async creation)
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
                "spark_version": spark_version,
                "driver_node_type": driver_node_type_id,
                "worker_node_type": node_type_id,
                "autoscale": {"min_workers": min_workers, "max_workers": max_workers},
                "data_security_mode": data_security_mode,
                "custom_tags": custom_tags,
                "cluster_state": (
                    cluster_info.get("state", "PENDING") if cluster_info else "PENDING"
                ),
                "warnings": validation_result.get("warnings", []),
            }

            self.logger.info(f"Successfully created cluster: {result}")
            return result

        except Exception as e:
            error_msg = f"Failed to create cluster: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "warnings": validation_result.get("warnings", []),
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

                time.sleep(30)  # Wait 30 seconds before checking again

            except Exception as e:
                self.logger.warning(f"Error checking cluster status: {str(e)}")
                time.sleep(30)

        self.logger.warning(f"Timeout waiting for cluster {cluster_id} to be ready")
        return None

    def list_clusters(self) -> List[Dict[str, Any]]:
        """List all clusters in the workspace."""
        try:
            clusters = self.client.clusters.list()
            result = []

            for cluster in clusters:
                cluster_dict = {
                    "cluster_id": cluster.cluster_id,
                    "cluster_name": cluster.cluster_name,
                    "state": cluster.state.value if cluster.state else "UNKNOWN",
                    "spark_version": cluster.spark_version,
                    "driver_node_type": cluster.driver_node_type_id,
                    "worker_node_type": cluster.node_type_id,
                    "num_workers": getattr(cluster, "num_workers", None),
                    "autoscale": (
                        {
                            "min_workers": (
                                cluster.autoscale.min_workers
                                if cluster.autoscale
                                else None
                            ),
                            "max_workers": (
                                cluster.autoscale.max_workers
                                if cluster.autoscale
                                else None
                            ),
                        }
                        if cluster.autoscale
                        else None
                    ),
                    "custom_tags": cluster.custom_tags or {},
                }
                result.append(cluster_dict)

            return result

        except Exception as e:
            self.logger.error(f"Failed to list clusters: {str(e)}")
            return []

    def delete_cluster(self, cluster_id: str) -> Dict[str, Any]:
        """Delete a cluster by ID."""
        try:
            self.client.clusters.permanent_delete(cluster_id)
            self.logger.info(f"Successfully deleted cluster {cluster_id}")
            return {
                "success": True,
                "message": f"Successfully deleted cluster {cluster_id}",
                "cluster_id": cluster_id,
            }
        except Exception as e:
            error_msg = f"Failed to delete cluster {cluster_id}: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg, "cluster_id": cluster_id}

    def terminate_cluster(self, cluster_id: str) -> Dict[str, Any]:
        """Terminate a cluster by ID (can be restarted)."""
        try:
            self.client.clusters.delete(cluster_id)
            self.logger.info(f"Successfully terminated cluster {cluster_id}")
            return {
                "success": True,
                "message": f"Successfully terminated cluster {cluster_id}",
                "cluster_id": cluster_id,
            }
        except Exception as e:
            error_msg = f"Failed to terminate cluster {cluster_id}: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg, "cluster_id": cluster_id}

    def start_cluster(self, cluster_id: str) -> Dict[str, Any]:
        """Start a terminated cluster."""
        try:
            self.client.clusters.start(cluster_id)
            self.logger.info(f"Successfully started cluster {cluster_id}")
            return {
                "success": True,
                "message": f"Successfully started cluster {cluster_id}",
                "cluster_id": cluster_id,
            }
        except Exception as e:
            error_msg = f"Failed to start cluster {cluster_id}: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg, "cluster_id": cluster_id}


# Simplified interface for agentic AI
def create_cluster_for_agent(
    workspace_url: str,
    cluster_name: str,
    spark_version: str = "16.4.x-scala2.12",
    driver_node_type_id: str = "Standard_D4a_v4",
    node_type_id: str = "Standard_D4a_v4",
    min_workers: int = 1,
    max_workers: int = 4,
    data_security_mode: str = "SINGLE_USER",
    aiml_use_case: Optional[str] = None,
    tenant_id: str = None,
    client_id: str = None,
    client_secret: str = None,
    wait_for_completion: bool = True,
) -> Dict[str, Any]:
    """
    Simplified interface for agentic AI to create Databricks compute clusters.

    Args:
        workspace_url: Databricks workspace URL
        cluster_name: Name of the cluster
        spark_version: Spark runtime version (default: "16.4.x-scala2.12")
        driver_node_type_id: Driver node type (default: "Standard_D4a_v4")
        node_type_id: Worker node type (default: "Standard_D4a_v4")
        min_workers: Minimum number of workers (default: 1)
        max_workers: Maximum number of workers (default: 4)
        data_security_mode: Data security mode (default: "SINGLE_USER")
        aiml_use_case: AIML use case (defaults to cluster_name if not provided)
        auto_termination_minutes: Auto termination time (default: 120)
        tenant_id: Azure tenant ID (optional, can use env var ARM_TENANT_ID)
        client_id: Azure client ID (optional, can use env var ARM_CLIENT_ID)
        client_secret: Azure client secret (optional, can use env var ARM_CLIENT_SECRET)
        wait_for_completion: bool = True

    Returns:
        Dict containing creation results
    """
    try:
        compute_manager = DatabricksComputeManager(
            workspace_url=workspace_url,
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
        )

        result = compute_manager.create_cluster(
            cluster_name=cluster_name,
            spark_version=spark_version,
            driver_node_type_id=driver_node_type_id,
            node_type_id=node_type_id,
            min_workers=min_workers,
            max_workers=max_workers,
            data_security_mode=data_security_mode,
            aiml_use_case=aiml_use_case,
            wait_for_completion=wait_for_completion,
        )

        return result

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to initialize Databricks client or create cluster: {str(e)}",
        }


# Standalone validation function for external use
def validate_cluster_inputs(
    cluster_name: str,
    spark_version: str = "16.4.x-scala2.12",
    driver_node_type_id: str = "Standard_D4a_v4",
    node_type_id: str = "Standard_D4a_v4",
    min_workers: int = 1,
    max_workers: int = 4,
    data_security_mode: str = "SINGLE_USER",
    aiml_use_case: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Standalone validation function that can be used without instantiating the class.

    Args:
        cluster_name: Name of the cluster
        spark_version: Spark runtime version
        driver_node_type_id: Driver node type
        node_type_id: Worker node type
        min_workers: Minimum number of workers
        max_workers: Maximum number of workers
        data_security_mode: Data security mode
        aiml_use_case: AIML use case

    Returns:
        Dict with validation results
    """
    # Create a temporary instance just for validation (no API calls needed)
    temp_manager = DatabricksComputeManager.__new__(DatabricksComputeManager)
    temp_manager.SPARK_VERSIONS = DatabricksComputeManager.SPARK_VERSIONS
    temp_manager.NODE_TYPES = DatabricksComputeManager.NODE_TYPES
    temp_manager.DATA_SECURITY_MODES = DatabricksComputeManager.DATA_SECURITY_MODES
    temp_manager.AIML_USE_CASES = DatabricksComputeManager.AIML_USE_CASES

    return temp_manager.validate_cluster_inputs(
        cluster_name,
        spark_version,
        driver_node_type_id,
        node_type_id,
        min_workers,
        max_workers,
        data_security_mode,
        aiml_use_case,
    )


# Example usage
if __name__ == "__main__":
    import json

    # Create cluster using the simplified interface
    print("Creating compute cluster...")
    result = create_cluster_for_agent(
        workspace_url="https://adb-1125343200912494.14.azuredatabricks.net/",
        cluster_name="agent_test_run",
        spark_version="16.4.x-scala2.12",
        driver_node_type_id="Standard_D4a_v4",
        node_type_id="Standard_D4a_v4",
        min_workers=1,
        max_workers=4,
        data_security_mode="SINGLE_USER",
        aiml_use_case="agent_test_run",  # Will default to cluster_name if not provided
        wait_for_completion=True,
        # Credentials will be picked up from environment variables:
        # ARM_TENANT_ID, ARM_CLIENT_ID, ARM_CLIENT_SECRET
    )

    print("Cluster creation result:", json.dumps(result, indent=2))
