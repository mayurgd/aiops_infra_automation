import os
from typing import Optional, Dict, Any
from databricks.sdk import WorkspaceClient
import logging
from dotenv import load_dotenv

load_dotenv()


class DatabricksResourceManager:
    """
    A class to manage Databricks resources including schemas, volumes, and permissions.
    """

    CATALOG = "npus_aiml_workbench"

    def __init__(
        self,
        workspace_url: str,
        tenant_id: str = None,
        client_id: str = None,
        client_secret: str = None,
    ):
        """
        Initialize the Databricks Resource Manager.

        Args:
            workspace_url: Databricks workspace URL
            tenant_id: Azure tenant ID (can be provided via environment variable ARM_TENANT_ID)
            client_id: Azure client ID (can be provided via environment variable ARM_CLIENT_ID)
            client_secret: Azure client secret (can be provided via environment variable ARM_CLIENT_SECRET)
        """
        self.workspace_url = workspace_url
        self.data_lake_uri = os.getenv("DATA_LAKE_URI")
        self.privileged_group = "AAD-SG-NPUS-aiml-privileged-stg"
        self.volume_name = "files"

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

    def _get_aiml_exploration_path(self, catalog: str, schema_name: str) -> str:
        """Generate the AIML exploration data lake path."""
        aiml_exploration_folder = (
            catalog.split("_")[2] if len(catalog.split("_")) > 2 else "default"
        )
        return f"{self.data_lake_uri}/{aiml_exploration_folder}/{schema_name}/files"

    def _create_directory_if_not_exists(self, path: str):
        """Create directory in data lake if it doesn't exist."""
        try:
            # Try to list the path to check if it exists
            self.client.dbfs.list(path)
            self.logger.info(f"The folder '{path}' already exists.")
        except Exception:
            self.logger.info(f"Creating directory: {path}")
            self.client.dbfs.mkdirs(path)

    def _execute_sql(self, sql_command: str) -> Any:
        """Execute SQL command using Databricks SQL execution context."""
        try:
            result = self.client.statement_execution.execute_statement(
                warehouse_id="51724be1a39f98a1",
                statement=sql_command,
                wait_timeout="30s",
            )
            self.logger.info(f"Executed SQL: {sql_command}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to execute SQL: {sql_command}. Error: {str(e)}")
            raise

    def _grant_permissions_to_group(self, catalog: str, schema: str, group_name: str):
        """Grant all necessary permissions to a group for the schema."""
        permissions = [
            f"GRANT EXECUTE ON SCHEMA {catalog}.{schema} TO `{group_name}`;",
            f"GRANT READ VOLUME ON SCHEMA {catalog}.{schema} TO `{group_name}`;",
            f"GRANT SELECT ON SCHEMA {catalog}.{schema} TO `{group_name}`;",
            f"GRANT MODIFY ON SCHEMA {catalog}.{schema} TO `{group_name}`;",
            f"GRANT REFRESH ON SCHEMA {catalog}.{schema} TO `{group_name}`;",
            f"GRANT WRITE VOLUME ON SCHEMA {catalog}.{schema} TO `{group_name}`;",
            f"GRANT CREATE FUNCTION ON SCHEMA {catalog}.{schema} TO `{group_name}`;",
            f"GRANT CREATE MATERIALIZED VIEW ON SCHEMA {catalog}.{schema} TO `{group_name}`;",
            f"GRANT CREATE MODEL ON SCHEMA {catalog}.{schema} TO `{group_name}`;",
            f"GRANT CREATE MODEL VERSION ON SCHEMA {catalog}.{schema} TO `{group_name}`;",
            f"GRANT CREATE TABLE ON SCHEMA {catalog}.{schema} TO `{group_name}`;",
            f"GRANT CREATE VOLUME ON SCHEMA {catalog}.{schema} TO `{group_name}`;",
        ]

        for permission in permissions:
            self._execute_sql(permission)

    def create_schema_and_volume(
        self,
        schema_name: str,
        internal_entra_id_group: str,
        external_entra_id_group: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create Databricks schema and volume with appropriate permissions.

        All inputs are assumed to be already validated and normalized by the server.

        Args:
            schema_name: Schema name (already normalized with underscores)
            internal_entra_id_group: Internal Entra ID group (already validated)
            external_entra_id_group: External Entra ID group (optional, already validated)

        Returns:
            Dict containing created resources information
        """
        catalog = self.CATALOG

        # Generate paths and tags
        aiml_exploration_path = self._get_aiml_exploration_path(catalog, schema_name)
        schema_tags = f"'aiml_use_case' = '{schema_name}'"

        self.logger.info(
            f"Creating resources for catalog: {catalog}, schema: {schema_name}"
        )

        try:
            # Create directory
            self._create_directory_if_not_exists(aiml_exploration_path)

            # Create schema
            self._execute_sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema_name};")
            self._execute_sql(
                f"ALTER SCHEMA {catalog}.{schema_name} SET OWNER TO `{self.privileged_group}`;"
            )
            self._execute_sql(
                f"ALTER SCHEMA {catalog}.{schema_name} SET TAGS ({schema_tags});"
            )

            # Create volume
            self._execute_sql(
                f"CREATE EXTERNAL VOLUME IF NOT EXISTS {catalog}.{schema_name}.{self.volume_name} LOCATION '{aiml_exploration_path}';"
            )
            self._execute_sql(
                f"ALTER VOLUME {catalog}.{schema_name}.{self.volume_name} SET OWNER TO `{self.privileged_group}`;"
            )

            # Grant permissions to internal group
            self._grant_permissions_to_group(
                catalog, schema_name, internal_entra_id_group
            )

            # Grant permissions to external group if provided
            if external_entra_id_group:
                self._grant_permissions_to_group(
                    catalog, schema_name, external_entra_id_group
                )

            result = {
                "success": True,
                "message": f"Successfully created schema and volume for {catalog}.{schema_name}",
                "catalog": catalog,
                "schema": schema_name,
                "volume_name": self.volume_name,
                "full_schema_name": f"{catalog}.{schema_name}",
                "full_volume_name": f"{catalog}.{schema_name}.{self.volume_name}",
                "aiml_exploration_path": aiml_exploration_path,
                "privileged_group": self.privileged_group,
                "internal_group": internal_entra_id_group,
                "external_group": external_entra_id_group,
                "tags": {
                    "aiml_use_case": schema_name,
                },
            }

            self.logger.info(f"Successfully created resources: {result}")
            return result

        except Exception as e:
            error_msg = f"Failed to create resources: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
            }


def create_schema_for_agent(
    workspace_url: str,
    schema_name: str,
    internal_entra_id_group: str,
    external_entra_id_group: Optional[str] = None,
    tenant_id: str = None,
    client_id: str = None,
    client_secret: str = None,
) -> Dict[str, Any]:
    """
    Create Databricks schema and volume with appropriate permissions.

    This function is called by the server after validation and normalization is complete.
    All inputs are assumed to be already validated and normalized.

    Args:
        workspace_url: Databricks workspace URL
        schema_name: Schema name (already normalized with underscores)
        internal_entra_id_group: Internal Entra ID group (validated)
        external_entra_id_group: External Entra ID group (optional, validated)
        tenant_id: Azure tenant ID (optional, can use env var ARM_TENANT_ID)
        client_id: Azure client ID (optional, can use env var ARM_CLIENT_ID)
        client_secret: Azure client secret (optional, can use env var ARM_CLIENT_SECRET)

    Returns:
        Dict containing creation results
    """
    try:
        resource_manager = DatabricksResourceManager(
            workspace_url=workspace_url,
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
        )

        result = resource_manager.create_schema_and_volume(
            schema_name=schema_name,
            internal_entra_id_group=internal_entra_id_group,
            external_entra_id_group=external_entra_id_group,
        )

        return result

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to initialize Databricks client or create resources: {str(e)}",
        }


# Example usage
if __name__ == "__main__":
    import json

    print("Creating schema and volume...")
    result = create_schema_for_agent(
        workspace_url="https://adb-1125343200912494.14.azuredatabricks.net/",
        schema_name="test_schema_creation",  # Already normalized
        internal_entra_id_group="AAD-SG-NPUS-aiml-SIG-contributors",
        external_entra_id_group=None,
        # Credentials will be picked up from environment variables:
        # ARM_TENANT_ID, ARM_CLIENT_ID, ARM_CLIENT_SECRET
    )

    print("Creation result:", json.dumps(result, indent=2))
