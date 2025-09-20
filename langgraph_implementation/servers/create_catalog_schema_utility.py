import os
from typing import Optional, List, Dict, Any
from databricks.sdk import WorkspaceClient
import logging
from dotenv import load_dotenv

load_dotenv()


class DatabricksResourceManager:
    """
    A class to manage Databricks resources including schemas, volumes, and permissions.
    Replicates the functionality from the provided Databricks notebook.
    """

    # Predefined dropdown options from the notebook
    CATALOG_OPTIONS = ["npus_aiml_workbench", "npus_aiml_stage"]
    SCHEMA_OPTIONS = [
        "category_forecast",
        "retailer_pos_kroger_forecast",
        "move37",
        "pricing_ppa_tool",
        "on_time_anomaly_detector",
        "sales_order_cleansing",
        "osprey_consumption",
        "order_fulfillment_smart_add_back",
        "order_add_back_recommendation",
        "predictive_maintenance",
        "price_elasticity",
        "sku_rationalization",
        "supply_demand_causal_inference",
        "sync_planning",
        "pm_optimization_poc",
        "pm_asset_anomaly_poc",
        "agent_test_run",
    ]
    AIML_SUPPORT_TEAMS = ["Digital Manufacturing", "EAI", "SIG", "SRM", "AIOps"]
    AIML_USE_CASES = [
        "retailer_pos_kroger_forecast",
        "Category Forecast",
        "Price Pack Architecture Tool",
        "Move 37",
        "On Time Anomaly Detection",
        "Order Add Back Recommendation",
        "order_fulfillment_smart_add_back",
        "Predictive Maintenance",
        "Price Elasticity",
        "SKU Rationalization",
        "Supply vs Demand Causal Inference",
        "Synchronized Planning",
        "agent_test_run",
    ]
    BUSINESS_OWNERS = [
        "kroger",
        "SRM",
        "Digital Manufacturing",
        "Enterprise-wide",
        "MDO",
        "ORM",
        "Transportation",
        "AIOps",
    ]
    INTERNAL_GROUPS = ["AAD-SG-NPUS-aiml-internal-contributors"]
    EXTERNAL_GROUPS = [
        "AAD-SG-NPUS-aiml-Kroger-team-contributors",
        "AAD-SG-NPUS-aiml-Deloitte-contributors",
        "AAD-SG-NPUS-aiml-Tredence-contributors",
        "AAD-SG-NPUS-AIML-Genpact-CONTRIBUTOR",
        "none",
    ]

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

    def validate_schema_inputs(
        self,
        catalog: str,
        schema: str,
        aiml_support_team: str,
        aiml_use_case: str,
        business_owner: str,
        internal_entra_id_group: str,
        external_entra_id_group: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate all inputs for schema and volume creation.

        Returns:
            Dict with validation results
        """
        errors = []
        warnings = []

        # Validate required fields
        required_fields = {
            "catalog": catalog,
            "schema": schema,
            "aiml_support_team": aiml_support_team,
            "aiml_use_case": aiml_use_case,
            "business_owner": business_owner,
            "internal_entra_id_group": internal_entra_id_group,
        }

        for field_name, field_value in required_fields.items():
            if not field_value or not str(field_value).strip():
                errors.append(f"Parameter '{field_name}' cannot be empty")

        # Validate against predefined options
        validations = [
            (catalog, self.CATALOG_OPTIONS, "catalog"),
            (schema, self.SCHEMA_OPTIONS, "schema"),
            (aiml_support_team, self.AIML_SUPPORT_TEAMS, "aiml_support_team"),
            (aiml_use_case, self.AIML_USE_CASES, "aiml_use_case"),
            (business_owner, self.BUSINESS_OWNERS, "business_owner"),
            (internal_entra_id_group, self.INTERNAL_GROUPS, "internal_entra_id_group"),
        ]

        for value, options, param_name in validations:
            if value and value not in options:
                errors.append(
                    f"Value '{value}' for '{param_name}' is not in predefined options: {options}"
                )

        # Validate external group if provided
        if (
            external_entra_id_group
            and external_entra_id_group not in self.EXTERNAL_GROUPS
        ):
            warnings.append(
                f"Value '{external_entra_id_group}' for 'external_entra_id_group' is not in predefined options: {self.EXTERNAL_GROUPS}"
            )

        # Validate schema name format
        if schema and not schema.replace("_", "").replace("-", "").isalnum():
            errors.append(
                "Schema name should only contain alphanumeric characters, hyphens, and underscores"
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "expected_volume_path": (
                self._get_aiml_exploration_path(catalog, schema)
                if len(errors) == 0
                else None
            ),
            "expected_full_schema_name": (
                f"{catalog}.{schema}"
                if len(errors) == 0 and catalog and schema
                else None
            ),
        }

    def _get_aiml_exploration_path(self, catalog: str, schema: str) -> str:
        """Generate the AIML exploration data lake path."""
        aiml_exploration_folder = (
            catalog.split("_")[2] if len(catalog.split("_")) > 2 else "default"
        )
        return f"abfss://restricted-aimlexploration@npusprdatalakesta.dfs.core.windows.net/{aiml_exploration_folder}/{schema}/external_files"

    def _create_directory_if_not_exists(self, path: str):
        """Create directory in data lake if it doesn't exist."""
        try:
            # Try to list the path to check if it exists
            self.client.dbfs.list(path)
            self.logger.info(f"The folder '{path}' already exists.")
        except Exception as e:
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

    def create_schema_and_volume(
        self,
        catalog: str,
        schema: str,
        aiml_support_team: str,
        aiml_use_case: str,
        business_owner: str,
        internal_entra_id_group: str,
        external_entra_id_group: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create Databricks schema and volume with appropriate permissions.

        Args:
            catalog: Databricks catalog name
            schema: Schema name
            aiml_support_team: AIML support team
            aiml_use_case: AIML use case
            business_owner: Business owner
            internal_entra_id_group: Internal Entra ID group
            external_entra_id_group: External Entra ID group (optional)

        Returns:
            Dict containing created resources information
        """
        # Use the centralized validation function
        validation_result = self.validate_schema_inputs(
            catalog,
            schema,
            aiml_support_team,
            aiml_use_case,
            business_owner,
            internal_entra_id_group,
            external_entra_id_group,
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

        # Generate paths and tags
        aiml_exploration_path = self._get_aiml_exploration_path(catalog, schema)
        schema_tags = f"'aiml_support_team' = '{aiml_support_team}','aiml_use_case' = '{aiml_use_case}', 'business_owner' = '{business_owner}'"

        self.logger.info(f"Creating resources for catalog: {catalog}, schema: {schema}")

        try:
            # Create directory
            self._create_directory_if_not_exists(aiml_exploration_path)

            # Create schema
            self._execute_sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema};")
            self._execute_sql(
                f"ALTER SCHEMA {catalog}.{schema} SET OWNER TO `{self.privileged_group}`;"
            )
            self._execute_sql(
                f"ALTER SCHEMA {catalog}.{schema} SET TAGS ({schema_tags});"
            )

            # Create volume
            self._execute_sql(
                f"CREATE EXTERNAL VOLUME IF NOT EXISTS {catalog}.{schema}.{self.volume_name} LOCATION '{aiml_exploration_path}';"
            )
            self._execute_sql(
                f"ALTER VOLUME {catalog}.{schema}.{self.volume_name} SET OWNER TO `{self.privileged_group}`;"
            )

            # Grant permissions to internal group
            self._grant_permissions_to_group(catalog, schema, internal_entra_id_group)

            # Grant permissions to external group if provided
            if external_entra_id_group:
                self._grant_permissions_to_group(
                    catalog, schema, external_entra_id_group
                )

            result = {
                "success": True,
                "message": f"Successfully created schema and volume for {catalog}.{schema}",
                "catalog": catalog,
                "schema": schema,
                "volume_name": self.volume_name,
                "full_schema_name": f"{catalog}.{schema}",
                "full_volume_name": f"{catalog}.{schema}.{self.volume_name}",
                "aiml_exploration_path": aiml_exploration_path,
                "privileged_group": self.privileged_group,
                "internal_group": internal_entra_id_group,
                "external_group": external_entra_id_group,
                "tags": {
                    "aiml_support_team": aiml_support_team,
                    "aiml_use_case": aiml_use_case,
                    "business_owner": business_owner,
                },
                "warnings": validation_result.get("warnings", []),
            }

            self.logger.info(f"Successfully created resources: {result}")
            return result

        except Exception as e:
            error_msg = f"Failed to create resources: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "warnings": validation_result.get("warnings", []),
            }

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

    def list_schemas(self, catalog: str) -> List[Dict[str, Any]]:
        """List all schemas in a catalog."""
        try:
            schemas = self.client.schemas.list(catalog_name=catalog)
            return [
                {"name": schema.name, "catalog": schema.catalog_name}
                for schema in schemas
            ]
        except Exception as e:
            self.logger.error(f"Failed to list schemas for catalog {catalog}: {str(e)}")
            return []

    def list_volumes(self, catalog: str, schema: str) -> List[Dict[str, Any]]:
        """List all volumes in a schema."""
        try:
            volumes = self.client.volumes.list(catalog_name=catalog, schema_name=schema)
            return [
                {"name": volume.name, "volume_type": volume.volume_type}
                for volume in volumes
            ]
        except Exception as e:
            self.logger.error(
                f"Failed to list volumes for {catalog}.{schema}: {str(e)}"
            )
            return []

    def delete_schema(self, catalog: str, schema: str, cascade: bool = False):
        """Delete a schema."""
        try:
            self._execute_sql(
                f"DROP SCHEMA {'CASCADE' if cascade else ''} {catalog}.{schema};"
            )
            self.logger.info(f"Successfully deleted schema {catalog}.{schema}")
        except Exception as e:
            self.logger.error(f"Failed to delete schema {catalog}.{schema}: {str(e)}")
            raise

    def delete_volume(self, catalog: str, schema: str, volume: str):
        """Delete a volume."""
        try:
            self._execute_sql(f"DROP VOLUME {catalog}.{schema}.{volume};")
            self.logger.info(f"Successfully deleted volume {catalog}.{schema}.{volume}")
        except Exception as e:
            self.logger.error(
                f"Failed to delete volume {catalog}.{schema}.{volume}: {str(e)}"
            )
            raise


# Simplified interface for agentic AI
def create_schema_for_agent(
    workspace_url: str,
    catalog: str,
    schema: str,
    aiml_support_team: str = "AIOps",
    aiml_use_case: str = "agent_test_run",
    business_owner: str = "AIOps",
    internal_entra_id_group: str = "AAD-SG-NPUS-aiml-internal-contributors",
    external_entra_id_group: Optional[str] = None,
    tenant_id: str = None,
    client_id: str = None,
    client_secret: str = None,
) -> Dict[str, Any]:
    """
    Simplified interface for agentic AI to create Databricks schemas and volumes.

    Args:
        workspace_url: Databricks workspace URL
        catalog: Databricks catalog name
        schema: Schema name
        aiml_support_team: AIML support team (default: "AIOps")
        aiml_use_case: AIML use case (default: "agent_test_run")
        business_owner: Business owner (default: "AIOps")
        internal_entra_id_group: Internal Entra ID group
        external_entra_id_group: External Entra ID group (optional)
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
            catalog=catalog,
            schema=schema,
            aiml_support_team=aiml_support_team,
            aiml_use_case=aiml_use_case,
            business_owner=business_owner,
            internal_entra_id_group=internal_entra_id_group,
            external_entra_id_group=external_entra_id_group,
        )

        return result

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to initialize Databricks client or create resources: {str(e)}",
        }


# Standalone validation function for external use
def validate_schema_inputs(
    catalog: str,
    schema: str,
    aiml_support_team: str,
    aiml_use_case: str,
    business_owner: str,
    internal_entra_id_group: str,
    external_entra_id_group: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Standalone validation function that can be used without instantiating the class.

    Args:
        catalog: Databricks catalog name
        schema: Schema name
        aiml_support_team: AIML support team
        aiml_use_case: AIML use case
        business_owner: Business owner
        internal_entra_id_group: Internal Entra ID group
        external_entra_id_group: External Entra ID group (optional)

    Returns:
        Dict with validation results
    """
    # Create a temporary instance just for validation (no API calls needed)
    temp_manager = DatabricksResourceManager.__new__(DatabricksResourceManager)
    temp_manager.CATALOG_OPTIONS = DatabricksResourceManager.CATALOG_OPTIONS
    temp_manager.SCHEMA_OPTIONS = DatabricksResourceManager.SCHEMA_OPTIONS
    temp_manager.AIML_SUPPORT_TEAMS = DatabricksResourceManager.AIML_SUPPORT_TEAMS
    temp_manager.AIML_USE_CASES = DatabricksResourceManager.AIML_USE_CASES
    temp_manager.BUSINESS_OWNERS = DatabricksResourceManager.BUSINESS_OWNERS
    temp_manager.INTERNAL_GROUPS = DatabricksResourceManager.INTERNAL_GROUPS
    temp_manager.EXTERNAL_GROUPS = DatabricksResourceManager.EXTERNAL_GROUPS
    temp_manager._get_aiml_exploration_path = (
        DatabricksResourceManager._get_aiml_exploration_path
    )

    return temp_manager.validate_schema_inputs(
        catalog,
        schema,
        aiml_support_team,
        aiml_use_case,
        business_owner,
        internal_entra_id_group,
        external_entra_id_group,
    )


# Example usage
if __name__ == "__main__":
    import json

    # Create schema and volume using the simplified interface
    # (validation is handled internally by create_schema_and_volume)
    print("Creating schema and volume...")
    result = create_schema_for_agent(
        workspace_url="https://adb-1125343200912494.14.azuredatabricks.net/",
        catalog="npus_aiml_workbench",
        schema="agent_test_run",
        aiml_support_team="AIOps",
        aiml_use_case="agent_test_run",
        business_owner="AIOps",
        internal_entra_id_group="AAD-SG-NPUS-aiml-internal-contributors",
        external_entra_id_group=None,
        # Credentials will be picked up from environment variables:
        # ARM_TENANT_ID, ARM_CLIENT_ID, ARM_CLIENT_SECRET
    )

    print("Creation result:", json.dumps(result, indent=2))
