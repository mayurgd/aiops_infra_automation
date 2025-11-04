import os
import re
from typing import Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
from databricks.sdk.service.compute import (
    AutoScale,
    DataSecurityMode,
    RuntimeEngine,
)

# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================


class TeamType(str, Enum):
    """Type of team - internal or external"""

    INTERNAL = "internal"
    EXTERNAL = "external"


class InternalTeam(str, Enum):
    """Available internal AI/ML teams"""

    SIG = "sig"
    SRM = "srm"
    BORA = "bora"
    DEMAND_PLANNING = "dp"
    EAI = "eai"
    KROGER_TEAM = "kroger"


class ExternalTeam(str, Enum):
    """Available external AI/ML teams"""

    NONE = "none"
    DELOITTE = "deloitte"
    GENPACT = "genpact"
    TIGER_ANALYTICS = "tiger"
    TREDENCE = "tredence"


class TemplateType(str, Enum):
    """Available repository templates"""

    MLOPS_STACKS = "npus-aiml-mlops-stacks-template"
    SKINNY_DAB = "npus-aiml-skinny-dab-template"


# ============================================================================
# CLUSTER CONFIGURATION
# ============================================================================


class ClusterConfig(BaseModel):
    """
    Centralized configuration for Databricks cluster creation.
    Update these values to change the standard cluster configuration.
    """

    # Spark Configuration
    spark_version: str = Field(
        default="16.4.x-scala2.13", description="Databricks runtime version"
    )
    runtime_engine: RuntimeEngine = Field(
        default=RuntimeEngine.STANDARD, description="Runtime engine type"
    )

    # Node Configuration
    node_type_id: str = Field(
        default="Standard_D4ds_v5", description="Worker node type"
    )
    driver_node_type_id: str = Field(
        default="Standard_D4ds_v5", description="Driver node type"
    )

    # Autoscaling Configuration
    min_workers: int = Field(default=1, ge=1, description="Minimum number of workers")
    max_workers: int = Field(default=8, ge=1, description="Maximum number of workers")

    # Security Configuration
    data_security_mode: DataSecurityMode = Field(
        default=DataSecurityMode.USER_ISOLATION, description="Data security mode"
    )
    enable_local_disk_encryption: bool = Field(
        default=False, description="Enable local disk encryption"
    )

    # Cluster Lifecycle Configuration
    autotermination_minutes: int = Field(
        default=60, ge=0, description="Auto-termination timeout in minutes"
    )

    # Tags Configuration
    custom_tags: Dict[str, str] = Field(
        default_factory=lambda: {"aiml_use_case": "mlops_platform"},
        description="Custom tags for cluster",
    )

    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True

    def get_autoscale(self) -> AutoScale:
        """Get AutoScale configuration object."""
        return AutoScale(min_workers=self.min_workers, max_workers=self.max_workers)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging/reporting."""
        return {
            "spark_version": self.spark_version,
            "runtime_engine": (
                self.runtime_engine.value
                if hasattr(self.runtime_engine, "value")
                else str(self.runtime_engine)
            ),
            "node_type": self.node_type_id,
            "driver_node_type": self.driver_node_type_id,
            "autoscale": {
                "min_workers": self.min_workers,
                "max_workers": self.max_workers,
            },
            "data_security_mode": (
                self.data_security_mode.value
                if hasattr(self.data_security_mode, "value")
                else str(self.data_security_mode)
            ),
            "autotermination_minutes": self.autotermination_minutes,
            "enable_local_disk_encryption": self.enable_local_disk_encryption,
            "custom_tags": self.custom_tags,
        }


# Default cluster configuration instance
DEFAULT_CLUSTER_CONFIG = ClusterConfig()


# ============================================================================
# PYDANTIC MODELS
# ============================================================================


class TeamMapping(BaseModel):
    """Configuration for a team's access groups"""

    github_group: str = Field(..., description="GitHub team name for repository access")
    entra_id_group: str = Field(..., description="Entra ID group for Databricks access")
    type: TeamType = Field(..., description="Team type - internal or external")


class TeamMappingsConfig(BaseModel):
    """Complete team mappings configuration"""

    sig: TeamMapping = TeamMapping(
        github_group="npus-aiml-sig-write",
        entra_id_group="AAD-SG-NPUS-aiml-SIG-contributors",
        type=TeamType.INTERNAL,
    )
    srm: TeamMapping = TeamMapping(
        github_group="npus-aiml-srm-write",
        entra_id_group="AAD-SG-NPUS-aiml-SRM-contributors",
        type=TeamType.INTERNAL,
    )
    dp: TeamMapping = Field(
        default=TeamMapping(
            github_group="npus-aiml-dp-write",
            entra_id_group="AAD-SG-NPUS-aiml-DP-contributors",
            type=TeamType.INTERNAL,
        ),
    )
    eai: TeamMapping = TeamMapping(
        github_group="npus-aiml-eai-write",
        entra_id_group="AAD-SG-NPUS-aiml-EAI-contributors",
        type=TeamType.INTERNAL,
    )
    kroger: TeamMapping = Field(
        default=TeamMapping(
            github_group="npus-aiml-kroger-write",
            entra_id_group="AAD-SG-NPUS-aiml-Kroger-team-contributors",
            type=TeamType.INTERNAL,
        ),
    )
    bora: TeamMapping = Field(
        default=TeamMapping(
            github_group="npus-aiml-bora-write",
            entra_id_group="AAD-SG-NPUS-aiml-BORA-contributors",
            type=TeamType.INTERNAL,
        ),
        alias="kroger-team",
    )
    deloitte: TeamMapping = TeamMapping(
        github_group="npus-aiml-deloitte-write",
        entra_id_group="AAD-SG-NPUS-aiml-Deloitte-contributors",
        type=TeamType.EXTERNAL,
    )
    genpact: TeamMapping = TeamMapping(
        github_group="npus-aiml-genpact-write",
        entra_id_group="AAD-SG-NPUS-aiml-Genpact-contributors",
        type=TeamType.EXTERNAL,
    )
    tiger: TeamMapping = Field(
        default=TeamMapping(
            github_group="npus-aiml-tiger-write",
            entra_id_group="AAD-SG-NPUS-aiml-Tiger-contributors",
            type=TeamType.EXTERNAL,
        ),
    )
    tredence: TeamMapping = TeamMapping(
        github_group="npus-aiml-tredence-write",
        entra_id_group="AAD-SG-NPUS-aiml-Tredence-contributors",
        type=TeamType.EXTERNAL,
    )

    class Config:
        populate_by_name = True

    def get_mapping(self, team_key: str) -> Optional[TeamMapping]:
        """Get team mapping by key"""
        return getattr(self, team_key.replace("-", "_"), None)


class ResourceNames(BaseModel):
    """Generated resource names for MLOps resources"""

    repo_name: str = Field(..., description="GitHub repository name")
    schema_name: str = Field(..., description="Databricks Unity Catalog schema name")
    compute_name: str = Field(..., description="Databricks compute cluster name")


class TeamInfo(BaseModel):
    """Information about a team in the context of onboarding"""

    name: str = Field(..., description="Original team name provided")
    normalized: str = Field(..., description="Normalized team key")
    github_group: str = Field(..., description="GitHub group for repository access")
    entra_id_group: str = Field(..., description="Entra ID group for Databricks access")


class OnboardingInput(BaseModel):
    """Input parameters for MLOps use case onboarding"""

    use_case_name: str = Field(
        ...,
        min_length=1,
        max_length=40,
        description="Name of the use case (max 40 characters)",
    )
    internal_team: str = Field(
        ..., description="Internal AI/ML team name (case-insensitive)"
    )
    external_team: str = Field(
        default="none", description="External AI/ML team name (optional)"
    )
    additional_team: str = Field(
        default="none", description="Additional internal practitioner team (optional)"
    )
    requires_model_registration: bool = Field(
        default=False,
        description="Whether use case requires model registration in Unity Catalog",
    )

    # Normalized team values (computed)
    internal_team_normalized: Optional[str] = None
    external_team_normalized: Optional[str] = None
    additional_team_normalized: Optional[str] = None

    @staticmethod
    def normalize_team_name(team_name: str) -> str:
        """Normalize team name to match mappings"""
        normalized = team_name.lower().strip()

        # Handle special cases
        if "demand" in normalized and "planning" in normalized:
            return "dp"
        if "tiger" in normalized:
            return "tiger"
        if "kroger" in normalized:
            return "kroger"

        return normalized.replace(" ", "-")

    @field_validator("use_case_name")
    @classmethod
    def validate_use_case_name(cls, v: str) -> str:
        """Validate use case name format"""
        if not v.strip():
            raise ValueError("Use case name cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def normalize_and_validate_teams(self) -> "OnboardingInput":
        """Normalize team names and validate against available teams"""
        team_config = TEAM_MAPPINGS_CONFIG

        # Normalize internal team
        self.internal_team_normalized = self.normalize_team_name(self.internal_team)
        if not team_config.get_mapping(self.internal_team_normalized):
            valid_teams = [t.value for t in InternalTeam]
            raise ValueError(
                f"Invalid internal team: '{self.internal_team}'. "
                f"Must be one of: {', '.join(valid_teams)}"
            )

        # Check if internal team is actually internal
        mapping = team_config.get_mapping(self.internal_team_normalized)
        if mapping and mapping.type != TeamType.INTERNAL:
            raise ValueError(
                f"'{self.internal_team}' is not an internal team. "
                f"Please use an internal team for the internal_team parameter."
            )

        # Normalize external team
        if self.external_team.lower() != "none":
            self.external_team_normalized = self.normalize_team_name(self.external_team)
            if not team_config.get_mapping(self.external_team_normalized):
                valid_teams = [t.value for t in ExternalTeam if t.value != "none"]
                raise ValueError(
                    f"Invalid external team: '{self.external_team}'. "
                    f"Must be one of: none, {', '.join(valid_teams)}"
                )

            # Check if external team is actually external
            ext_mapping = team_config.get_mapping(self.external_team_normalized)
            if ext_mapping and ext_mapping.type != TeamType.EXTERNAL:
                raise ValueError(
                    f"'{self.external_team}' is not an external team. "
                    f"Please use an external team for the external_team parameter."
                )
        else:
            self.external_team_normalized = "none"

        # Normalize additional team
        if self.additional_team.lower() != "none":
            self.additional_team_normalized = self.normalize_team_name(
                self.additional_team
            )
            if not team_config.get_mapping(self.additional_team_normalized):
                valid_teams = [t.value for t in InternalTeam]
                raise ValueError(
                    f"Invalid additional team: '{self.additional_team}'. "
                    f"Must be one of: none, {', '.join(valid_teams)}"
                )

            # Check if additional team is internal
            add_mapping = team_config.get_mapping(self.additional_team_normalized)
            if add_mapping and add_mapping.type != TeamType.INTERNAL:
                raise ValueError(
                    f"'{self.additional_team}' is not an internal team. "
                    f"Additional team must be internal."
                )
        else:
            self.additional_team_normalized = "none"

        return self


class OnboardingResult(BaseModel):
    """Result of MLOps onboarding operation"""

    success: bool = Field(..., description="Whether onboarding completed successfully")
    use_case_name: str = Field(..., description="Original use case name")
    resource_names: ResourceNames = Field(..., description="Generated resource names")
    template_used: TemplateType = Field(..., description="Repository template used")
    requires_model_registration: bool = Field(
        ..., description="Model registration requirement"
    )
    team_mappings: Dict[str, TeamInfo] = Field(..., description="Applied team mappings")
    github_repo: Optional[Dict[str, Any]] = Field(
        None, description="GitHub repo creation result"
    )
    databricks_schema: Optional[Dict[str, Any]] = Field(
        None, description="Schema creation result"
    )
    databricks_compute: Optional[Dict[str, Any]] = Field(
        None, description="Compute creation result"
    )
    warnings: list[str] = Field(default_factory=list, description="Warnings or notes")
    message: Optional[str] = Field(None, description="Summary message")
    error: Optional[str] = Field(None, description="Error message if failed")
    formatted_output: str = Field(..., description="Results formattd in clear format")

    class Config:
        use_enum_values = True


class EnvironmentConfig(BaseModel):
    """Environment configuration for external services"""

    github_token: str = Field(..., description="GitHub personal access token")
    github_org: str = Field(default="nestle-it", description="GitHub organization")
    automation_repo: str = Field(
        default="npus-aiml-utilities-create-repo",
        description="Automation repository name",
    )
    workspace_url: str = Field(..., description="Databricks workspace URL")
    tenant_id: str = Field(..., description="Azure tenant ID")
    client_id: str = Field(..., description="Azure client ID")
    client_secret: str = Field(..., description="Azure client secret")

    @classmethod
    def from_env(cls) -> "EnvironmentConfig":
        """Load configuration from environment variables"""
        return cls(
            github_token=os.getenv("GITHUB_TOKEN", ""),
            github_org=os.getenv("GITHUB_ORG", "nestle-it"),
            automation_repo=os.getenv(
                "AUTOMATION_REPO", "npus-aiml-utilities-create-repo"
            ),
            workspace_url=os.getenv("DATABRICKS_WORKSPACE_URL", ""),
            tenant_id=os.getenv("ARM_TENANT_ID", ""),
            client_id=os.getenv("ARM_CLIENT_ID", ""),
            client_secret=os.getenv("ARM_CLIENT_SECRET", ""),
        )

    @model_validator(mode="after")
    def validate_credentials(self) -> "EnvironmentConfig":
        """Validate that all required credentials are present"""
        if not self.github_token:
            raise ValueError("GITHUB_TOKEN environment variable is required")

        if not self.workspace_url:
            raise ValueError(
                "DATABRICKS_WORKSPACE_URL environment variable is required"
            )

        if not all([self.tenant_id, self.client_id, self.client_secret]):
            raise ValueError(
                "Azure credentials required: ARM_TENANT_ID, ARM_CLIENT_ID, ARM_CLIENT_SECRET"
            )

        return self


# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

TEAM_MAPPINGS_CONFIG = TeamMappingsConfig()


# ============================================================================
# RESOURCE NAME GENERATION
# ============================================================================


class ResourceNameGenerator:
    """Generator for standardized MLOps resource names"""

    @staticmethod
    def normalize_schema_name(use_case_name: str) -> str:
        """Normalize Unity Catalog schema name"""
        return re.sub(r"[^a-z0-9]+", "_", use_case_name.lower()).strip("_")

    @staticmethod
    def normalize_repo_name(use_case_name: str) -> str:
        """Normalize git repository name"""
        return re.sub(r"[^a-z0-9]+", "-", use_case_name.lower().strip()).strip("-")

    @staticmethod
    def normalize_cluster_name(use_case_name: str) -> str:
        """Normalize Databricks cluster name"""
        name = re.sub(r"[^a-z0-9]+", "_", use_case_name.lower()).strip("_")
        name = f"cluster_{name}" if not name.startswith("cluster_") else name
        name = f"{name}_standard" if not name.endswith("_standard") else name
        return name

    @classmethod
    def generate_all(cls, use_case_name: str) -> ResourceNames:
        """Generate all standardized resource names"""
        return ResourceNames(
            repo_name=cls.normalize_repo_name(use_case_name),
            schema_name=cls.normalize_schema_name(use_case_name),
            compute_name=cls.normalize_cluster_name(use_case_name),
        )
