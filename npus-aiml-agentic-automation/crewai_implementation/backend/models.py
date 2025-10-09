from pydantic import BaseModel
from typing import Any, Dict, Optional


class CrewRequest(BaseModel):
    user_query: str
    inputs: Optional[Dict[str, Any]] = {}


class InputResponse(BaseModel):
    input: str


class CreateRepoRequest(BaseModel):
    use_case_name: str
    template: str = "npus-aiml-mlops-stacks-template"
    internal_team: str = "eai"
    development_team: str = "none"
    additional_team: str = "none"


class CreateSchemaRequest(BaseModel):
    catalog: str
    schema: str
    aiml_support_team: str
    aiml_use_case: str
    business_owner: str
    internal_entra_id_group: str = "AAD-SG-NPUS-aiml-internal-contributors"
    external_entra_id_group: str = "none"


class CreateComputeRequest(BaseModel):
    cluster_name: str
    spark_version: str = "16.4.x-scala2.12"
    driver_node_type_id: str = "Standard_D4a_v4"
    node_type_id: str = "Standard_D4a_v4"
    min_workers: int = 1
    max_workers: int = 4
    data_security_mode: str = "SINGLE_USER"
    aiml_use_case: Optional[str] = None
