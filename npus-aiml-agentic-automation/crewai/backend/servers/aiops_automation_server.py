import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from servers.create_repo_utility import create_repo_for_agent
from servers.create_catalog_schema_utility import create_schema_for_agent
from servers.create_compute_utility import create_cluster_for_agent
from servers.config import (
    OnboardingInput,
    EnvironmentConfig,
    ResourceNameGenerator,
    TemplateType,
    TEAM_MAPPINGS_CONFIG,
    TeamInfo,
    OnboardingResult,
)
from custom_llm.nestle_llm import NestleLLM
from servers.knowledge_base_utility import MLOpsKnowledgeBase, EmbeddingConfig
from pathlib import Path

load_dotenv()

mcp = FastMCP("AIOps-Automation-Server", port=8060)


def _format_onboarding_output_with_llm(result: OnboardingResult, llm: NestleLLM) -> str:
    """Use LLM to format the onboarding result into a user-friendly output"""

    # Prepare the data for the LLM
    result_data = result.model_dump()

    # Create a focused prompt for formatting
    formatting_prompt = f"""You are a technical documentation formatter for an MLOps platform onboarding system.

Your task is to take the structured onboarding result data and format it into a clear, professional, user-friendly output.

# Input Data:
{json.dumps(result_data, indent=2)}

# Formatting Guidelines:

## For Successful Onboarding (success=True):
Structure your output as follows:

```
✅ MLOps Onboarding Completed Successfully

Use Case: [use_case_name]
Template: [template_used]
Model Registration: [Required/Not required based on requires_model_registration]

📦 Created Resources:
• GitHub Repository: [repo name from github_repo]
  URL: [repo_url from github_repo]

• Databricks Schema: [schema_name from resource_names]
  Location: Unity Catalog

• Compute Cluster: [compute_name from resource_names OR explanation if skipped]
  Status: [status from databricks_compute OR reason why skipped]

👥 Team Access Configured:
[For each team in team_mappings, show:]
• [Team Type]: [team name]
  - GitHub: [github_group]
  - Databricks: [entra_id_group]

[If there are warnings:]
⚠️ Warnings:
• [list each warning]

🎯 Next Steps:
1. Clone the repository and configure your local environment
2. Access Databricks workspace and verify schema permissions
3. [If compute created: "Wait for compute cluster to finish provisioning" 
    If compute skipped: "Work with MLOps engineers to create single-user compute clusters"]
4. Start developing your ML workflows!

📚 Resources:
- Need help? Contact: aiml-mlops-engineers@nestle.com
```

## For Failed Onboarding (success=False):
Structure your output as follows:

```
❌ MLOps Onboarding Failed

Use Case: [use_case_name]
Error: [error message]

[If any resources were created successfully:]
✅ Successfully Created:
• [List successfully created resources with details]

❌ Failed Resources:
• [List failed resources with error details]

🔧 Required Action:
Please contact the MLOps engineering team for assistance:
📧 Email: aiml-mlops-engineers@nestle.com

Include this information in your request:
- Use Case Name: [use_case_name]
- Error Details: [error]
- What Succeeded: [list successful resources]
- What Failed: [list failed resources]

The MLOps engineers will investigate and complete the remaining onboarding steps.
```

## Important Rules:
1. Use the exact data from the input - don't invent information
2. If a field is missing or None, use "N/A" or omit that section
3. Keep the tone professional but friendly
4. Use emojis as shown for visual clarity
5. Be concise - don't add unnecessary explanations
6. If databricks_compute has "skipped": true, explain why (model registration requirement)
7. Show all warnings if present
8. Format URLs as clickable links where applicable

Generate the formatted output now:"""

    try:
        # Make LLM call for formatting
        formatted_output = llm.call(
            messages=formatting_prompt,
            temperature=0.3,  # Low temperature for consistent formatting
            max_tokens=1500,
        )

        return formatted_output

    except Exception as e:
        # Fallback to basic formatting if LLM fails
        print(f"LLM formatting failed: {e}, falling back to basic format")
        return _basic_fallback_format(result)


def _basic_fallback_format(result: OnboardingResult) -> str:
    """Simple fallback formatting if LLM is unavailable"""
    if result.success:
        status = "✅ Onboarding Completed Successfully"
        resources = f"""
Created Resources:
- Repository: {result.resource_names.repo_name}
- Schema: {result.resource_names.schema_name}
- Compute: {result.resource_names.compute_name if not result.databricks_compute.get('skipped') else 'Not created (model registration required)'}
"""
        if result.warnings:
            resources += f"\nWarnings:\n" + "\n".join(f"- {w}" for w in result.warnings)

        return f"""{status}

Use Case: {result.use_case_name}
{resources}

For detailed information, see the full result data or contact: aiml-mlops-engineers@nestle.com"""
    else:
        return f"""❌ Onboarding Failed

Use Case: {result.use_case_name}
Error: {result.error}

Please contact the MLOps engineering team for assistance:
📧 Email: aiml-mlops-engineers@nestle.com

Include this information:
- Use Case Name: {result.use_case_name}
- Error: {result.error}
"""


@mcp.tool()
def onboard_mlops_use_case(
    use_case_name: str,
    internal_team: str,
    external_team: str = "none",
    additional_team: str = "none",
    requires_model_registration: bool = False,
) -> Dict[str, Any]:
    """
    Complete MLOps onboarding workflow that creates all necessary resources.

    This tool follows the standard MLOps onboarding flow:
    1. Validates inputs and team mappings
    2. Generates standardized resource names
    3. Creates GitHub repository with appropriate team permissions
    4. Creates Databricks schema with appropriate Entra ID group permissions
    5. Creates standard compute cluster (only if model registration NOT required)
    6. Formats the result into a user-friendly output using LLM

    Args:
        use_case_name: Name of the use case (max 40 characters)
                      Example: "agent-test-run"

        internal_team: Internal AI/ML team name
                      Options: "Demand Planning", "EAI", "Kroger Team", "SIG", "SRM", "BORA"
                      Case-insensitive, will be normalized

        external_team: External AI/ML team name (optional)
                      Options: "none", "Deloitte", "Genpact", "Tiger Analytics", "Tredence"
                      Default: "none"

        additional_team: Additional internal practitioner team (optional)
                        Options: "none", "Demand Planning", "EAI", "Kroger Team", "SIG", "SRM"
                        Default: "none"

        requires_model_registration: Whether use case requires model registration in Unity Catalog
                                    If True: Uses MLOps stacks template, does NOT create compute
                                    If False: Uses skinny dabs template, creates standard compute
                                    Default: False

    Returns:
        Dict with comprehensive results including:
        - success: Boolean indicating overall success
        - formatted_output: User-friendly formatted report (generated by LLM)
        - All other details: resource names, creation results, team mappings, warnings, errors
    """

    try:
        # Validate and normalize input
        input_data = OnboardingInput(
            use_case_name=use_case_name,
            internal_team=internal_team,
            external_team=external_team,
            additional_team=additional_team,
            requires_model_registration=requires_model_registration,
        )
    except ValueError as e:
        error_result = {
            "success": False,
            "error": str(e),
            "use_case_name": use_case_name,
            "formatted_output": f"❌ Input Validation Failed\n\nError: {str(e)}\n\nPlease check your input and try again.",
        }
        return error_result

    try:
        # Load and validate environment configuration
        env_config = EnvironmentConfig.from_env()
    except ValueError as e:
        error_result = {
            "success": False,
            "error": f"Configuration error: {str(e)}",
            "use_case_name": use_case_name,
            "formatted_output": f"❌ Configuration Error\n\nError: {str(e)}\n\nPlease contact: aiml-mlops-engineers@nestle.com",
        }
        return error_result

    # Generate resource names
    resource_names = ResourceNameGenerator.generate_all(use_case_name)

    # Determine template
    template = (
        TemplateType.MLOPS_STACKS
        if requires_model_registration
        else TemplateType.SKINNY_DAB
    )

    # Build team mappings for result
    team_mappings = {}

    # Internal team
    internal_mapping = TEAM_MAPPINGS_CONFIG.get_mapping(
        input_data.internal_team_normalized
    )
    team_mappings["internal_team"] = TeamInfo(
        name=internal_team,
        normalized=input_data.internal_team_normalized,
        github_group=internal_mapping.github_group,
        entra_id_group=internal_mapping.entra_id_group,
    )

    # External team
    external_entra_group = None
    if input_data.external_team_normalized != "none":
        external_mapping = TEAM_MAPPINGS_CONFIG.get_mapping(
            input_data.external_team_normalized
        )
        external_entra_group = external_mapping.entra_id_group
        team_mappings["external_team"] = TeamInfo(
            name=external_team,
            normalized=input_data.external_team_normalized,
            github_group=external_mapping.github_group,
            entra_id_group=external_mapping.entra_id_group,
        )

    # Additional team
    if input_data.additional_team_normalized != "none":
        additional_mapping = TEAM_MAPPINGS_CONFIG.get_mapping(
            input_data.additional_team_normalized
        )
        team_mappings["additional_team"] = TeamInfo(
            name=additional_team,
            normalized=input_data.additional_team_normalized,
            github_group=additional_mapping.github_group,
            entra_id_group=additional_mapping.entra_id_group,
        )

    # Initialize result
    result = OnboardingResult(
        success=True,
        use_case_name=use_case_name,
        resource_names=resource_names,
        template_used=template,
        requires_model_registration=requires_model_registration,
        team_mappings={k: v.model_dump() for k, v in team_mappings.items()},
        formatted_output="",
    )

    try:
        # Step 1: Create GitHub Repository
        print(f"Creating GitHub repository: {resource_names.repo_name}")
        repo_result = create_repo_for_agent(
            github_token=env_config.github_token,
            org=env_config.github_org,
            automation_repo=env_config.automation_repo,
            use_case_name=resource_names.repo_name,
            internal_team=input_data.internal_team_normalized,
            external_team=input_data.external_team_normalized,
            additional_team=input_data.additional_team_normalized,
            template=template.value,
            wait_for_completion=False,
        )

        result.github_repo = repo_result

        if not repo_result.get("success"):
            result.success = False
            result.error = f"Failed to create GitHub repository: {repo_result.get('error', 'Unknown error')}"
            # Format output with LLM even for failures
            result.formatted_output = _format_with_llm_or_fallback(result)
            return result.model_dump()

        # Step 2: Create Databricks Schema
        print(f"Creating Databricks schema: {resource_names.schema_name}")
        schema_result = create_schema_for_agent(
            workspace_url=env_config.workspace_url,
            schema_name=resource_names.schema_name,
            internal_entra_id_group=internal_mapping.entra_id_group,
            external_entra_id_group=external_entra_group,
            tenant_id=env_config.tenant_id,
            client_id=env_config.client_id,
            client_secret=env_config.client_secret,
        )

        result.databricks_schema = schema_result

        if not schema_result.get("success"):
            result.success = False
            result.error = (
                f"GitHub repo created, but failed to create Databricks schema: "
                f"{schema_result.get('error', 'Unknown error')}"
            )
            result.warnings.append(
                "GitHub repository was created successfully, but schema creation failed"
            )
            # Format output with LLM
            result.formatted_output = _format_with_llm_or_fallback(result)
            return result.model_dump()

        # Step 3: Create Databricks Compute (conditionally)
        if requires_model_registration:
            result.databricks_compute = {
                "skipped": True,
                "reason": "Model registration required - single-user clusters needed",
                "message": (
                    "Compute cluster not created. Use case requires model registration "
                    "in Unity Catalog. Please work with MLOps engineers to create "
                    "single-user clusters with appropriate permissions."
                ),
            }
            result.warnings.append(
                "Compute cluster not created due to model registration requirement"
            )
        else:
            print(f"Creating Databricks compute: {resource_names.compute_name}")
            compute_result = create_cluster_for_agent(
                workspace_url=env_config.workspace_url,
                cluster_name=resource_names.compute_name,
                tenant_id=env_config.tenant_id,
                client_id=env_config.client_id,
                client_secret=env_config.client_secret,
                wait_for_completion=False,
            )

            result.databricks_compute = compute_result

            if not compute_result.get("success"):
                result.warnings.append(
                    f"Repo and schema created successfully, but compute creation failed: "
                    f"{compute_result.get('error', 'Unknown error')}"
                )

        # Generate summary message
        if result.success:
            if requires_model_registration:
                result.message = (
                    f"MLOps onboarding completed for '{use_case_name}'.\n\n"
                    f"Created resources:\n"
                    f"✓ GitHub Repository: {resource_names.repo_name}\n"
                    f"✓ Databricks Schema: {resource_names.schema_name}\n"
                    f"⚠ Compute Cluster: NOT CREATED (requires single-user clusters for model registration)\n\n"
                    f"Next steps: Work with MLOps engineers to create single-user compute clusters."
                )
            else:
                result.message = (
                    f"MLOps onboarding completed successfully for '{use_case_name}'.\n\n"
                    f"Created resources:\n"
                    f"✓ GitHub Repository: {resource_names.repo_name}\n"
                    f"✓ Databricks Schema: {resource_names.schema_name}\n"
                    f"✓ Standard Compute: {resource_names.compute_name}\n\n"
                    f"All resources are ready for use in the MLOps Platform."
                )

        # Format output with LLM
        result.formatted_output = _format_with_llm_or_fallback(result)

        return result.model_dump()

    except Exception as e:
        result.success = False
        result.error = f"Unexpected error during onboarding: {str(e)}"
        result.formatted_output = _format_with_llm_or_fallback(result)
        return result.model_dump()


def _format_with_llm_or_fallback(result: OnboardingResult) -> str:
    """Helper function to format output with LLM or use fallback"""
    try:
        llm = NestleLLM(
            model=os.getenv("NESTLE_MODEL", "gpt-4.1"),
            client_id=os.getenv("NESTLE_CLIENT_ID"),
            client_secret=os.getenv("NESTLE_CLIENT_SECRET"),
        )

        return _format_onboarding_output_with_llm(result, llm)

    except Exception as e:
        print(f"Failed to format with LLM: {e}, using fallback")
        return _basic_fallback_format(result)


@mcp.tool()
def query_mlops_knowledge_base(
    query: str,
    num_sources: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    search_only: bool = False,
) -> Dict[str, Any]:
    """
    Query the MLOps knowledge base to get answers from documentation.

    This tool searches through the MLOps documentation and either returns
    relevant document chunks (search_only=True) or generates a comprehensive
    answer using the LLM with retrieved context.

    Args:
        query: The question or search query about MLOps documentation
        num_sources: Number of relevant document chunks to retrieve (default: 5, max: 20)
        temperature: LLM temperature for answer generation (0.0-1.0, default: 0.7)
        max_tokens: Maximum tokens in the generated answer (default: 1000)
        search_only: If True, return only retrieved chunks without LLM generation (default: False)

    Returns:
        Dict with answer/results, sources, and metadata

    Examples:
        - "What are the different workflow files in MLOps template?"
        - "How do I configure CI/CD pipelines?"
        - "Explain the deployment process for ML models"
    """
    try:
        # Validate inputs
        if not query or not query.strip():
            return {
                "success": False,
                "error": "Query cannot be empty",
            }

        # Limit num_sources to reasonable range
        num_sources = max(1, min(num_sources, 20))
        temperature = max(0.0, min(temperature, 1.0))
        max_tokens = max(100, min(max_tokens, 4000))

        # Initialize configuration
        config = EmbeddingConfig(
            chunk_size=800,
            chunk_overlap=150,
            persist_directory=os.getenv("VECTOR_DB_LOC"),
        )

        # Check if vector database exists
        db_path = Path(config.persist_directory)
        if not db_path.exists() or not any(db_path.iterdir()):
            return {
                "success": False,
                "error": "Knowledge base not initialized. Please run embedding creation first.",
                "query": query,
            }

        # Initialize LLM if needed for answer generation
        llm = None
        if not search_only:
            try:
                client_id = os.getenv("NESTLE_CLIENT_ID")
                client_secret = os.getenv("NESTLE_CLIENT_SECRET")
                model = os.getenv("NESTLE_MODEL", "gpt-4.1")

                if not all([client_id, client_secret]):
                    return {
                        "success": False,
                        "error": "LLM credentials not configured (NESTLE_CLIENT_ID, NESTLE_CLIENT_SECRET)",
                        "query": query,
                    }

                llm = NestleLLM(
                    model=model,
                    client_id=client_id,
                    client_secret=client_secret,
                )
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to initialize LLM: {str(e)}",
                    "query": query,
                }

        # Initialize knowledge base
        kb = MLOpsKnowledgeBase(config, llm=llm)

        # Perform search or generate answer
        if search_only:
            # Pure retrieval without LLM
            results = kb.search_knowledge_base(
                query=query, k=num_sources, include_scores=True
            )

            return {
                "success": True,
                "query": query,
                "mode": "search_only",
                "results": results,
                "num_results": len(results),
            }
        else:
            # RAG: Retrieve + Generate
            result = kb.generate_answer(
                query=query,
                k=num_sources,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Check if generation failed
            if "error" in result:
                return {
                    "success": False,
                    "error": result["error"],
                    "query": query,
                }

            return {
                "success": True,
                "query": result["query"],
                "answer": result["answer"],
                "sources": result["sources"],
                "num_sources": result["num_sources"],
                "mode": "rag_generation",
            }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to query knowledge base: {str(e)}",
            "query": query if "query" in locals() else "N/A",
        }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
