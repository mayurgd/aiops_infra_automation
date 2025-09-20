@mcp.tool()
def validate_github_repository_inputs(
    use_case_name: str,
    template: str = "npus-aiml-mlops-stacks-template",
    internal_team: str = "eai",
    development_team: str = "none",
    additional_team: str = "none",
) -> Dict[str, Any]:
    """
    Validate inputs for GitHub repository creation without actually creating the repository.

    Args:
        use_case_name: Name of the use case for the repository
        template: Template to use (npus-aiml-mlops-stacks-template or npus-aiml-skinny-dab-template)
        internal_team: Internal team (eai, deloitte, sig, tredence, bora, genpact, tiger, srm, kroger)
        development_team: Development team (same options as internal_team or 'none')
        additional_team: Additional team (same options as internal_team or 'none')

    Returns:
        Dict with validation results
    """
    try:
        from create_repo_utility import validate_repository_inputs

        result = validate_repository_inputs(
            use_case_name=use_case_name,
            template=template,
            internal_team=internal_team,
            development_team=development_team,
            additional_team=additional_team,
        )

        return result

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to validate repository inputs: {str(e)}",
        }


@mcp.tool()
def validate_databricks_schema_inputs(
    catalog: str,
    schema: str,
    aiml_support_team: str = "AIOps",
    aiml_use_case: str = "agent_test_run",
    business_owner: str = "AIOps",
    internal_entra_id_group: str = "AAD-SG-NPUS-aiml-internal-contributors",
    external_entra_id_group: str = None,
) -> Dict[str, Any]:
    """
    Validate inputs for Databricks schema creation without actually creating the schema.

    Args:
        catalog: Databricks catalog name (npus_aiml_workbench or npus_aiml_stage)
        schema: Schema name (e.g., category_forecast, move37, pricing_ppa_tool, etc.)
        aiml_support_team: AIML support team (Digital Manufacturing, EAI, SIG, SRM, AIOps)
        aiml_use_case: AIML use case (retailer_pos_kroger_forecast, Category Forecast, etc.)
        business_owner: Business owner (kroger, SRM, Digital Manufacturing, Enterprise-wide, MDO, ORM, Transportation, AIOps)
        internal_entra_id_group: Internal Entra ID group (default: AAD-SG-NPUS-aiml-internal-contributors)
        external_entra_id_group: External Entra ID group (optional)

    Returns:
        Dict with validation results
    """
    try:
        from create_catalog_schema_utility import validate_schema_inputs

        result = validate_schema_inputs(
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
            "error": f"Failed to validate schema inputs: {str(e)}",
        }
