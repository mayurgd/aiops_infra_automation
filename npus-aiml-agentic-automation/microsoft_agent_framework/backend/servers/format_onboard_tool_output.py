import os
import json
from data_models.server_models import (
    OnboardingResult,
)
from prompts import onboard_mlops_use_case_output_prompt
from custom_llm.nestle_llm import NestleLLM


def _format_onboarding_output_with_llm(result: OnboardingResult, llm: NestleLLM) -> str:
    """Use LLM to format the onboarding result into a user-friendly output"""

    # Prepare the data for the LLM
    result_data = result.model_dump()

    # Create a focused prompt for formatting
    formatting_prompt = onboard_mlops_use_case_output_prompt.format(
        result_data=json.dumps(result_data, indent=2)
    )
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
