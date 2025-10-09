#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from crew import AiopsAgenticAutomation
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run_repo_only():
    """
    Run the crew for GitHub repository creation only.
    """
    inputs = {
        "user_query": "Create a new GitHub repository for my ML project",
        "use_case_name": "agent-test-run",
        "template": "npus-aiml-mlops-stacks-template",
        "internal_team": "eai",
        "development_team": "eai",
        "additional_team": "none",
    }

    print("=== Running GitHub Repository Creation Only ===")
    try:
        result = AiopsAgenticAutomation().kickoff(inputs=inputs)
        print("Repository creation completed successfully!")
        return result
    except Exception as e:
        raise Exception(f"An error occurred while creating repository: {e}")


def run_schema_only():
    """
    Run the crew for Databricks schema creation only.
    """
    inputs = {
        "user_query": "Set up Databricks infrastructure for my data project",
        "catalog": "npus_aiml_workbench",
        "schema": "agent_test_run",
        "aiml_support_team": "AIOps",
        "aiml_use_case": "agent_test_run",
        "business_owner": "AIOps",
        "internal_entra_id_group": "AAD-SG-NPUS-aiml-internal-contributors",
        "external_entra_id_group": "none",
    }

    print("=== Running Databricks Schema Creation Only ===")
    try:
        result = AiopsAgenticAutomation().kickoff(inputs=inputs)
        print("Databricks schema creation completed successfully!")
        return result
    except Exception as e:
        raise Exception(f"An error occurred while creating Databricks schema: {e}")


def run_compute_only():
    """
    Run the crew for Databricks compute creation only.
    """
    inputs = {
        "user_query": "Create a Databricks compute cluster for my ML workload",
        "cluster_name": "agent_test_run",
        "spark_version": "16.4.x-scala2.12",
        "driver_node_type_id": "Standard_D4a_v4",
        "node_type_id": "Standard_D4a_v4",
        "min_workers": 1,
        "max_workers": 4,
        "data_security_mode": "SINGLE_USER",
        "aiml_use_case": "agent_test_run",
    }

    print("=== Running Databricks Compute Creation Only ===")
    try:
        result = AiopsAgenticAutomation().kickoff(inputs=inputs)
        print("Databricks compute creation completed successfully!")
        return result
    except Exception as e:
        raise Exception(f"An error occurred while creating Databricks compute: {e}")


def run_all():
    """
    Run the crew for GitHub repository, Databricks schema, and compute creation.
    """
    inputs = {
        "user_query": "Set up complete infrastructure with GitHub repo, Databricks schema, and compute cluster for my ML project",
        # GitHub inputs
        "use_case_name": "agent-test-run",
        "template": "npus-aiml-mlops-stacks-template",
        "internal_team": "eai",
        "development_team": "eai",
        "additional_team": "none",
        # Databricks schema inputs
        "catalog": "npus_aiml_workbench",
        "schema": "agent_test_run",
        "aiml_support_team": "AIOps",
        "aiml_use_case": "agent_test_run",
        "business_owner": "AIOps",
        "internal_entra_id_group": "AAD-SG-NPUS-aiml-internal-contributors",
        "external_entra_id_group": "none",
        # Databricks compute inputs
        "cluster_name": "agent_test_run",
        "spark_version": "16.4.x-scala2.12",
        "driver_node_type_id": "Standard_D4a_v4",
        "node_type_id": "Standard_D4a_v4",
        "min_workers": 1,
        "max_workers": 4,
        "data_security_mode": "SINGLE_USER",
        "aiml_use_case": "agent_test_run",
    }

    print(
        "=== Running Complete Infrastructure Setup (GitHub + Databricks Schema + Compute) ==="
    )
    try:
        result = AiopsAgenticAutomation().kickoff(inputs=inputs)
        print("Complete infrastructure setup completed successfully!")
        return result
    except Exception as e:
        raise Exception(f"An error occurred while setting up infrastructure: {e}")


def run_supervision_example():
    """
    Run the crew with ambiguous inputs to demonstrate supervising functionality.
    """
    inputs = {
        "user_query": "I need to set up infrastructure for my new project",
    }

    print("=== Running Supervising Example (Ambiguous Inputs) ===")
    try:
        result = AiopsAgenticAutomation().kickoff(inputs=inputs)
        print("Supervising completed successfully!")
        return result
    except Exception as e:
        raise Exception(f"An error occurred during supervision: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        scenario = sys.argv[1].lower()
        if scenario == "repo":
            run_repo_only()
        elif scenario == "schema":
            run_schema_only()
        elif scenario == "compute":
            run_compute_only()
        elif scenario == "all":
            run_all()
        elif scenario == "supervision":
            run_supervision_example()
        else:
            print("Valid options: repo, schema, compute, all, supervision")
            print("Example: python main.py all")
    else:
        print("Valid options: repo, schema, compute, all, supervision")
        print("eg: python main.py repo")
