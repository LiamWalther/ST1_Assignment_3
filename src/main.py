"""
******************************
Author:
U3280553 U3243132 Assignment 3 Group 5 17/05/2026
Programming: Non-interactive entry point. Runs the full Stage 1 and Stage 2
pipeline in one command without the console menu.
******************************
"""

"""
main.py
Non-interactive entry point that runs the full Stage 1 + Stage 2 pipeline.
Useful for testing that everything works end-to-end without the menu.

Run with: python -m src.main
"""

from src.services.workflow_service import WorkflowService


def main() -> None:
    """Run the complete pipeline: EDA then classifier training."""
    workflow = WorkflowService()
    workflow.run_full_pipeline()


if __name__ == "__main__":
    main()
