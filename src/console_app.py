"""
******************************
Author:
U3280553 U3243132 Assignment 3 Group 5 17/05/2026
Programming: Stage 3 deployment. Menu-driven console application providing
an interactive interface for dataset summary, EDA generation, model training,
and image prediction.
******************************
"""

"""
console_app.py
Stage 3 deployment: menu-driven console application.

This is the main interactive interface for the project. It lets the user
navigate between all stages of the system from a simple numbered menu.
Run with: python -m src.console_app
"""

from src.services.workflow_service import WorkflowService


class ConsoleApp:
    """Menu-driven console application for the Macroinvertebrate Analysis System.

    Provides a simple numbered menu that loops until the user chooses to exit.
    Each option calls the appropriate method on WorkflowService.
    """

    def __init__(self, workflow_service: WorkflowService) -> None:
        """Initialise with a shared WorkflowService instance.

        Args:
            workflow_service: The shared workflow coordinator.
        """
        self.workflow_service = workflow_service

    def _print_menu(self) -> None:
        """Print the main menu options to the console."""
        print("\n" + "=" * 50)
        print("  Macroinvertebrate Image Analysis System")
        print("=" * 50)
        print("  1. Show dataset summary")
        print("  2. Generate EDA outputs (charts)")
        print("  3. Train classifier (Stage 2)")
        print("  4. Predict an image class")
        print("  5. Exit")
        print("=" * 50)

    def run(self) -> None:
        """Start the menu loop and keep running until the user exits."""
        print("\nWelcome to the Macroinvertebrate Image Analysis System.")

        while True:
            self._print_menu()
            choice = input("  Select an option (1-5): ").strip()

            if choice == "1":
                try:
                    self.workflow_service.show_summary()
                except (FileNotFoundError, ValueError) as error:
                    print(f"\n  Error: {error}")

            elif choice == "2":
                try:
                    self.workflow_service.generate_eda()
                    print(f"\n  Charts saved to: outputs/eda/")
                    print("  Open the folder to view the PNG files.")
                except (FileNotFoundError, ValueError) as error:
                    print(f"\n  Error: {error}")

            elif choice == "3":
                try:
                    results = self.workflow_service.train_model()
                    print(f"\n  Accuracy: {results['accuracy']:.4f}")
                    print("\n  Classification Report:")
                    print(results["report"])
                except (FileNotFoundError, ValueError) as error:
                    print(f"\n  Error: {error}")

            elif choice == "4":
                image_path = input("\n  Enter the path to an image file: ").strip()
                if not image_path:
                    print("  No path entered. Returning to menu.")
                    continue
                try:
                    self.workflow_service.predict_image(image_path)
                except FileNotFoundError as error:
                    print(f"\n  Error: {error}")
                except ValueError as error:
                    print(f"\n  Error: {error}")

            elif choice == "5":
                print("\n  Exiting. Goodbye!")
                break

            else:
                print("  Invalid option. Please enter a number between 1 and 5.")


def main() -> None:
    """Start the console application."""
    workflow = WorkflowService()
    app = ConsoleApp(workflow)
    app.run()


if __name__ == "__main__":
    main()
