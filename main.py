import os
from agents import AgentContext
from agents.planner_agent import PlannerAgent


def main():
    # 1) Point to your CSV file
    dataset_path = os.path.join("data", "sample.csv")

    # 2) Create context
    context = AgentContext(dataset_path=dataset_path)

    # 3) Run planner
    planner = PlannerAgent()
    context = planner.run_pipeline(context)

    print("\n✅ Done!")
    print("Report saved at:", context.report_path)
    print("Plots saved at:", context.plots)


if __name__ == "__main__":
    main()