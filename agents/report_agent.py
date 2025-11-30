import os
import datetime
from . import AgentContext


class ReportAgent:
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    def run(self, context: AgentContext) -> AgentContext:
        print("[ReportAgent] Creating markdown report...")

        summary = context.summary
        insights = context.insights
        plots = context.plots

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.reports_dir, f"report_{timestamp}.md")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# 📊 AI Multi-Agent Data Analyst Report\n\n")
            f.write(f"Generated on: `{timestamp}`\n\n")

            # Basic summary
            f.write("## 📁 Dataset Overview\n\n")
            f.write(f"- Rows: **{summary.get('num_rows')}**\n")
            f.write(f"- Columns: **{summary.get('num_columns')}**\n\n")

            f.write("### Columns & Types\n\n")
            for col, dtype in summary.get("dtypes", {}).items():
                f.write(f"- **{col}**: `{dtype}`\n")
            f.write("\n")


            # Insights
            f.write("## 💡 Key Insights\n\n")
            for ins in insights:
                f.write(f"- {ins}\n")
            f.write("\n")

        context.report_path = report_path
        print(f"[ReportAgent] Report generated at: {report_path}")
        return context