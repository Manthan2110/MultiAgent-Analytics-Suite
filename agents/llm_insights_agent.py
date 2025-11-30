import json
from . import AgentContext
from .llm_client import generate_llm_response


class LLMInsightsAgent:
    def run(self, context: AgentContext) -> AgentContext:
        print("[LLMInsightsAgent] Generating graph-aware LLM insights...")

        df = context.df
        summary = context.summary
        plot_structure = context.visual_structure  # NEW

        if df is None:
            raise ValueError("DataFrame not loaded in context")

        # Prepare Compact Summary
        compact_summary = {
            "num_rows": summary.get("num_rows"),
            "num_columns": summary.get("num_columns"),
            "columns": summary.get("columns"),
            "dtypes": summary.get("dtypes"),
            "missing_values": summary.get("missing_values"),
            "numeric_summary": summary.get("numeric_summary"),
            "correlation": summary.get("correlation"),
        }

        sample_rows = df.head(5).to_dict(orient="records")

        # SYSTEM PROMPT — Graph-Aware Analysis
        system_prompt = """
You are a senior data analyst with advanced knowledge of:
- Visual pattern recognition
- Statistical analysis
- Categorical reasoning
- Trend interpretation
- Outlier detection
- Correlation interpretation

Your job: Interpret BOTH summary statistics AND plotted visualizations provided.

For EACH visualization group (distribution, outliers, violin, category frequency, missing value heatmap, correlation heatmap, pairplot, time-series), explain:

1. What patterns are visible?
2. Does the feature have skewness?
3. Any outliers?
4. Are categories imbalanced?
5. Any suspicious spikes or drops?
6. Any strong or weak correlations?
7. Time trends or seasonal effects?
8. Business meaning of patterns
9. Recommendations (actions analyst should take)

Write insights in clear markdown sections using:

## 📌 Overview
## 📊 Distribution Insights
## 🚨 Outlier Insights
## 🎻 Distribution Shape (Violin Plots)
## 🔠 Category Insights
## 🧩 Missing Value Insights
## 🔗 Correlation Insights
## 📈 Time-Series Insights (if any)
## 🔁 Pairwise Relationship Insights
## 🤖 Final AI Summary (MOST IMPORTANT)
## 🎯 Recommendations

BE SPECIFIC about what you observe.
Use facts from the dataset summary AND visual behavior.
"""

        # USER PROMPT — Includes Visual Structure
        user_prompt = f"""
DATASET SUMMARY (JSON):
{json.dumps(compact_summary, indent=2)}

SAMPLE ROWS (JSON):
{json.dumps(sample_rows, indent=2)}

STRUCTURED VISUALIZATION METADATA (JSON):
{json.dumps(plot_structure, indent=2)}

Generate graph-aware insights.
"""

        insights_text = generate_llm_response(system_prompt, user_prompt)

        context.insights.append("### 🤖 Graph-Aware LLM Insights\n" + insights_text)

        print("[LLMInsightsAgent] Graph-aware insights generated.")
        return context