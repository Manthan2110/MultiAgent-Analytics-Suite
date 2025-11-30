import json
from . import AgentContext
from .llm_client import generate_llm_response


class MLInsightsAgent:
    def run(self, context: AgentContext) -> AgentContext:
        print("[MLInsightsAgent] Generating ML-based insights...")

        if not hasattr(context, "feature_importance"):
            print("[MLInsightsAgent] No feature importance found.")
            return context

        fi = context.feature_importance

        system_prompt = """
You are a senior machine learning engineer and data scientist.
You analyze feature importance results from RandomForest models.

Your goal:
- Explain which features influence the target
- Why they matter
- Provide actionable ML recommendations
- Suggest preprocessing steps
- Suggest next modeling steps

Write in clean markdown with sections:
## 🎯 Task Type
## 🔍 Key Influential Features
## 📊 What the Feature Importance Tells Us
## 🧠 Interpretation & Insights
## 🛠 Recommendations
"""

        user_prompt = f"""
TARGET COLUMN: {fi['target_column']}
TASK TYPE: {fi['task_type']}

FEATURE IMPORTANCE TABLE:
{json.dumps(fi['importance_table'], indent=2)}
"""

        insights_text = generate_llm_response(system_prompt, user_prompt)

        context.insights.append("### 🤖 ML Feature Importance Insights\n" + insights_text)

        print("[MLInsightsAgent] ML insights added.")
        return context