from . import AgentContext
from .data_loader_agent import DataLoaderAgent
from .eda_agent import EDAAgent
from .visualization_agent import VisualizationAgent
from .insights_agent import InsightsAgent
from .llm_insights_agent import LLMInsightsAgent
from .report_agent import ReportAgent
from .ml_agent import MLAgent
from .ml_insights_agent import MLInsightsAgent

class PlannerAgent:
    def __init__(self, use_llm_insights: bool = True):
        self.data_loader = DataLoaderAgent()
        self.eda_agent = EDAAgent()
        self.viz_agent = VisualizationAgent()
        self.rule_insights_agent = InsightsAgent()
        self.llm_insights_agent = LLMInsightsAgent() if use_llm_insights else None
        self.report_agent = ReportAgent()

    def run_pipeline(self, context: AgentContext) -> AgentContext:
        print("[PlannerAgent] Starting analysis pipeline...")

        # STEP 1 — Load Data
        context = self.data_loader.run(context)

        # STEP 2 — Basic EDA (missing values, stats, correlation, etc.)
        context = self.eda_agent.run(context)

        # STEP 3 — Generate all visuals (10+ advanced plots)
        context = self.viz_agent.run(context)

        # STEP 4 — Add rule-based insights (numeric, categorical, outliers)
        context = self.rule_insights_agent.run(context)

        # STEP 5 — Add LLM-based advanced analytical insights
        if self.llm_insights_agent is not None:
            context = self.llm_insights_agent.run(context)

        # STEP 6 — Generate final report
        context = self.report_agent.run(context)

        print("[PlannerAgent] Pipeline completed successfully.")
        return context
    
    def run_feature_importance(self, context: AgentContext, target_column: str) -> AgentContext:
        ml_agent = MLAgent()
        ml_insight_agent = MLInsightsAgent()

        context = ml_agent.run(context, target_column)
        context = ml_insight_agent.run(context)

        return context