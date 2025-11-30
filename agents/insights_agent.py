from . import AgentContext

class InsightsAgent:
    def run(self, context: AgentContext) -> AgentContext:
        print("[InsightsAgent] Generating structured insights...")

        summary = context.summary
        insights = []

        # --- Section 1: Dataset Overview ---
        rows, cols = summary["shape"]
        insights.append(f"### 📁 Dataset Overview")
        insights.append(f"- The dataset contains **{rows} rows** and **{cols} columns**.")
        insights.append(f"- Memory usage: **{summary['memory_usage'] / 1024:.2f} KB**.")

        # --- Section 2: Missing Values ---
        missing = summary["missing_percentage"]
        high_missing = [col for col, pct in missing.items() if pct > 30]
        insights.append(f"\n### 🧩 Missing Values Insights")
        if high_missing:
            insights.append(f"- Columns with high missing values (>30%): **{', '.join(high_missing)}**.")
        else:
            insights.append("- No critical missing value issues identified.")

        # --- Section 3: Numeric Insights ---
        numeric_stats = summary["numeric_stats"]
        insights.append(f"\n### 🔢 Numeric Feature Insights")
        for col, stats in numeric_stats.items():
            insights.append(
                f"- **{col}**: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
                f"skew={stats['skewness']:.2f}, outliers={stats['outliers']}"
            )

        # --- Section 4: Categorical Insights ---
        cat_stats = summary["categorical_stats"]
        insights.append(f"\n### 🔠 Categorical Feature Insights")
        for col, stats in cat_stats.items():
            insights.append(
                f"- **{col}** has **{stats['unique_values']} unique values**. "
                f"Top categories: {list(stats['top_categories'].items())[:3]}"
            )

        # --- Section 5: Correlation Insights ---
        corr = summary.get("correlation", {})
        insights.append(f"\n### 🔗 Correlation Insights")
        high_corr_pairs = []
        for col, row in corr.items():
            for col2, val in row.items():
                if col != col2 and abs(val) > 0.7:
                    high_corr_pairs.append((col, col2, val))

        if high_corr_pairs:
            for c1, c2, v in high_corr_pairs:
                insights.append(f"- Strong correlation between **{c1}** and **{c2}** → {v:.2f}")
        else:
            insights.append("- No strong correlations identified.")

        context.insights = insights
        return context