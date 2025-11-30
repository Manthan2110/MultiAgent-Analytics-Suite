import pandas as pd
import numpy as np
from . import AgentContext

class EDAAgent:
    def run(self, context: AgentContext) -> AgentContext:
        print("[EDAAgent] Running detailed EDA...")

        df = context.df
        
        # 1. Basic Information
        context.summary["shape"] = df.shape
        context.summary["memory_usage"] = df.memory_usage(deep=True).sum()

        # 2. Column Types Breakdown
        context.summary["numeric_columns"] = df.select_dtypes(include="number").columns.tolist()
        context.summary["categorical_columns"] = df.select_dtypes(include=["object", "category"]).columns.tolist()
        context.summary["datetime_columns"] = df.select_dtypes(include="datetime").columns.tolist()
        context.summary["boolean_columns"] = df.select_dtypes(include="bool").columns.tolist()

        # 3. Duplicate Rows
        context.summary["duplicate_rows"] = df.duplicated().sum()

        # 4. Missing Percentage
        context.summary["missing_percentage"] = (
            df.isna().mean().round(3) * 100
        ).to_dict()

        # 5. Numeric Stats
        numeric_df = df.select_dtypes(include="number")
        numeric_stats = {}

        for col in numeric_df.columns:
            numeric_stats[col] = {
                "mean": float(numeric_df[col].mean()),
                "median": float(numeric_df[col].median()),
                "std": float(numeric_df[col].std()),
                "min": float(numeric_df[col].min()),
                "max": float(numeric_df[col].max()),
                "skewness": float(numeric_df[col].skew()),
                "kurtosis": float(numeric_df[col].kurt()),
                "outliers": int(
                    ((numeric_df[col] < (numeric_df[col].quantile(0.25) - 1.5 * (numeric_df[col].quantile(0.75) - numeric_df[col].quantile(0.25)))) |
                     (numeric_df[col] > (numeric_df[col].quantile(0.75) + 1.5 * (numeric_df[col].quantile(0.75) - numeric_df[col].quantile(0.25)))))
                    .sum()
                ),
            }

        context.summary["numeric_stats"] = numeric_stats

        # 6. Categorical Stats
        categorical_df = df.select_dtypes(include=["object", "category"])
        cat_stats = {}

        for col in categorical_df.columns:
            cat_stats[col] = {
                "unique_values": categorical_df[col].nunique(),
                "top_categories": categorical_df[col].value_counts().head(5).to_dict(),
            }

        context.summary["categorical_stats"] = cat_stats

        # 7. Correlation Analysis
        if len(numeric_df.columns) >= 2:
            corr = numeric_df.corr().round(3)
            context.summary["correlation"] = corr.to_dict()

        # -----------------------------------------
        # 8. NUMERIC TABLE (NEW)
        # -----------------------------------------
        if len(numeric_df.columns) > 0:
            numeric_table = numeric_df.describe().transpose().reset_index()
            numeric_table = numeric_table.rename(columns={"index": "Feature"})
            context.summary["numeric_table"] = numeric_table

        # -----------------------------------------
        # 9. CATEGORICAL TABLE (NEW)
        # -----------------------------------------
        if len(categorical_df.columns) > 0:
            categorical_table = pd.DataFrame({
                "Feature": categorical_df.columns,
                "Unique Values": [categorical_df[col].nunique() for col in categorical_df.columns],
                "Top Category": [categorical_df[col].value_counts().idxmax() if categorical_df[col].nunique() > 0 else None for col in categorical_df.columns],
                "Top Count": [categorical_df[col].value_counts().max() if categorical_df[col].nunique() > 0 else None for col in categorical_df.columns],
            })
            context.summary["categorical_table"] = categorical_table

        # -----------------------------------------
        # 10. MISSING VALUES TABLE (NEW)
        # -----------------------------------------
        missing = df.isnull().sum()
        missing_table = pd.DataFrame({
            "Feature": missing.index,
            "Missing Count": missing.values,
            "Missing %": (missing.values / len(df) * 100).round(2)
        })
        context.summary["missing_table"] = missing_table

        print("[EDAAgent] Detailed EDA completed.")
        return context