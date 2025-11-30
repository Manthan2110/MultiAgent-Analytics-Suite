import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from . import AgentContext


class MLAgent:
    def __init__(self, output_dir="plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def detect_task_type(self, series: pd.Series):
        """Detect if task is regression or classification."""
        if series.dtype in ["float64", "int64"]:
            # If too many unique numeric values → regression
            return "regression"
        else:
            return "classification"

    def run(self, context: AgentContext, target_column: str) -> AgentContext:
        print("[MLAgent] Running AutoML Feature Importance Analysis...")

        df = context.df
        if df is None:
            raise ValueError("DataFrame not loaded")

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        # Separate target + features
        y = df[target_column]
        X = df.drop(columns=[target_column]).copy()

        # Encode categorical features
        for col in X.select_dtypes(include="object").columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        # Drop rows with missing values
        X = X.fillna(0)
        y = y.fillna(0)

        task_type = self.detect_task_type(y)
        print(f"[MLAgent] Detected task type: {task_type}")

        # Choose model
        if task_type == "classification":
            model = RandomForestClassifier()
        else:
            model = RandomForestRegressor()

        # Fit model
        model.fit(X, y)

        # Compute feature importance
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            "feature": X.columns,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        # Save importance graph
        fig = plt.figure(figsize=(8, 6))
        sns.barplot(
            x=feature_importance_df["importance"],
            y=feature_importance_df["feature"]
        )
        plt.title("Feature Importance")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")

        plot_path = os.path.join(self.output_dir, "feature_importance.png")
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)

        # Store results in context
        context.feature_importance = {
            "importance_table": feature_importance_df.to_dict(orient="records"),
            "plot_path": plot_path,
            "task_type": task_type,
            "target_column": target_column,
        }

        print("[MLAgent] Feature importance generated.")
        return context