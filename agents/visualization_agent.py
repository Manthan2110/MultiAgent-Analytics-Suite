import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from . import AgentContext


class VisualizationAgent:
    def __init__(self, plots_dir="plots"):
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

    def save_plot(self, fig, filename):
        path = os.path.join(self.plots_dir, filename)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def run(self, context: AgentContext) -> AgentContext:
        df = context.df
        if df is None:
            raise ValueError("DataFrame not loaded.")

        print("[VisualizationAgent] Generating structured visualizations...")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include="object").columns.tolist()
        date_cols = df.select_dtypes(include=["datetime64", "datetime"]).columns.tolist()

        # Fully structured dict for plots
        visual_structure = {
            "distribution": [],
            "outliers": [],
            "violin": [],
            "category_frequency": [],
            "category_numeric_mean": [],
            "missing_values": [],
            "correlation": [],
            "pairplot": [],
            "time_series": [],
        }

        # =======================================================
        # 1. DISTRIBUTION PLOTS
        # =======================================================
        for col in numeric_cols:
            fig = plt.figure(figsize=(6, 4))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Distribution of {col}")
            path = self.save_plot(fig, f"dist_{col}.png")
            visual_structure["distribution"].append({"column": col, "path": path})

        # =======================================================
        # 2. OUTLIER DETECTION (Boxplots)
        # =======================================================
        for col in numeric_cols:
            fig = plt.figure(figsize=(6, 4))
            sns.boxplot(x=df[col])
            plt.title(f"Outlier Detection: {col}")
            path = self.save_plot(fig, f"box_{col}.png")
            visual_structure["outliers"].append({"column": col, "path": path})

        # =======================================================
        # 3. VIOLIN PLOTS (Spread + Shape)
        # =======================================================
        for col in numeric_cols:
            fig = plt.figure(figsize=(6, 4))
            sns.violinplot(x=df[col])
            plt.title(f"Violin Plot: {col}")
            path = self.save_plot(fig, f"violin_{col}.png")
            visual_structure["violin"].append({"column": col, "path": path})

        # =======================================================
        # 4. CATEGORY FREQUENCY (Top 10)
        # =======================================================
        for col in categorical_cols:
            fig = plt.figure(figsize=(8, 5))
            df[col].value_counts().head(10).plot(kind='bar')
            plt.title(f"Top 10 Categories: {col}")
            path = self.save_plot(fig, f"cat_top10_{col}.png")
            visual_structure["category_frequency"].append({"column": col, "path": path})

        # =======================================================
        # 5. CATEGORY VS NUMERIC MEAN
        # =======================================================
        if numeric_cols and categorical_cols:
            col_cat = categorical_cols[0]
            col_num = numeric_cols[0]

            fig = plt.figure(figsize=(10, 5))
            df.groupby(col_cat)[col_num].mean().sort_values(ascending=False).head(10).plot(kind='bar')
            plt.title(f"Avg {col_num} per Category of {col_cat}")
            path = self.save_plot(fig, "cat_vs_num_mean.png")
            visual_structure["category_numeric_mean"].append(
                {"category": col_cat, "numeric": col_num, "path": path}
            )

        # =======================================================
        # 7. CORRELATION HEATMAP
        # =======================================================
        if len(numeric_cols) > 1:
            fig = plt.figure(figsize=(10, 6))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
            plt.title("Correlation Heatmap")
            path = self.save_plot(fig, "correlation_heatmap.png")
            visual_structure["correlation"].append({"path": path})

        # =======================================================
        # 8. PAIRPLOT
        # =======================================================
        if 2 <= len(numeric_cols) <= 5:
            sns_plot = sns.pairplot(df[numeric_cols].dropna())
            path = os.path.join(self.plots_dir, "pairplot.png")
            sns_plot.savefig(path)
            plt.close()
            visual_structure["pairplot"].append({"columns": numeric_cols, "path": path})

        # =======================================================
        # 9. TIME SERIES
        # =======================================================
        if date_cols:
            date_col = date_cols[0]
            sorted_df = df.sort_values(by=date_col)
            num_col = numeric_cols[0]

            fig = plt.figure(figsize=(10, 5))
            plt.plot(sorted_df[date_col], sorted_df[num_col])
            plt.title(f"Trend of {num_col} over {date_col}")
            path = self.save_plot(fig, f"time_series_{num_col}.png")
            visual_structure["time_series"].append(
                {"date_column": date_col, "numeric": num_col, "path": path}
            )

        # SAVE IN CONTEXT
        context.visual_structure = visual_structure
        context.plots = [item["path"] for section in visual_structure.values() for item in section]
        context.summary["plot_list"] = context.plots
        context.summary["structured_plots"] = visual_structure

        print("[VisualizationAgent] Structured visualization complete.")
        return context