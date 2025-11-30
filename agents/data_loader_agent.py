import pandas as pd
from . import AgentContext


class DataLoaderAgent:
    def run(self, context: AgentContext) -> AgentContext:
        print("[DataLoaderAgent] Loading dataset...")
        df = pd.read_csv(context.dataset_path)

        context.df = df
        context.summary["num_rows"] = df.shape[0]
        context.summary["num_columns"] = df.shape[1]
        context.summary["columns"] = list(df.columns)
        context.summary["dtypes"] = df.dtypes.astype(str).to_dict()
        context.summary["missing_values"] = df.isna().sum().to_dict()

        print("[DataLoaderAgent] Dataset loaded with shape:", df.shape)
        return context