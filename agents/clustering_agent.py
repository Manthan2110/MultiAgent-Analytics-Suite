import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from . import AgentContext

class ClusteringAgent:
    def __init__(self, plots_dir="plots"):
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

    def run(self, context: AgentContext, n_clusters=None) -> AgentContext:
        df = context.df.select_dtypes(include="number").dropna()

        if df.shape[1] < 2:
            raise ValueError("Need at least 2 numeric columns for clustering")

        # Auto-detect K if not provided
        if n_clusters is None:
            inertia_list = []
            for k in range(2, 8):
                km = KMeans(n_clusters=k, random_state=42).fit(df)
                inertia_list.append((k, km.inertia_))
            # choose elbow (minimum slope change)
            n_clusters = sorted(inertia_list, key=lambda x: x[1])[0][0]

        scaler = StandardScaler()
        scaled = scaler.fit_transform(df)

        km = KMeans(n_clusters=n_clusters, random_state=42)
        labels = km.fit_predict(scaled)

        df_clustered = df.copy()
        df_clustered["Cluster"] = labels

        # Store statistics
        cluster_stats = df_clustered.groupby("Cluster").mean().round(3)

        # PCA for 2D plotting
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(scaled)
        df_clustered["PC1"] = pcs[:, 0]
        df_clustered["PC2"] = pcs[:, 1]

        # Plot cluster scatter
        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=df_clustered, x="PC1", y="PC2", hue="Cluster", palette="tab10")
        plt.title("Clustering (PCA 2D Visualization)")
        plot_path = os.path.join(self.plots_dir, "cluster_scatter.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()

        context.clustering = {
            "n_clusters": n_clusters,
            "cluster_stats": cluster_stats,
            "plot_path": plot_path
        }

        return context