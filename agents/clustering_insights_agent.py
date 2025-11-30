from .llm_client import generate_llm_response
from . import AgentContext
import json

class ClusteringInsightsAgent:
    def run(self, context: AgentContext):
        clustering = context.clustering
        if clustering is None:
            return context

        system_prompt = """
You are an expert ML analyst. Analyze the clustering results:
- Cluster centroids meaning
- Differences between clusters
- Patterns in numeric features
- Why clusters form
- Real-world interpretations
- Recommendations
"""

        user_prompt = json.dumps({
            "n_clusters": clustering["n_clusters"],
            "cluster_stats": clustering["cluster_stats"].to_dict()
        }, indent=2)

        insights = generate_llm_response(system_prompt, user_prompt)

        context.insights.append("### 🤖 Clustering Insights\n" + insights)
        return context