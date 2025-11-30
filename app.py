import os
import streamlit as st
import pandas as pd

from agents import AgentContext
from agents.planner_agent import PlannerAgent
from agents.llm_client import generate_llm_response

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "context" not in st.session_state:
    st.session_state["context"] = None

if "df" not in st.session_state:
    st.session_state["df"] = None

if "plots" not in st.session_state:
    st.session_state["plots"] = None

if "report_path" not in st.session_state:
    st.session_state["report_path"] = None

# ==========================================
# STREAMLIT CONFIG
# ==========================================
st.set_page_config(
    page_title="AI Multi-Agent Data Analyst",
    layout="wide",
    page_icon="🤖"
)

st.markdown("""
<h1 style='text-align: center;'>🤖 AI Multi-Agent Data Analyst Dashboard</h1>
<p style='text-align: center; font-size:16px; opacity:0.8;'>
Upload a CSV, explore visualizations, generate insights, and chat with your dataset — all powered by Multi-Agent AI.
</p>
""", unsafe_allow_html=True)


# ==========================================
# FILE UPLOAD
# ==========================================
uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    os.makedirs("data", exist_ok=True)
    dataset_path = os.path.join("data", "uploaded.csv")

    with open(dataset_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ File uploaded successfully!")

    # ----------------------------
    # RUN FULL ANALYSIS
    # ----------------------------
    if st.button("🚀 Run Full Analysis"):
        with st.spinner("Running multi-agent analysis pipeline..."):
            context = AgentContext(dataset_path=dataset_path)
            planner = PlannerAgent(use_llm_insights=True)
            context = planner.run_pipeline(context)

        # Save to session_state
        st.session_state["context"] = context
        st.session_state["df"] = context.df
        st.session_state["plots"] = context.plots
        st.session_state["report_path"] = context.report_path

        st.success("🎉 Analysis Completed!")


# ==========================================
# TABS SECTION
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📄 Report", "📊 Visualizations", "💡 Insights", "💬 Chat", "🤖 ML Feature Importance", "🌀 Clustering"]
)


# ==========================================
# TAB 1 — REPORT
# ==========================================
with tab1:
    st.header("📄 Generated Analytical Report")

    context = st.session_state.get("context")
    if context is None:
        st.info("Run full analysis to generate the report.")
        st.stop()

    # Show report markdown
    with open(context.report_path, "r", encoding="utf-8") as f:
        st.markdown(f.read(), unsafe_allow_html=True)

    # -----------------------------
    # NUMERIC TABLE
    # -----------------------------
    if "numeric_table" in context.summary:
        st.subheader("📊 Numeric Feature Summary")
        st.dataframe(context.summary["numeric_table"])

    # -----------------------------
    # CATEGORICAL TABLE
    # -----------------------------
    if "categorical_table" in context.summary:
        st.subheader("🔠 Categorical Feature Summary")
        st.dataframe(context.summary["categorical_table"])

    # -----------------------------
    # MISSING VALUE TABLE
    # -----------------------------
    if "missing_table" in context.summary:
        st.subheader("❗ Missing Value Table")
        st.dataframe(context.summary["missing_table"])

    # PDF EXPORT
    from utils.pdf_exporter import export_report_to_pdf

    st.markdown("### 📥 Download Report as PDF")
    if st.button("Generate PDF"):
        pdf_path = export_report_to_pdf(
            report_text=open(context.report_path, "r", encoding="utf-8").read(),
            plots=context.plots
        )
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📄 Download PDF",
                data=pdf_file,
                file_name="AI_Data_Analysis_Report.pdf",
                mime="application/pdf",
            )


# ==========================================
# TAB 2 — VISUALIZATIONS
# ==========================================
with tab2:
    st.header("📊 Visualizations")

    context = st.session_state.get("context")
    if context is None:
        st.info("Run analysis first.")
        st.stop()

    visual_sections = context.visual_structure

    for section, plots in visual_sections.items():
        if len(plots) == 0:
            continue

        st.markdown(f"## 📦 {section.replace('_', ' ').title()}")
        cols = st.columns(2)

        for i, p in enumerate(plots):
            with cols[i % 2]:
                st.image(p["path"], use_container_width=True)

        st.markdown("---")


# ==========================================
# TAB 3 — INSIGHTS (CLEAN + STRUCTURED)
# ==========================================
with tab3:
    st.header("💡 AI Insights (Clean + Structured)")

    context = st.session_state.get("context")
    if context is None:
        st.info("Run analysis first.")
        st.stop()

    rule_based = []
    llm_based = []
    ml_based = []

    for ins in context.insights:
        if "ML Feature" in ins:
            ml_based.append(ins)
        elif "Graph-Aware" in ins or "LLM" in ins:
            llm_based.append(ins)
        else:
            rule_based.append(ins)

    if rule_based:
        st.subheader("📊 Rule-Based Insights")
        st.markdown(rule_based[0], unsafe_allow_html=True)

    if llm_based:
        st.subheader("🤖 LLM Insights")
        st.markdown(llm_based[0], unsafe_allow_html=True)

    if ml_based:
        st.subheader("📈 ML Insights (Feature Importance)")
        st.markdown(ml_based[0], unsafe_allow_html=True)


# ==========================================
# TAB 4 — CHAT WITH DATASET
# ==========================================
with tab4:
    st.header("💬 Chat with Dataset")

    df = st.session_state.get("df")
    if df is None:
        st.info("Upload dataset and run analysis first.")
        st.stop()

    question = st.text_input("Ask your dataset something:")

    if st.button("Ask AI"):
        schema_info = {
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
        }
        sample_rows = df.head(5).to_dict(orient="records")

        system_prompt = (
            "You are a data analyst AI. Answer questions based strictly "
            "using the provided schema and sample data."
        )

        user_prompt = f"""
Schema:
{schema_info}

Sample Rows:
{sample_rows}

User Question:
{question}
"""

        with st.spinner("AI interpreting data..."):
            answer = generate_llm_response(system_prompt, user_prompt)

        st.markdown("### 🧠 Answer:")
        st.write(answer)


# ==========================================
# TAB 5 — ML FEATURE IMPORTANCE
# ==========================================
with tab5:
    st.header("🤖 ML Feature Importance Analysis")

    df = st.session_state.get("df")
    if df is None:
        st.info("Upload dataset and run analysis first.")
        st.stop()

    target_col = st.selectbox("Select Target Column", df.columns)

    if st.button("Run ML Analysis"):
        from agents.ml_agent import MLAgent
        from agents.ml_insights_agent import MLInsightsAgent

        with st.spinner("Training model & computing importance..."):
            context = AgentContext(dataset_path="data/uploaded.csv")
            context.df = df

            ml = MLAgent()
            context = ml.run(context, target_col)

            ml_insights = MLInsightsAgent()
            context = ml_insights.run(context)

        st.subheader("📊 Feature Importance Plot")
        st.image(context.feature_importance["plot_path"], use_container_width=True)

        st.subheader("🧠 ML Insights")
        st.markdown(context.insights[-1], unsafe_allow_html=True)

with tab6:
    st.header("🌀 Clustering Analysis (KMeans)")

    df = st.session_state.get("df")
    if df is None:
        st.info("Run analysis first.")
        st.stop()

    from agents.clustering_agent import ClusteringAgent
    from agents.clustering_insights_agent import ClusteringInsightsAgent

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns for clustering.")
        st.stop()

    if st.button("Run Clustering"):
        with st.spinner("Clustering data..."):
            context = AgentContext("data/uploaded.csv")
            context.df = df

            cluster_agent = ClusteringAgent()
            context = cluster_agent.run(context)

            cluster_insights = ClusteringInsightsAgent()
            context = cluster_insights.run(context)

        st.subheader("📊 Cluster Scatter Plot")
        st.image(context.clustering["plot_path"], use_container_width=True)

        st.subheader("📘 Cluster Summary Table")
        st.dataframe(context.clustering["cluster_stats"])

        st.subheader("🧠 LLM Insights")
        st.markdown(context.insights[-1], unsafe_allow_html=True)
