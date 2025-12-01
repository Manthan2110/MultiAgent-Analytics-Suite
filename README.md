# 🚀 InsightForge AI – Multi-Agent Automated Data Analyst

_“Transform any dataset into insights — instantly.”_ 📊🤖

**InsightForge AI** is an advanced **AI-powered multi-agent data analyst** that automatically performs **EDA, clustering, ML insights, data cleaning analysis, visualizations, and report generation** using intelligent agents.

Powered by **LLMs (Gemini / OpenAI)**, **Plotly**, **Pandas**, and **Streamlit**, it converts messy datasets into **beautiful dashboards, summaries, and ready-to-share analytical reports** — all with just one upload.

* * *

## 🧠 Problem Statement

Exploratory Data Analysis (EDA) is time-consuming.  
Most users (even analysts) struggle with:

> ❓ _“How do I understand this dataset quickly?”_  
> ❓ _“What patterns, outliers, or trends matter?”_  
> ❓ _“Which features are important for prediction?”_  
> ❓ _“How do I generate visualizations without coding?”_

**InsightForge AI solves this.**  
Upload a CSV → instantly receive:

✔ Interactive Plotly charts  
✔ LLM-powered insights  
✔ Clustering analysis  
✔ Feature importance  
✔ Data cleaning recommendations  
✔ A structured analytics report  
✔ Chat-based Q&A with your dataset

* * *

## 🌐 Interface Preview

### 📊 Dashboard – Automated EDA (Streamlit)
> Interactive distribution plots, boxplots, correlation maps, and time-series.
<img width="1918" height="1015" alt="image" src="https://github.com/user-attachments/assets/d950128e-12fa-4bc4-b913-a418c4d2d91b" />

<img width="1918" height="1012" alt="image" src="https://github.com/user-attachments/assets/65e8c906-a739-4fad-b1c3-66448ab63e95" />


### 🤖 AI Insights
> LLM-written insights for numeric features, categories, trends, and patterns.
<img width="1918" height="1017" alt="image" src="https://github.com/user-attachments/assets/482fc0fe-0507-410e-b7f1-dc846704ce64" />

<img width="1918" height="1013" alt="image" src="https://github.com/user-attachments/assets/024cf1fe-0816-4138-8e09-9eb5cd6ebe34" />

### 🌀 Clustering Visualization
> PCA-based cluster scatter, cluster centroids, and AI interpretations.
<img width="1125" height="868" alt="image" src="https://github.com/user-attachments/assets/3c497fb7-9efc-4cb7-bdbb-91718475f8ef" />

### 💬 Chat with Dataset
> Query your dataset using natural language.
<img width="1907" height="916" alt="image" src="https://github.com/user-attachments/assets/148a5e51-1bf6-4c69-b2ae-f10dce8ce1af" />

<img width="1902" height="920" alt="image" src="https://github.com/user-attachments/assets/7f467ac3-e355-4fe9-a9a5-4f032589ad21" />


* * *

## 🏗️ System Architecture

              ┌──────────────────────────────────────────┐
              │             User Uploads CSV             │
              └──────────────────────────────────────────┘
                                  │
                                  ▼
     ┌─────────────── Multi-Agent Processing Pipeline ────────────────┐
     │                                                                │
     │  DataLoader → EDAAgent → VisualizationAgent (Plotly)           │
     │      → InsightsAgent → LLMInsightsAgent → MLAgent              │
     │      → ClusteringAgent → CleaningAgent → ReportGenerator       │
     │                                                                │
     └────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
            ┌────────────────────────────────────────────────┐
            │ Interactive Dashboard • Insights • Visuals • PDF │
            └────────────────────────────────────────────────┘
    

* * *

## 🚀 Key Features

### 🧩 1\. Multi-Agent Architecture

Each task is handled by a specialized agent:

*   Data Loader Agent
*   EDA Agent
*   Visualization Agent (Plotly)
*   Rule-Based Insights Agent
*   LLM Insights Agent
*   ML Feature Importance Agent
*   Clustering Agent (KMeans + PCA)
*   Data Cleaning Agent
*   Report Agent
    

* * *

### 📊 2\. Automated EDA

*   Dataset shape, memory usage, and duplicates
*   Automatically detects numeric, categorical, boolean, datetime fields
*   Missing values + heatmap
*   Skewness, kurtosis, outliers, distributions
*   Category frequency analysis
  
* * *

### 📈 3\. Interactive Plotly Visualizations

*   Histograms  
*   Boxplots
*   Violin plots
*   Category frequency bars
*   Category vs numeric analysis
*   Correlation heatmap
*   Pairplot (scatter matrix)
*   Time-series trends
*   Cluster plots (PCA 2D)
    
* * *

### 🤖 4\. LLM-Powered Insights

Using OpenAI/Gemini:

*   Numeric feature insights
*   Categorical feature insights
*   Trend detection
*   Correlation explanations
*   Cluster interpretation
*   Data quality recommendations
    

* * *

### 🧮 5\. Machine Learning Feature Importance

*   Auto model selection (RandomForest)
*   Feature impact scoring
*   Plotly interactive importance chart
*   AI-written insights

* * *

### 🌀 6\. Clustering Analysis (KMeans)

*   Automatic K detection
*   PCA-based cluster scatter
*   Cluster summary table
*   LLM-driven interpretation

* * *

### 🧾 7\. Automated Report Generation

*   Clean HTML report
*   Tables for numeric, categorical, missing data
*   Embedded insights
*   PDF export ready
    

* * *

### 💬 8\. Chat With Dataset

Ask questions like:
*   “Which category has the highest sales?”
*   “What is the average value of feature X?”
*   “Summarize the dataset in 2 lines.”
    

LLM responds using:

*   Schema
*   Sample rows
*   Data context
    

* * *

## 📁 Project Structure

| Folder / File | Description |
| --- | --- |
| app.py | Streamlit dashboard |
| agents/ | All agent modules (EDA, ML, LLM, clustering, cleaning) |
| utils/pdf_exporter.py | HTML → PDF generator |
| requirements.txt | Python dependencies |
| .streamlit/config.toml | UI theme config |
| README.md | Documentation |

* * *

## 🔧 Technologies Used

*   **Python**
*   **Streamlit**
*   **Plotly**
*   **Pandas / NumPy**
*   **scikit-learn**
*   **OpenAI / Gemini APIs**
*   **ReportLab or FPDF**
*   **Multi-Agent Orchestration Architecture**
    

* * *

## ⚙️ Installation & Setup

### 1️⃣ Clone the repo
    git clone https://github.com/Manthan2110/MultiAgent-Analytics-Suite.git
    cd MultiAgent-Analytics-Suite


### 2️⃣ Create virtual environment
    python -m venv venv
    venv\Scripts\activate        # Windows
    source venv/bin/activate     # macOS/Linux
    

### 3️⃣ Install dependencies
    pip install -r requirements.txt
    

### 4️⃣ Add your API key (OpenAI / Gemini)

Create `.env` file:

    OPENAI_API_KEY=your_key_here
    

### 5️⃣ Run the app

    streamlit run app.py
    

* * *

## 📈 Example Outputs

### 🔍 Insights Example

    • The feature “age” shows right-skew distribution.
    • Missing values detected in column “income”.
    • Category “High” contributes 42% of total records.
    • Cluster 2 contains higher-income groups with lower variance.
    

### 🌀 Clustering Stats Sample

    Cluster 0:
      mean age: 35.4
      mean income: $54,200
      
    Cluster 1:
      mean age: 50.1
      mean income: $82,900
    

* * *

## 🎯 Future Enhancements

| Feature | Description |
| --- | --- |
| 🔮 Auto Feature Engineering | Suggest transformations & encodings |
| 📊 Smart Chart Selector | AI chooses best chart type automatically |
| 🧠 Conversational Memory | Multi-turn dialogue with dataset |
| 🗃️ Multiple Dataset Support | Compare two datasets side-by-side |
| ⚙️ API Mode | Use as a backend service for other apps |
| 🌐 OAuth Login | Personalized workspaces |

* * *

## 👨‍💻 Author

Made with ❤️ and 💡 by **Manthan Jadav**

*   🌐 LinkedIn: [https://www.linkedin.com/in/manthanjadav/](https://www.linkedin.com/in/manthanjadav/)
*   💻 GitHub: [https://github.com/Manthan2110](https://github.com/Manthan2110)
*   📧 Email: [manthanjadav746@gmail.com](mailto:manthanjadav746@gmail.com)
    

* * *

## 📜 License

MIT License – feel free to use, modify, and improve.
