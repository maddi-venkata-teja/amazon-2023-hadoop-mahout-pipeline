import os
import re
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Amazon 2023 Big Data & ML Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Amazon 2023 Reviews: Hadoop MapReduce & Mahout ML Dashboard")
st.caption("Dataset: Arts, Crafts & Sewing (8,187,011 Reviews) | Stack: Apache Hadoop 3.3.6 & Apache Mahout 0.13.0")

# --- Top KPI Row ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews Processed", "8,187,011")
col2.metric("Vectorized Documents", "8,184,798")
col3.metric("Naive Bayes Accuracy", "80.98%", "Weighted F1: 0.7512")
col4.metric("K-Means Clusters", "5 Topics", "Cosine Distance")

st.divider()

tab1, tab2, tab3 = st.tabs([
    "1. MapReduce Analytics",
    "2. Complementary Naive Bayes",
    "3. K-Means Topic Clusters (k=5)"
])

# ==================== TAB 1: MAPREDUCE ====================
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Category-Wide Star Rating Distribution")
        dist_file = "reports/category_rating_distribution.txt"
        rows = []
        if os.path.exists(dist_file):
            with open(dist_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        rows.append({"Star Rating": parts[1].replace("_", " "), "Review Count": int(parts[2])})
        if not rows:
            rows = [
                {"Star Rating": "1 Star", "Review Count": 857377},
                {"Star Rating": "2 Star", "Review Count": 407967},
                {"Star Rating": "3 Star", "Review Count": 585217},
                {"Star Rating": "4 Star", "Review Count": 874245},
                {"Star Rating": "5 Star", "Review Count": 5462205},
            ]
        df_dist = pd.DataFrame(rows)
        st.bar_chart(df_dist.set_index("Star Rating")["Review Count"], color="#FF9900")
        st.dataframe(df_dist, use_container_width=True, hide_index=True)

    with c2:
        st.subheader("Top-10 Most Frequently Reviewed Products")
        top_file = "reports/top10_reviewed_products.txt"
        top_rows = []
        if os.path.exists(top_file):
            with open(top_file) as f:
                for idx, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        top_rows.append({"Rank": idx, "Parent ASIN": parts[0], "Review Count": int(parts[1])})
        if not top_rows:
            top_rows = [
                {"Rank": 1, "Parent ASIN": "B01LY9ERQC", "Review Count": 7862},
                {"Rank": 2, "Parent ASIN": "B0BPRQR2DH", "Review Count": 6553},
                {"Rank": 3, "Parent ASIN": "B08KVH58MS", "Review Count": 6113},
            ]
        df_top = pd.DataFrame(top_rows)
        st.bar_chart(df_top.set_index("Parent ASIN")["Review Count"], color="#146EB4")
        st.dataframe(df_top, use_container_width=True, hide_index=True)

# ==================== TAB 2: NAIVE BAYES ====================
with tab2:
    st.subheader("Complementary Naive Bayes Evaluation (8,184,798 Instances)")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Correctly Classified", "6,628,277 (80.98%)")
    m2.metric("Weighted Precision", "0.7867")
    m3.metric("Weighted Recall", "0.8098")
    m4.metric("Weighted F1-Score", "0.7512")

    st.markdown("#### Confusion Matrix (Actual vs. Predicted)")
    cm_df = pd.DataFrame(
        [
            [322430, 277, 942254, 1264961],
            [36073, 961, 548088, 585122],
            [28927, 902, 6304886, 6334715],
        ],
        index=["Actual: Negative (a)", "Actual: Neutral (b)", "Actual: Positive (c)"],
        columns=["Predicted: Negative (a)", "Predicted: Neutral (b)", "Predicted: Positive (c)", "Total Actual"]
    )
    st.dataframe(cm_df, use_container_width=True)

    with st.expander("View Raw Mahout Evaluation Report (reports/naive_bayes_summary.txt)"):
        if os.path.exists("reports/naive_bayes_summary.txt"):
            with open("reports/naive_bayes_summary.txt") as f:
                st.code(f.read(), language="text")

# ==================== TAB 3: K-MEANS CLUSTERING ====================
with tab3:
    st.subheader("Cosine-Distance K-Means Discovered Topic Clusters (k = 5)")

    cluster_meta = {
        "VL-5186261": ("Ease of Use & Build Quality", "Customers praising usability, finish, and overall craftsmanship."),
        "VL-2033205": ("Value for Money & Utility", "Reviews highlighting price-to-performance and reliability in crafting."),
        "VL-5877508": ("Exact Fit & Expectations", "Buyers confirming the item matched their exact project requirements."),
        "VL-7151045": ("Painting, Colors & Gifting", "Art supplies, pigment/color quality, and handmade gift projects."),
        "VL-7067819": ("Quantity, Sizing & Comparison", "Feedback comparing pack counts, dimensions, and size expectations."),
    }

    clusters = {}
    cluster_file = "reports/kmeans_topic_clusters.txt"
    if os.path.exists(cluster_file):
        current_id = None
        with open(cluster_file) as f:
            for line in f:
                id_match = re.search(r'"identifier":"(VL-\d+)"', line)
                if id_match:
                    current_id = id_match.group(1)
                    clusters[current_id] = []
                elif "=>" in line and current_id:
                    term, weight = line.strip().split("=>")
                    clusters[current_id].append((term.strip(), float(weight.strip())))

    if clusters:
        selected = st.selectbox(
            "Select a Discovered Cluster:",
            list(clusters.keys()),
            format_func=lambda cid: f"{cid} — {cluster_meta.get(cid, ('Topic Cluster', ''))[0]}"
        )
        title, desc = cluster_meta.get(selected, ("Topic Cluster", ""))
        st.info(f"**Interpreted Theme:** {title} — {desc}")

        df_terms = pd.DataFrame(clusters[selected], columns=["Term", "TF-IDF Centroid Weight"])
        col_chart, col_tbl = st.columns([2, 1])
        with col_chart:
            st.bar_chart(df_terms.set_index("Term")["TF-IDF Centroid Weight"], color="#2E8B57")
        with col_tbl:
            st.dataframe(df_terms, use_container_width=True, hide_index=True)

    with st.expander("View Raw Mahout ClusterDumper Output (reports/kmeans_topic_clusters.txt)"):
        if os.path.exists(cluster_file):
            with open(cluster_file) as f:
                st.code(f.read(), language="text")
