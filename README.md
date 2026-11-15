[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://amazon-2023-hadoop-mahout-pipeline.streamlit.app)

An end-to-end distributed data engineering, MapReduce analytics, machine learning, and interactive visualization pipeline built on **Apache Hadoop 3.3.6 (HDFS & YARN)**, **Apache Mahout 0.13.0**, and **Streamlit**. The pipeline ingests, transforms, analyzes, and models **8,187,011** customer reviews (`8,184,798` vectorized text documents) from the UCSD Amazon Reviews 2023 dataset (`Arts_Crafts_and_Sewing`) on a memory-constrained single-node cluster (**8 GB RAM / 8 vCores**).

---

## Table of Contents
1. [Live Interactive Dashboard (Streamlit UI)](#1-live-interactive-dashboard-streamlit-ui)
2. [System Architecture & Pipeline Workflow](#2-system-architecture--pipeline-workflow)
3. [Environment & YARN Memory Tuning](#3-environment--yarn-memory-tuning)
4. [Repository Structure](#4-repository-structure)
5. [Step-by-Step Execution Procedure & Commands](#5-step-by-step-execution-procedure--commands)
   - [Phase 1: Data Ingestion & HDFS Staging](#phase-1-data-ingestion--hdfs-staging)
   - [Phase 2: Compilation & Packaging](#phase-2-compilation--packaging)
   - [Phase 3: Distributed MapReduce Analytics](#phase-3-distributed-mapreduce-analytics)
   - [Phase 4: Mahout SequenceFile & TF-IDF Vectorization](#phase-4-mahout-sequencefile--tf-idf-vectorization)
   - [Phase 5: Complementary Naive Bayes Sentiment Classification](#phase-5-complementary-naive-bayes-sentiment-classification)
   - [Phase 6: Cosine-Distance K-Means Topic Clustering](#phase-6-cosine-distance-kmeans-topic-clustering)
   - [Phase 7: Launching the Streamlit Web Dashboard (Port 8501)](#phase-7-launching-the-streamlit-web-dashboard-port-8501)
6. [Experimental Results & Evaluation Metrics](#6-experimental-results--evaluation-metrics)
7. [Engineering Challenges & Resolutions](#7-engineering-challenges--resolutions)

---

## 1. Live Interactive Dashboard (Streamlit UI)

An interactive web application (`app.py`) built with **Streamlit** and **Pandas** visualizes the distributed outputs stored in `reports/` across three analytical views:
- **Tab 1 — MapReduce Analytics:** Interactive bar charts and tabular breakdowns of the category-wide star rating distribution (`8,187,011` reviews) and Top-10 most reviewed products.
- **Tab 2 — Complementary Naive Bayes Evaluation:** Summary KPI cards (`80.98%` accuracy, `0.7512` weighted F1-score), the `3x3` multi-class confusion matrix, and raw Mahout `testnb` logs.
- **Tab 3 — K-Means Topic Clusters (`k = 5`):** Interactive cluster selector rendering top TF-IDF centroid weights and interpreted customer themes.

- **Streamlit Cloud URL:** `https://amazon-2023-hadoop-mahout-pipeline.streamlit.app` *(update if you chose a custom subdomain)*
- **Local Port:** `http://localhost:8501`

---

## 2. System Architecture & Pipeline Workflow

```text
+-----------------------------------------------------------------------------------+
| 1. RAW DATA INGESTION (Local -> HDFS)                                             |
|    Arts_Crafts_and_Sewing.jsonl.gz (8,187,011 reviews)                            |
|    -> Stream-parsed to structured CSV (1.7 GB on HDFS: /input)                    |
+-----------------------------------------+-----------------------------------------+
                                          |
                 +------------------------+------------------------+
                 v                                                 v
+-----------------------------------+   +-------------------------------------------+
| 2. CUSTOM HADOOP MAPREDUCE        |   | 3. APACHE MAHOUT 0.13.0 ML PIPELINE       |
|    (Java MapReduce API)           |   |    (Distributed YARN Execution)           |
|                                   |   |                                           |
|  - Job 1: Product Rating Stats    |   |  - SequenceFile Conversion (/seqfiles)    |
|    (Combiner + Reducer -> 31.5MB) |   |    Keys: /Positive/..., /Neutral/...,     |
|                                   |   |          /Negative/... (1.6 GB)           |
|  - Job 2: Rating Distribution     |   |                                           |
|    (1-Star to 5-Star -> 186 B)    |   |  - TF-IDF Vectorization (seq2sparse)      |
|                                   |   |    L2-normalized vectors (5.6 GB)         |
|  - Job 3a/3b: Two-Stage Top-10    |   |                                           |
|    Most Reviewed Products (160 B) |   |  - Complementary Naive Bayes              |
+-----------------+-----------------+   |    (trainnb / testnb -> 80.98% Accuracy)  |
                  |                     |                                           |
                  |                     |  - Cosine-Distance K-Means (k = 5)        |
                  |                     |    (kmeans + clusterdump -> 5 Topics)     |
                  |                     +---------------------+---------------------+
                  |                                           |
                  +---------------------+---------------------+
                                        v
+-----------------------------------------------------------------------------------+
| 4. INTERACTIVE VISUALIZATION LAYER (Streamlit UI - app.py)                        |
|    Reads verified summary artifacts from reports/ -> Served on Port 8501 / Cloud  |
+-----------------------------------------------------------------------------------+
3. Environment & YARN Memory TuningSoftware & Hardware StackComponentSpecificationOS & HardwareLinux (HP 15s Workstation), 8 GB Physical RAM, 8 vCoresJava RuntimeOpenJDK 11Distributed Storage & ComputeApache Hadoop 3.3.6 (Pseudo-Distributed HDFS + YARN)Distributed Machine LearningApache Mahout 0.13.0Visualization & Web UIPython 3.12, Streamlit, PandasDatasetUCSD Amazon Reviews 2023 — Arts_Crafts_and_Sewing (8,187,011 records)Memory Optimization Strategy (Preventing YARN Container OOM Kills)Processing a 5.6 GB sparse TF-IDF matrix across 18 HDFS input splits on an 8 GB node requires strict container sizing and spill tuning so ApplicationMaster, Map, and Reduce containers fit concurrently within physical RAM without triggering Linux OOM killer or YARN virtual memory limits:Bashexport HADOOP_HEAPSIZE=2048

MEM_OPTS="\
-Dyarn.app.mapreduce.am.resource.mb=1024 \
-Dyarn.app.mapreduce.am.command-opts=-Xmx768m \
-Dmapreduce.map.memory.mb=1536 \
-Dmapreduce.map.java.opts=-Xmx1150m \
-Dmapreduce.reduce.memory.mb=3072 \
-Dmapreduce.reduce.java.opts=-Xmx2400m \
-Dmapreduce.reduce.shuffle.input.buffer.percent=0.20 \
-Dmapreduce.reduce.shuffle.merge.percent=0.35 \
-Dmapreduce.task.io.sort.mb=128"
4. Repository StructurePlaintextamazon_2023_analytics/
├── src/                                      # Custom Java MapReduce & SequenceFile source files
│   ├── CsvReviewAnalytics.java               # Multi-job MapReduce analytics (Avg Rating, Distribution, Top-K)
│   └── CsvToMahoutSeqFile.java               # Distributed CSV -> Mahout SequenceFile converter
├── reports/                                  # Verified analytics & ML evaluation outputs
│   ├── category_rating_distribution.txt      # Star rating counts (1-Star to 5-Star)
│   ├── top10_reviewed_products.txt           # Top 10 ASINs by review volume
│   ├── sample_product_ratings_top50.txt      # Top 50 sample from 32 MB product rating aggregation
│   ├── naive_bayes_summary.txt               # Complementary Naive Bayes accuracy & confusion matrix
│   ├── naive_bayes_evaluation.txt            # Full Mahout testnb execution log
│   └── kmeans_topic_clusters.txt             # Top weighted TF-IDF terms per K-Means cluster (k=5)
├── app.py                                    # Interactive Streamlit dashboard (Port 8501 / Cloud)
├── requirements.txt                          # Python dependencies (streamlit, pandas)
├── .gitignore                                # Excludes multi-GB datasets, .venv, JARs, and class files
└── README.md                                 # Technical documentation & execution guide
5. Step-by-Step Execution Procedure & CommandsPhase 1: Data Ingestion & HDFS StagingBashstart-dfs.sh && start-yarn.sh
HDFS_BASE="/user/student278/venkat/amazon_2023_arts"
hdfs dfs -mkdir -p ${HDFS_BASE}/input
hdfs dfs -put raw_data/arts_reviews_clean.csv ${HDFS_BASE}/input/
Phase 2: Compilation & PackagingBashcd ~/amazon_2023_analytics
mkdir -p build
javac -cp "$(hadoop classpath)" -d build src/*.java
jar -cvf amazon-2023-analytics.jar -C build/ .
Phase 3: Distributed MapReduce AnalyticsBashhadoop jar amazon-2023-analytics.jar CsvReviewAnalytics \
  ${HDFS_BASE}/input \
  ${HDFS_BASE}/out_rating \
  ${HDFS_BASE}/out_distribution \
  ${HDFS_BASE}/out_topk_stage1 \
  ${HDFS_BASE}/out_topk

mkdir -p reports
hdfs dfs -getmerge ${HDFS_BASE}/out_distribution reports/category_rating_distribution.txt
hdfs dfs -getmerge ${HDFS_BASE}/out_topk reports/top10_reviewed_products.txt
hdfs dfs -getmerge ${HDFS_BASE}/out_rating reports/product_ratings_summary.txt
head -n 50 reports/product_ratings_summary.txt > reports/sample_product_ratings_top50.txt
Phase 4: Mahout SequenceFile & TF-IDF VectorizationBashMAHOUT_WORK="${HDFS_BASE}/mahout_work"
hadoop jar amazon-2023-analytics.jar CsvToMahoutSeqFile \
  ${HDFS_BASE}/input \
  ${MAHOUT_WORK}/seqfiles

mahout seq2sparse $MEM_OPTS \
  -i ${MAHOUT_WORK}/seqfiles \
  -o ${MAHOUT_WORK}/vectors \
  -lnorm -nv -wt tfidf \
  --minDF 5 --maxDFPercent 85 \
  --numReducers 4
Phase 5: Complementary Naive Bayes Sentiment ClassificationBashmahout trainnb $MEM_OPTS \
  -i ${MAHOUT_WORK}/vectors/tfidf-vectors \
  -o ${MAHOUT_WORK}/nb-model \
  -li ${MAHOUT_WORK}/nb-labelindex \
  --tempDir ${MAHOUT_WORK}/nb-temp \
  -c -ow

mahout testnb $MEM_OPTS \
  -i ${MAHOUT_WORK}/vectors/tfidf-vectors \
  -m ${MAHOUT_WORK}/nb-model \
  -l ${MAHOUT_WORK}/nb-labelindex \
  -o ${MAHOUT_WORK}/nb-test-results \
  --tempDir ${MAHOUT_WORK}/nb-test-temp \
  -c -ow 2>&1 | tee reports/naive_bayes_evaluation.txt

tail -n 35 reports/naive_bayes_evaluation.txt > reports/naive_bayes_summary.txt
Phase 6: Cosine-Distance K-Means Topic ClusteringBashmahout kmeans $MEM_OPTS \
  -i ${MAHOUT_WORK}/vectors/tfidf-vectors \
  -c ${MAHOUT_WORK}/kmeans-initial-clusters \
  -o ${MAHOUT_WORK}/kmeans-clusters \
  -dm org.apache.mahout.common.distance.CosineDistanceMeasure \
  -k 5 -cd 0.05 -x 3 -ow

FINAL_CLUSTERS=$(hdfs dfs -ls -d ${MAHOUT_WORK}/kmeans-clusters/clusters-*-final \vert{} awk '{print$NF}' | tail -n 1)

mahout clusterdump \
  -i ${FINAL_CLUSTERS} \
  -d ${MAHOUT_WORK}/vectors/dictionary.file-0 \
  -dt sequencefile \
  -n 12 -b 100 \
  -o reports/kmeans_topic_clusters.txt
Phase 7: Launching the Streamlit Web Dashboard (Port 8501)Bashcd ~/amazon_2023_analytics
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py --server.port 8501
Open http://localhost:8501 in your browser to interact with the dashboard locally.6. Experimental Results & Evaluation MetricsA. Category-Wide Star Rating Distribution (8,187,011 Reviews)Rating TierReview CountPercentage ShareSentiment Mapping5 Star5,462,20566.72%Positive4 Star874,24510.68%Positive3 Star585,2177.15%Neutral2 Star407,9674.98%Negative1 Star857,37710.47%NegativeTotal8,187,011100.00%—B. Top-10 Most Frequently Reviewed ProductsRankParent ASINReview CountRankParent ASINReview Count1B01LY9ERQC7,8626B001PNIWT25,4042B0BPRQR2DH6,5537B09XYL6FS75,3893B08KVH58MS6,1138B0C7WDNTMV5,2694B0757JLTY76,0309B09JHWHYYV5,2655B07LCTD7C75,44210B0C2RW2TXH4,908C. Complementary Naive Bayes Evaluation (8,184,798 Vectorized Reviews)Correctly Classified Instances: 6,628,277 (80.9828%)Incorrectly Classified Instances: 1,556,521 (19.0172%)Weighted Precision / Recall / F1: 0.7867 / 0.8098 / 0.7512Actual \ Predicteda (Negative)b (Neutral)c (Positive)Total Actuala = Negative322,430277942,2541,264,961b = Neutral36,073961548,088585,122c = Positive28,9279026,304,8866,334,715D. Discovered K-Means Topic Clusters (k = 5, Cosine Distance)Cluster IDTop Weighted TermsInterpreted Customer ThemeVL-5186261very (0.73), good (0.68), nice (0.48), easy (0.46), quality (0.43), perfect (0.32)Ease of Use & Build QualityVL-2033205great (2.25), making (0.45), product (0.45), price (0.44), works (0.38), quality (0.38)Value for Money & UtilityVL-5877508just (1.05), what (1.02), have (0.88), like (0.64), use (0.59), would (0.57)Exact Fit & ExpectationsVL-7151045love (1.68), used (0.76), make (0.65), paint (0.61), colors (0.43), gift (0.40)Painting, Colors & GiftingVL-7067819were (1.15), more (1.01), than (0.90), little (0.83), one (0.61), had (0.55)Quantity, Sizing & Comparison7. Engineering Challenges & ResolutionsYARN Container Memory Exhaustion during seq2sparse (27.05 min): Tuned mapreduce.reduce.shuffle.input.buffer.percent=0.20, merge.percent=0.35, and capped map/reduce container limits (1536 MB / 3072 MB) to fit within 8 GB RAM.Mahout IndexInstancesMapper Label Extraction (SKIPPED_INSTANCES): Encoded /Positive/, /Neutral/, and /Negative/ prefixes directly during SequenceFile generation (CsvToMahoutSeqFile.java) and passed explicit HDFS --tempDir paths to trainnb and testnb.
