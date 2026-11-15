
# Distributed Big Data Analytics & Machine Learning Pipeline: Amazon 2023 Reviews (Arts, Crafts & Sewing)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://amazon-2023-hadoop-mahout-pipeline.streamlit.app)
![Apache Hadoop](https://img.shields.io/badge/Apache%20Hadoop-3.3.6-66CCFF?logo=apachehadoop&logoColor=black)
![Apache Mahout](https://img.shields.io/badge/Apache%20Mahout-0.13.0-C71A36?logo=apache&logoColor=white)
![Java](https://img.shields.io/badge/Java-OpenJDK%2011-ED8B00?logo=openjdk&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)

An end-to-end distributed data engineering, custom Java MapReduce analytics, distributed machine learning, and interactive visualization pipeline built on **Apache Hadoop 3.3.6 (HDFS & YARN)**, **Apache Mahout 0.13.0**, and **Streamlit**. The pipeline ingests, transforms, analyzes, and models **8,187,011** customer reviews (`8,184,798` vectorized text documents) from the UCSD Amazon Reviews 2023 dataset (`Arts_Crafts_and_Sewing`) on a memory-constrained single-node workstation (**8 GB RAM / 8 vCores**).

---

## Table of Contents
1. [Project Uniqueness & Engineering Highlights](#1-project-uniqueness--engineering-highlights)
2. [Technology Stack & Hardware Environment](#2-technology-stack--hardware-environment)
3. [System Architecture & Pipeline Workflow](#3-system-architecture--pipeline-workflow)
4. [YARN Container & JVM Memory Tuning](#4-yarn-container--jvm-memory-tuning)
5. [Live Interactive Dashboard (Streamlit UI)](#5-live-interactive-dashboard-streamlit-ui)
6. [Repository Structure](#6-repository-structure)
7. [Step-by-Step Execution Procedure & Commands](#7-step-by-step-execution-procedure--commands)
8. [Experimental Results & Evaluation Metrics](#8-experimental-results--evaluation-metrics)
9. [Engineering Challenges & Resolutions](#9-engineering-challenges--resolutions)

---

## 1. Project Uniqueness & Engineering Highlights

Unlike toy datasets or cloud-rented clusters with abundant RAM, this project demonstrates **resource-aware distributed systems engineering** at multi-million-record scale:

1. **Multi-Million-Record Scale on Commodity Hardware (8.18M Reviews on 8 GB RAM):**
   Processes `1.7 GB` of structured CSV data and a `5.6 GB` L2-normalized sparse TF-IDF matrix across 18 HDFS splits on an 8 GB physical RAM workstation without triggering Linux OOM kills or YARN virtual memory container terminations.
2. **Custom Java MapReduce Optimization Patterns:**
   - **Network-Shuffle Reduction via Local Combiners:** Pre-aggregates `(sum_ratings, review_count)` per `parent_asin` inside mapper nodes before shuffling `31.5 MB` of final product statistics.
   - **Two-Stage Distributed Top-K Ranking:** Uses bounded `TreeMap` structures inside mappers to emit only local top-10 candidates per split, reducing global sorting overhead to a single lightweight reducer (`160 bytes` final output).
3. **Hybrid Supervised + Unsupervised Distributed ML (Apache Mahout):**
   - **Complementary Naive Bayes (`trainnb -c`):** Explicitly selected over standard Multinomial Naive Bayes to counter severe real-world class imbalance (`77.40%` Positive vs. `7.15%` Neutral), achieving **80.98% accuracy** across `8,184,798` vectorized documents.
   - **Cosine-Distance K-Means (`k = 5`):** Discovers interpretable customer themes directly from high-dimensional sparse text vectors without dense matrix materialization.
4. **Decoupled Distributed Compute & Interactive Cloud UI:**
   Separates heavy YARN batch execution from a zero-latency **Streamlit** web dashboard (`app.py`), enabling instant exploration of confusion matrices, star distributions, and cluster centroids locally (port `8501`) and on Streamlit Community Cloud.

---

## 2. Technology Stack & Hardware Environment

| Layer | Technology / Specification | Role in Pipeline |
|---|---|---|
| **Storage Layer** | **Apache Hadoop HDFS 3.3.6** | Distributed block storage (`1.7 GB` raw CSV, `1.6 GB` SequenceFiles, `5.6 GB` TF-IDF vectors) |
| **Resource Management** | **Apache Hadoop YARN 3.3.6** | Container scheduling, JVM heap enforcement, and shuffle memory isolation |
| **Distributed Analytics** | **Java MapReduce API (OpenJDK 11)** | Custom `Mapper`, `Combiner`, and `Reducer` jobs (`CsvReviewAnalytics`, `CsvToMahoutSeqFile`) |
| **Distributed ML Engine** | **Apache Mahout 0.13.0** | `seq2sparse` TF-IDF vectorization, Complementary Naive Bayes, Cosine K-Means, `clusterdump` |
| **Visualization & UI** | **Python 3.12, Streamlit, Pandas** | Interactive multi-tab web dashboard (`app.py`) deployed on port `8501` & Streamlit Cloud |
| **Dataset** | **UCSD Amazon Reviews 2023** | `Arts_Crafts_and_Sewing` (`8,187,011` reviews; `8,184,798` non-empty vectorized texts) |
| **Compute Node** | **HP 15s Workstation (Linux)** | 8 GB Physical RAM, 8 vCores (Pseudo-Distributed Single-Node Cluster) |

---

## 3. System Architecture & Pipeline Workflow

```text
+-----------------------------------------------------------------------------------+
| 1. RAW DATA INGESTION & HDFS STAGING                                              |
|    Arts_Crafts_and_Sewing.jsonl.gz (8,187,011 reviews)                            |
|    -> Stream-parsed to clean CSV -> Uploaded to HDFS (/input: 1.7 GB)             |
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
|    Reads verified outputs from reports/ -> Served on Port 8501 & Streamlit Cloud  |
+-----------------------------------------------------------------------------------+

```

---

## 4. YARN Container & JVM Memory Tuning

Running `seq2sparse` and `trainnb` across 18 HDFS splits (`5.6 GB` TF-IDF vectors) on an 8 GB node requires strict memory budgeting so ApplicationMaster, Map, and Reduce containers fit concurrently within physical RAM:

```bash
export HADOOP_HEAPSIZE=2048

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

```

* **Reduced Shuffle Input Buffer (`0.20`) & Merge Threshold (`0.35`):** Forces early disk spills during high-fan-in reduce phases in `seq2sparse` and `trainnb`, preventing `OutOfMemoryError: Java heap space`.
* **Bounded Sort Buffer (`128 MB`) & Heap Headroom:** Leaves off-heap memory for native Zlib decompression and Netty shuffle threads.

---

## 5. Live Interactive Dashboard (Streamlit UI)

The Streamlit web application (`app.py`) provides an interactive interface over the verified Hadoop and Mahout outputs in `reports/`:

* **Tab 1 — MapReduce Analytics:** Interactive bar charts and tables for the 1–5 star rating distribution (`8,187,011` reviews) and Top-10 most reviewed products.
* **Tab 2 — Complementary Naive Bayes Evaluation:** Model KPI cards (`80.98%` accuracy, `0.7867` precision, `0.8098` recall, `0.7512` F1-score), the `3x3` confusion matrix, and raw `testnb` logs.
* **Tab 3 — K-Means Topic Clusters (`k = 5`):** Interactive cluster selector visualizing top weighted TF-IDF terms and customer theme interpretations.
* **Cloud Deployment:** `https://amazon-2023-hadoop-mahout-pipeline.streamlit.app` *(update if using a custom Streamlit subdomain)*
* **Local Server:** `http://localhost:8501`

---

## 6. Repository Structure

```text
amazon_2023_analytics/
├── src/                                      # Custom Java MapReduce & SequenceFile source files
│   ├── CsvReviewAnalytics.java               # Multi-job MapReduce analytics (Avg Rating, Distribution, Top-K)
│   └── CsvToMahoutSeqFile.java               # Distributed CSV -> Mahout SequenceFile converter
├── reports/                                  # Verified analytics & ML evaluation outputs
│   ├── category_rating_distribution.txt      # Star rating counts (1-Star to 5-Star)
│   ├── top10_reviewed_products.txt           # Top 10 ASINs by review volume
│   ├── sample_product_ratings_top50.txt      # Top 50 sample from 31.5 MB product rating aggregation
│   ├── naive_bayes_summary.txt               # Complementary Naive Bayes accuracy & confusion matrix
│   ├── naive_bayes_evaluation.txt            # Full Mahout testnb execution log
│   ├── kmeans_topic_clusters.txt             # Top weighted TF-IDF terms per K-Means cluster (k=5)
│   └── hdfs_exports/                         # HDFS directory manifests and exported label indices
├── app.py                                    # Interactive Streamlit dashboard (Port 8501 / Cloud)
├── requirements.txt                          # Python dependencies (streamlit, pandas)
├── .gitignore                                # Excludes multi-GB datasets, .venv, JARs, and class files
└── README.md                                 # Technical documentation & execution guide

```

---

## 7. Step-by-Step Execution Procedure & Commands

### Phase 1: Data Ingestion & HDFS Staging

```bash
start-dfs.sh && start-yarn.sh
jps

HDFS_BASE="/user/student278/venkat/amazon_2023_arts"
hdfs dfs -mkdir -p ${HDFS_BASE}/input
hdfs dfs -put raw_data/arts_reviews_clean.csv ${HDFS_BASE}/input/
hdfs dfs -du -h ${HDFS_BASE}/input

```

### Phase 2: Compilation & Packaging

```bash
cd ~/amazon_2023_analytics
mkdir -p build
javac -cp "$(hadoop classpath)" -d build src/*.java
jar -cvf amazon-2023-analytics.jar -C build/ .

```

### Phase 3: Distributed MapReduce Analytics

```bash
hadoop jar amazon-2023-analytics.jar CsvReviewAnalytics \
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

```

### Phase 4: Mahout SequenceFile & TF-IDF Vectorization

Converts CSV rows into `SequenceFile<Text, Text>` (`1.6 GB`) with Mahout hierarchical label keys (`/Positive/`, `/Neutral/`, `/Negative/`) and generates L2-normalized TF-IDF vectors (`5.6 GB`, `--minDF 5`, `--maxDFPercent 85`):

```bash
MAHOUT_WORK="${HDFS_BASE}/mahout_work"
hadoop jar amazon-2023-analytics.jar CsvToMahoutSeqFile \
  ${HDFS_BASE}/input \
  ${MAHOUT_WORK}/seqfiles

export MAHOUT_HOME=~/apache-mahout-distribution-0.13.0
export PATH=$PATH:$MAHOUT_HOME/bin

mahout seq2sparse $MEM_OPTS \
  -i ${MAHOUT_WORK}/seqfiles \
  -o ${MAHOUT_WORK}/vectors \
  -lnorm -nv -wt tfidf \
  --minDF 5 --maxDFPercent 85 \
  --numReducers 4

```

### Phase 5: Complementary Naive Bayes Sentiment Classification

```bash
mahout trainnb $MEM_OPTS \
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

```

### Phase 6: Cosine-Distance K-Means Topic Clustering

```bash
mahout kmeans $MEM_OPTS \
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

```

### Phase 7: Launching the Streamlit Web Dashboard (Port 8501)

```bash
cd ~/amazon_2023_analytics
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py --server.port 8501

```

Open **`http://localhost:8501`** in your browser to interact with the dashboard locally.

---

## 8. Experimental Results & Evaluation Metrics

### A. Category-Wide Star Rating Distribution (`8,187,011` Reviews)

| Rating Tier | Review Count | Percentage Share | Sentiment Mapping |
| --- | --- | --- | --- |
| **5 Star** | 5,462,205 | 66.72% | Positive |
| **4 Star** | 874,245 | 10.68% | Positive |
| **3 Star** | 585,217 | 7.15% | Neutral |
| **2 Star** | 407,967 | 4.98% | Negative |
| **1 Star** | 857,377 | 10.47% | Negative |
| **Total** | **8,187,011** | **100.00%** | — |

### B. Top-10 Most Frequently Reviewed Products

| Rank | Parent ASIN | Review Count | Rank | Parent ASIN | Review Count |
| --- | --- | --- | --- | --- | --- |
| 1 | `B01LY9ERQC` | 7,862 | 6 | `B001PNIWT2` | 5,404 |
| 2 | `B0BPRQR2DH` | 6,553 | 7 | `B09XYL6FS7` | 5,389 |
| 3 | `B08KVH58MS` | 6,113 | 8 | `B0C7WDNTMV` | 5,269 |
| 4 | `B0757JLTY7` | 6,030 | 9 | `B09JHWHYYV` | 5,265 |
| 5 | `B07LCTD7C7` | 5,442 | 10 | `B0C2RW2TXH` | 4,908 |

### C. Complementary Naive Bayes Evaluation (`8,184,798` Vectorized Reviews)

* **Correctly Classified Instances:** `6,628,277` (**80.9828%**)
* **Incorrectly Classified Instances:** `1,556,521` (**19.0172%**)
* **Weighted Precision / Recall / F1:** `0.7867` / `0.8098` / `0.7512`

| Actual \ Predicted | `a` (Negative) | `b` (Neutral) | `c` (Positive) | Total Actual |
| --- | --- | --- | --- | --- |
| **`a` = Negative** | **322,430** | 277 | 942,254 | 1,264,961 |
| **`b` = Neutral** | 36,073 | **961** | 548,088 | 585,122 |
| **`c` = Positive** | 28,927 | 902 | **6,304,886** | 6,334,715 |

### D. Discovered K-Means Topic Clusters (`k = 5`, Cosine Distance)

| Cluster ID | Top Weighted Terms | Interpreted Customer Theme |
| --- | --- | --- |
| **VL-5186261** | `very` (0.73), `good` (0.68), `nice` (0.48), `easy` (0.46), `quality` (0.43), `perfect` (0.32) | **Ease of Use & Build Quality** — usability and product finish |
| **VL-2033205** | `great` (2.25), `making` (0.45), `product` (0.45), `price` (0.44), `works` (0.38), `quality` (0.38) | **Value for Money & Utility** — price-to-performance in crafting |
| **VL-5877508** | `just` (1.05), `what` (1.02), `have` (0.88), `like` (0.64), `use` (0.59), `would` (0.57) | **Exact Fit & Expectations** — items matching project specifications |
| **VL-7151045** | `love` (1.68), `used` (0.76), `make` (0.65), `paint` (0.61), `colors` (0.43), `gift` (0.40) | **Painting, Colors & Gifting** — art mediums and handmade gifts |
| **VL-7067819** | `were` (1.15), `more` (1.01), `than` (0.90), `little` (0.83), `one` (0.61), `had` (0.55) | **Quantity, Sizing & Comparison** — pack counts and dimensions |

---

## 9. Engineering Challenges & Resolutions

1. **YARN Container Memory Exhaustion during `seq2sparse` (`27.05 min`):**
* *Cause:* Default reducer shuffle buffers (`0.70`) exceeded physical RAM on an 8 GB node when sorting `5.6 GB` of TF-IDF partial vectors.
* *Fix:* Tuned `mapreduce.reduce.shuffle.input.buffer.percent=0.20`, `merge.percent=0.35`, and capped map/reduce container limits (`1536 MB` / `3072 MB`).


2. **Mahout `IndexInstancesMapper` Label Extraction (`SKIPPED_INSTANCES`):**
* *Cause:* Mahout's `trainnb` parses class labels using leading slash delimiters (`/Label/DocID`) and skips instances when `split -xm mapreduce` alters key metadata via `MultipleOutputs`.
* *Fix:* Encoded `/Positive/`, `/Neutral/`, and `/Negative/` prefixes directly during SequenceFile generation (`CsvToMahoutSeqFile.java`) and fed `tfidf-vectors` directly into `trainnb` and `testnb` with explicit HDFS `--tempDir` paths.



```

