# Distributed Big Data Analytics & Machine Learning Pipeline: Amazon 2023 Reviews (Arts, Crafts & Sewing)

An end-to-end distributed data engineering, MapReduce analytics, and machine learning pipeline built on **Apache Hadoop 3.3.6 (HDFS & YARN)** and **Apache Mahout 0.13.0**. The pipeline ingests, transforms, analyzes, and models **8,187,011** customer reviews (`8,184,798` vectorized text documents) from the UCSD Amazon Reviews 2023 dataset (`Arts_Crafts_and_Sewing`) on a memory-constrained single-node cluster (**8 GB RAM / 8 vCores**).

---

## Table of Contents
1. [System Architecture & Pipeline Workflow](#1-system-architecture--pipeline-workflow)
2. [Environment & YARN Memory Tuning](#2-environment--yarn-memory-tuning)
3. [Repository Structure](#3-repository-structure)
4. [Step-by-Step Execution Procedure & Commands](#4-step-by-step-execution-procedure--commands)
   - [Phase 1: Data Ingestion & HDFS Staging](#phase-1-data-ingestion--hdfs-staging)
   - [Phase 2: Compilation & Packaging](#phase-2-compilation--packaging)
   - [Phase 3: Distributed MapReduce Analytics](#phase-3-distributed-mapreduce-analytics)
   - [Phase 4: Mahout SequenceFile & TF-IDF Vectorization](#phase-4-mahout-sequencefile--tf-idf-vectorization)
   - [Phase 5: Complementary Naive Bayes Sentiment Classification](#phase-5-complementary-naive-bayes-sentiment-classification)
   - [Phase 6: Cosine-Distance K-Means Topic Clustering](#phase-6-cosine-distance-kmeans-topic-clustering)
5. [Experimental Results & Evaluation Metrics](#5-experimental-results--evaluation-metrics)
6. [Engineering Challenges & Resolutions](#6-engineering-challenges--resolutions)

---

## 1. System Architecture & Pipeline Workflow

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
|                                   |   |    (trainnb / testnb -> 80.98% Accuracy)  |
|                                   |   |                                           |
|                                   |   |  - Cosine-Distance K-Means (k = 5)        |
|                                   |   |    (kmeans + clusterdump -> 5 Topics)     |
+-----------------------------------+   +-------------------------------------------+
2. Environment & YARN Memory TuningSoftware & Hardware StackComponentSpecificationOS & HardwareLinux (HP 15s Workstation), 8 GB Physical RAM, 8 vCoresJava RuntimeOpenJDK 11Distributed Storage & ComputeApache Hadoop 3.3.6 (Pseudo-Distributed HDFS + YARN)Distributed Machine LearningApache Mahout 0.13.0DatasetUCSD Amazon Reviews 2023 — Arts_Crafts_and_Sewing (8,187,011 records)Memory Optimization Strategy (Preventing YARN Container OOM Kills)Processing a 5.6 GB sparse TF-IDF matrix across 18 HDFS input splits on an 8 GB node requires strict container sizing and spill tuning so ApplicationMaster, Map, and Reduce containers fit concurrently within physical RAM without triggering Linux OOM killer or YARN virtual memory limits:Bashexport HADOOP_HEAPSIZE=2048

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
Reduced Shuffle Buffers (0.20 / 0.35): Prevents heap exhaustion during high-fan-in reduce phases in seq2sparse and trainnb.Bounded Sort Buffer (128 MB): Leaves sufficient off-heap headroom for native Zlib decompression and Netty shuffle threads.3. Repository StructurePlaintextamazon_2023_analytics/
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
├── .gitignore                                # Excludes multi-GB datasets, JARs, and class files
└── README.md                                 # Technical documentation & execution guide
4. Step-by-Step Execution Procedure & CommandsPhase 1: Data Ingestion & HDFS StagingStart HDFS and YARN daemons and verify active nodes:Bashstart-dfs.sh && start-yarn.sh
jps
Stream-convert the raw .jsonl.gz archive into a clean, tab/comma-delimited CSV (parent_asin, rating, review_text) and upload it to HDFS (1.7 GB):BashHDFS_BASE="/user/student278/venkat/amazon_2023_arts"
hdfs dfs -mkdir -p ${HDFS_BASE}/input
hdfs dfs -put raw_data/arts_reviews_clean.csv ${HDFS_BASE}/input/
hdfs dfs -du -h ${HDFS_BASE}/input
Phase 2: Compilation & PackagingCompile all custom Hadoop MapReduce drivers and package them into amazon-2023-analytics.jar:Bashcd ~/amazon_2023_analytics
mkdir -p build
javac -cp "$(hadoop classpath)" -d build src/*.java
jar -cvf amazon-2023-analytics.jar -C build/ .
Phase 3: Distributed MapReduce AnalyticsExecute the three MapReduce analytics jobs on HDFS:Product Rating Aggregation (out_rating): Uses a local Combiner (sum_ratings, count) to minimize network shuffle before computing per-ASIN mean ratings (31.5 MB output).Category Rating Distribution (out_distribution): Aggregates global review volume per star tier (1_Star through 5_Star).Two-Stage Distributed Top-10 Products (out_topk):Stage 1: Counts total reviews per parent_asin across all mappers/reducers (9.6 MB).Stage 2: Each mapper maintains a local bounded TreeMap of its top 10 ASINs, and a single reducer merges local top-10 candidates into the global Top-10 ranking.BashHDFS_BASE="/user/student278/venkat/amazon_2023_arts"

# Run MapReduce Analytics Jobs
hadoop jar amazon-2023-analytics.jar CsvReviewAnalytics \
  ${HDFS_BASE}/input \
  ${HDFS_BASE}/out_rating \
  ${HDFS_BASE}/out_distribution \
  ${HDFS_BASE}/out_topk_stage1 \
  ${HDFS_BASE}/out_topk

# Export merged summaries to local reports/
mkdir -p reports
hdfs dfs -getmerge ${HDFS_BASE}/out_distribution reports/category_rating_distribution.txt
hdfs dfs -getmerge ${HDFS_BASE}/out_topk reports/top10_reviewed_products.txt
hdfs dfs -getmerge ${HDFS_BASE}/out_rating reports/product_ratings_summary.txt
head -n 50 reports/product_ratings_summary.txt > reports/sample_product_ratings_top50.txt
Phase 4: Mahout SequenceFile & TF-IDF VectorizationConvert CSV records into Hadoop SequenceFile<Text, Text> (1.6 GB on HDFS), encoding sentiment labels directly into Mahout-compliant hierarchical keys (/Positive/REV23_..., /Neutral/REV23_..., /Negative/REV23_...):Positive: 4–5 StarsNeutral: 3 StarsNegative: 1–2 StarsBashMAHOUT_WORK="${HDFS_BASE}/mahout_work"
hadoop jar amazon-2023-analytics.jar CsvToMahoutSeqFile \
  ${HDFS_BASE}/input \
  ${MAHOUT_WORK}/seqfiles
Generate L2-normalized TF-IDF sparse vectors (5.6 GB across 18 splits, filtering rare terms via --minDF 5 and capping high-frequency noise with --maxDFPercent 85):Bashexport MAHOUT_HOME=~/apache-mahout-distribution-0.13.0
export PATH=$PATH:$MAHOUT_HOME/bin

mahout seq2sparse $MEM_OPTS \
  -i ${MAHOUT_WORK}/seqfiles \
  -o ${MAHOUT_WORK}/vectors \
  -lnorm -nv -wt tfidf \
  --minDF 5 --maxDFPercent 85 \
  --numReducers 4
Phase 5: Complementary Naive Bayes Sentiment ClassificationTrain a Complementary Naive Bayes (-c) classifier (designed to mitigate class imbalance in skewed review datasets) and evaluate classification performance across all 8,184,798 vectorized reviews:Bashmahout trainnb $MEM_OPTS \
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
Phase 6: Cosine-Distance K-Means Topic ClusteringRun unsupervised K-Means clustering (k = 5, CosineDistanceMeasure, convergence delta 0.05) on the TF-IDF vectors and extract the top 12 weighted vocabulary terms per cluster using clusterdump:Bashmahout kmeans $MEM_OPTS \
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
5. Experimental Results & Evaluation MetricsA. Category-Wide Star Rating Distribution (8,187,011 Reviews)The dataset exhibits a strong positive skew typical of e-commerce craft supplies, with 77.40% of reviews rated 4 or 5 stars and 10.47% rated 1 star.Rating TierReview CountPercentage ShareSentiment Mapping5 Star5,462,20566.72%Positive4 Star874,24510.68%Positive3 Star585,2177.15%Neutral2 Star407,9674.98%Negative1 Star857,37710.47%NegativeTotal8,187,011100.00%—B. Top-10 Most Frequently Reviewed ProductsRankParent ASINReview CountRankParent ASINReview Count1B01LY9ERQC7,8626B001PNIWT25,4042B0BPRQR2DH6,5537B09XYL6FS75,3893B08KVH58MS6,1138B0C7WDNTMV5,2694B0757JLTY76,0309B09JHWHYYV5,2655B07LCTD7C75,44210B0C2RW2TXH4,908C. Complementary Naive Bayes Evaluation (8,184,798 Vectorized Reviews)Correctly Classified Instances: 6,628,277 (80.9828%)Incorrectly Classified Instances: 1,556,521 (19.0172%)Weighted Precision: 0.7867Weighted Recall: 0.8098Weighted F1-Score: 0.7512Confusion MatrixActual \ Predicteda (Negative)b (Neutral)c (Positive)Total Actuala = Negative322,430277942,2541,264,961b = Neutral36,073961548,088585,122c = Positive28,9279026,304,8866,334,715D. Discovered K-Means Topic Clusters (k = 5, Cosine Distance)Cluster IDTop Weighted TermsInterpreted Customer ThemeVL-5186261very (0.73), good (0.68), nice (0.48), easy (0.46), quality (0.43), perfect (0.32)Ease of Use & Build Quality — usability and product finishVL-2033205great (2.25), making (0.45), product (0.45), price (0.44), works (0.38), quality (0.38)Value for Money & Utility — price-to-performance in craftingVL-5877508just (1.05), what (1.02), have (0.88), like (0.64), use (0.59), would (0.57)Exact Fit & Expectations — items matching project specificationsVL-7151045love (1.68), used (0.76), make (0.65), paint (0.61), colors (0.43), gift (0.40)Painting, Colors & Gifting — art mediums and handmade giftsVL-7067819were (1.15), more (1.01), than (0.90), little (0.83), one (0.61), had (0.55)Quantity, Sizing & Comparison — pack counts and dimensions6. Engineering Challenges & ResolutionsYARN Container Memory Exhaustion during seq2sparse (27.05 min):Cause: Default reducer shuffle buffers (0.70) exceeded physical RAM on an 8 GB node when sorting 5.6 GB of TF-IDF partial vectors.Fix: Tuned mapreduce.reduce.shuffle.input.buffer.percent=0.20, merge.percent=0.35, and capped map/reduce container limits (1536 MB / 3072 MB).Mahout IndexInstancesMapper Label Extraction (SKIPPED_INSTANCES):Cause: Mahout's trainnb parses class labels using leading slash delimiters (/Label/DocID) and skips instances when split -xm mapreduce alters key metadata via MultipleOutputs.Fix: Encoded /Positive/, /Neutral/, and /Negative/ prefixes directly during SequenceFile generation (CsvToMahoutSeqFile.java) and fed tfidf-vectors directly into trainnb and testnb with explicit HDFS --tempDir paths.
