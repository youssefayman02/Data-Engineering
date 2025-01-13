# Fintech Data Engineering Project

## Overview
This project involves building a comprehensive data processing and analysis pipeline for fintech data. The tasks are divided into four milestones, covering data cleaning, feature engineering, real-time processing, and visualization using ETL pipelines and dashboards.

---

## Milestone 1: Data Ingestion and Initial Cleaning

### Objectives
1. **Data Ingestion**  
   - Load raw fintech data from CSV files.  
   - Preview and understand data structure.

2. **EDA**
    - Perform exploratory data analysis to understand the data.
    - Ask at least 5 questions and visualize the answer to these questions

3. **Data Cleaning**  
   - Remove duplicates.  
   - Handle incorrect data types.  
   - Observe and handle missing values.
   - Observe and handle outliers

4. **Data Transformation and Feature Engineering**
    - Adding new features
    - Encoding categorical columns
    - Applying normalization techniques

5. **Data Storage**  
   - Save the cleaned dataset in CSV/Parquet format for further processing.

### Deliverables
- Notebook: `M1 MAJOR GroupNo ID.ipynb`  
- Cleaned Data: `fintech_data_{MAJOR}_{GROUP}_{ID}_clean.csv/parquet`  
- Lookup Table: `lookup_table_{MAJOR}_{GROUP}_{ID}_clean.csv/parquet`  

---

## Milestone 2: Real-time Data Streaming with Kafka

### Objectives
1. **Kafka Integration**  
   - Set up Kafka producers and consumers for streaming data.

2. **Streaming Data Processing**  
   - Process incoming data in real-time using Python.

3. **Data Storage**  
   - Store processed data in PostgreSQL for analysis.

### Deliverables
- Kafka Producer/Consumer Scripts  
- PostgreSQL Database with Streamed Data  

---

## Milestone 3: Data Cleaning and Feature Engineering

### Objectives
1. **Loading the Dataset (5%)**  
   - Load the provided Parquet dataset.  
   - Preview the first 20 rows.  
   - Adjust partitions to match the number of logical cores.

2. **Data Cleaning (30%)**  
   - **Column Renaming (10%)**: Replace spaces with underscores and convert to lowercase.  
   - **Detect Missing Values (35%)**: Identify and report missing values.  
   - **Handle Missing Values (35%)**: Replace missing numerical values with 0 and categorical values with mode.  
   - **Verify Cleaning (20%)**: Confirm no missing values remain.

3. **Feature Engineering (15%)**  
   - Add features for previous loan amounts and dates by grade and state.

4. **Categorical Encoding (10%)**  
   - Encode specified categorical columns using label and one-hot encoding.

5. **Lookup Table (5%)**  
   - Create and save a lookup table for encodings.

6. **Saving Outputs (5%)**  
   - Save the cleaned dataset and lookup table.

7. **Bonus (5%)**  
   - Load cleaned data into PostgreSQL with PGAdmin screenshots.

### Deliverables
- Notebook: `m3_spark_52_XXXX.ipynb`  
- Cleaned Data: `fintech_spark_52_XXXX_clean.parquet`  
- Lookup Table: `lookup_spark_52_XXXX.parquet`  
- (Bonus) PGAdmin Screenshots

---

## Milestone 4: ETL Pipeline and Dashboard with Airflow

### Objectives
1. **ETL Pipeline with Airflow**  
   - **extract_clean**: Clean raw data.  
   - **transform**: Apply data transformations.  
   - **load_to_db**: Load data into PostgreSQL.

2. **Interactive Dashboard**  
   - Visualize key insights using Plotly Dash/Streamlit.

3. **Airflow DAG**  
   - Define ETL workflow with `extract_clean` → `transform` → `load_to_db` → `run_dashboard`.

4. **Submission Video**  
   - Record a 5–10 minute demo of Airflow execution and dashboard.

5. **Bonus (5%)**  
   - Exceptional dashboard UI/UX.

### Deliverables
- `dags/`  
  - `fintech_dag.py`  
  - `functions.py`  
  - `fintech_dashboard.py`  
- Video: `fintech_dashboard_showcase_{your_id}.mp4`

---

## Tools & Technologies
- **Python**
- **Pandas**
- **Matplotlib**
- **scikit-learn**
- **SQLAlchemy**
- **Apache Spark (PySpark)**  
- **Apache Kafka**  
- **Apache Airflow**  
- **PostgreSQL**  
- **Plotly Dash**  
- **Docker**