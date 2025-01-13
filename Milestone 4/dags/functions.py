import pandas as pd

from preprocessing import initialize_tables,handling_inconsistent_data, imputation, handling_outliers, feature_engineering, encoding, scaling
from db import save_to_db


def extract_clean(filename):
    """
    Extracts and cleans data from a CSV file.
    This function reads data from the specified CSV file, handles inconsistent data, 
    performs imputation, and saves the cleaned data along with various tables to CSV files.
    Args:
        filename (str): The path to the CSV file to be read.
    Returns:
        None
    Raises:
        FileExistsError: If any of the output CSV files already exist.
    The function performs the following steps:
    1. Reads the CSV file into a DataFrame.
    2. Initializes lookup, imputed mode, imputed mean, IQR threshold, one-hot categories, and scaler tables.
    3. Handles inconsistent data in the DataFrame.
    4. Performs imputation on the DataFrame and updates the lookup, imputed mode, and imputed mean tables.
    5. Prints various statistics and null value counts from the DataFrame.
    6. Attempts to save the cleaned DataFrame and tables to CSV files in the specified directory.
    """

    print("Extracting and cleaning data...")

    df = pd.read_csv(filename)

    lookup_table, imputed_mode_table, imputed_mean_table, iqr_threshold_table, one_hot_categories_table, scaler_table = initialize_tables()
    df = handling_inconsistent_data(df)
    df, lookup_table, imputed_mode_table, imputed_mean_table = imputation(df, lookup_table, imputed_mode_table, imputed_mean_table, False)

    try: 
        df.to_csv('/opt/airflow/data/fintech_clean.csv', index=False, mode='x')
        lookup_table.to_csv('/opt/airflow/data/lookup_table.csv', index=False, mode='x')
        imputed_mode_table.to_csv('/opt/airflow/data/imputed_mode_table.csv', index=False, mode='x')
        imputed_mean_table.to_csv('/opt/airflow/data/imputed_mean_table.csv', index=False, mode='x')
        iqr_threshold_table.to_csv('/opt/airflow/data/iqr_threshold_table.csv', index=False, mode='x')
        one_hot_categories_table.to_csv('/opt/airflow/data/one_hot_categories_table.csv', index=False, mode='x')
        scaler_table.to_csv('/opt/airflow/data/scaler_table.csv', index=False, mode='x')
    except FileExistsError:
        print("File already exists")

    print("Data extraction and cleaning complete")

def transform(filename):
    """
    Transforms the input CSV file by performing various data preprocessing steps including handling outliers, 
    feature engineering, encoding, and scaling. The transformed data is then saved to a new CSV file.
    Args:
        filename (str): The path to the input CSV file.
    Returns:
        None
    Raises:
        FileExistsError: If the output file already exists.
    The function performs the following steps:
        1. Reads the input CSV file into a DataFrame.
        2. Reads lookup tables and other necessary CSV files for transformation.
        3. Handles outliers in the data.
        4. Performs feature engineering on the data.
        5. Encodes categorical variables.
        6. Scales numerical features.
        7. Prints various statistics and null value counts of the transformed DataFrame.
        8. Saves the transformed DataFrame to a new CSV file.
        9. Updates and saves the lookup tables and other necessary CSV files.
    """

    print("Transforming data...")

    df = pd.read_csv(filename)
    lookup_table = pd.read_csv('/opt/airflow/data/lookup_table.csv')
    iqr_threshold_table = pd.read_csv('/opt/airflow/data/iqr_threshold_table.csv')
    one_hot_categories_table = pd.read_csv('/opt/airflow/data/one_hot_categories_table.csv')
    scaler_table = pd.read_csv('/opt/airflow/data/scaler_table.csv')

    df, iqr_threshold_table = handling_outliers(df, lookup_table, iqr_threshold_table, False)
    df = feature_engineering(df, lookup_table)
    df, lookup_table, one_hot_categories_table = encoding(df, lookup_table, one_hot_categories_table, False)
    df, scaler_table = scaling(df, scaler_table, False)

    print(df)
    print(df.annual_inc_joint.value_counts())
    print(df.emp_title.value_counts())
    print(df.emp_length.value_counts())
    print(df.int_rate.value_counts())
    print(df.description.value_counts())
    print(df.isnull().sum())

    try:
        df.to_csv('/opt/airflow/data/fintech_transformed.csv', index=False, mode='x')
    except FileExistsError:
        print("File already exists")

    try:
        lookup_table.to_csv('/opt/airflow/data/lookup_table.csv', index=False, mode='w')
        iqr_threshold_table.to_csv('/opt/airflow/data/iqr_threshold_table.csv', index=False, mode='w')
        one_hot_categories_table.to_csv('/opt/airflow/data/one_hot_categories_table.csv', index=False, mode='w')
        scaler_table.to_csv('/opt/airflow/data/scaler_table.csv', index=False, mode='w')
    except FileExistsError:
        print("File can not be written")

    print("Data transformation complete")

def load_to_postgres(filename):
    """
    Loads data from a CSV file into a PostgreSQL database after performing necessary transformations.
    Args:
        filename (str): The path to the CSV file to be loaded.
    Reads the following CSV files for transformation purposes:
        - /opt/airflow/data/lookup_table.csv
        - /opt/airflow/data/imputed_mode_table.csv
        - /opt/airflow/data/imputed_mean_table.csv
        - /opt/airflow/data/iqr_threshold_table.csv
        - /opt/airflow/data/one_hot_categories_table.csv
        - /opt/airflow/data/scaler_table.csv
    Calls the `save_to_db` function to save the transformed data into the database.
    Returns:
        None
    """
    print("Loading data to PostgreSQL...")

    df = pd.read_csv(filename)
    lookup_table = pd.read_csv('/opt/airflow/data/lookup_table.csv')
    imputed_mode_table = pd.read_csv('/opt/airflow/data/imputed_mode_table.csv')
    imputed_mean_table = pd.read_csv('/opt/airflow/data/imputed_mean_table.csv')
    iqr_threshold_table = pd.read_csv('/opt/airflow/data/iqr_threshold_table.csv')
    one_hot_categories_table = pd.read_csv('/opt/airflow/data/one_hot_categories_table.csv')
    scaler_table = pd.read_csv('/opt/airflow/data/scaler_table.csv')

    save_to_db(df, lookup_table, imputed_mode_table, imputed_mean_table, iqr_threshold_table, one_hot_categories_table, scaler_table)

    print("Data loaded to PostgreSQL")



