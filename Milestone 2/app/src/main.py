import pandas as pd
from cleaning import clean
from db import save_to_db
import os
from run_producer import start_producer, stop_container
from consumer import consume_data
import time

print('> Current working directory:', os.getcwd())

data_path = 'data/fintech_data_22_52_14669.csv'

def load_data(data_path):
    # Load the data
    data = pd.read_csv(data_path)
    return data

def save_data(df: pd.DataFrame, path: str, mode='w', index=False):
    print(f'Saving data to {path} with mode={mode}')
    df.to_csv(path, index=index, mode=mode, header=not os.path.exists(path) or mode == 'w')

if __name__ == '__main__':

    if not os.path.exists('data/fintech_data_22_52_14669_clean.csv') or not os.path.exists('data/lookup_table.csv'):

        print('Initializing tables...')
        lookup_table = pd.DataFrame(columns=['Column', 'Original', 'Imputed'])
        imputed_mode_table = pd.DataFrame(columns=['Column', 'Group', 'Imputed_Mode'])
        imputed_mean_table = pd.DataFrame(columns=['Column', 'Group', 'Imputed_Mean'])
        iqr_threshold_table = pd.DataFrame(columns=['Column', 'Lower_Threshold', 'Upper_Threshold'])
        one_hot_categories_table = pd.DataFrame(columns=['Column', 'Category', 'Encoded_Column'])
        scaler_table = pd.DataFrame(columns=['Column', 'Min', 'Max'])


        print('Cleaning data...')
        df = load_data(data_path)
        print(df.isnull().sum())
        df, lookup_table, imputed_mode_table, imputed_mean_table, iqr_threshold_table, one_hot_categories_table, scaler_table = clean(df, lookup_table, imputed_mode_table, imputed_mean_table, iqr_threshold_table, one_hot_categories_table, scaler_table, False)
        
        print(df)
        print(df.annual_inc_joint.value_counts())
        print(df.emp_title.value_counts())
        print(df.emp_length.value_counts())
        print(df.int_rate.value_counts())
        print(df.description.value_counts())
        print(df.isnull().sum())

        save_data(df, 'data/fintech_data_22_52_14669_clean.csv', mode='w', index=True)
        save_data(lookup_table, 'data/lookup_table.csv')
        save_data(imputed_mode_table, 'data/imputed_mode_table.csv')
        save_data(imputed_mean_table, 'data/imputed_mean_table.csv')
        save_data(iqr_threshold_table, 'data/iqr_threshold_table.csv')
        save_data(one_hot_categories_table, 'data/one_hot_categories_table.csv')
        save_data(scaler_table, 'data/scaler_table.csv')
        print('Data cleaned and saved')

        print('Saving data to database...')
        save_to_db(df, lookup_table, imputed_mode_table, imputed_mean_table, iqr_threshold_table, one_hot_categories_table, scaler_table)
    else:
        df = pd.read_csv('data/fintech_data_22_52_14669_clean.csv')
        lookup_table = pd.read_csv('data/lookup_table.csv')
        imputed_mode_table = pd.read_csv('data/imputed_mode_table.csv')
        imputed_mean_table = pd.read_csv('data/imputed_mean_table.csv')
        iqr_threshold_table = pd.read_csv('data/iqr_threshold_table.csv')
        one_hot_categories_table = pd.read_csv('data/one_hot_categories_table.csv')
        scaler_table = pd.read_csv('data/scaler_table.csv')

        print(df)
        print(df.annual_inc_joint.value_counts())
        print(df.emp_title.value_counts())
        print(df.emp_length.value_counts())
        print(df.int_rate.value_counts())
        print(df.description.value_counts())
        print(df.isnull().sum())

        print('Data already cleaned and saved')

        print('Saving data to database...')
        save_to_db(df, lookup_table, imputed_mode_table, imputed_mean_table, iqr_threshold_table, one_hot_categories_table, scaler_table)

    time.sleep(5)

    print('Starting producer...')
    producer_id = start_producer('52_14669', kafka_url='localhost:9092', topic_name='fintech')

    time.sleep(5)

    print('Consuming data...')
    consume_data('fintech', 'kafka:29092')

    print('Stopping producer...')
    stop_container(producer_id)

    print('Done!')
