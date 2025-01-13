# consumer.py
from kafka import KafkaConsumer
import json
import pandas as pd
import os
from cleaning import clean
from db import append_to_db

COLUMNS=[
    "Customer Id", "Emp Title", "Emp Length", "Home Ownership", "Annual Inc",
    "Annual Inc Joint", "Verification Status", "Zip Code", "Addr State",
    "Avg Cur Bal", "Tot Cur Bal", "Loan Id", "Loan Status", "Loan Amount",
    "State", "Funded Amount", "Term", "Int Rate", "Grade", "Issue Date",
    "Pymnt Plan", "Type", "Purpose", "Description"
]

def initialize_df():
    # Define the schema of the dataframe
    return pd.DataFrame(columns=COLUMNS)

def process_message(df):
    print("Processing data...")
    return df

def save_data(df: pd.DataFrame, path: str, mode='w', index=False):
    print(f'Saving data to {path} with mode={mode}')
    df.to_csv(path, index=index, mode=mode, header=not os.path.exists(path) or mode == 'w')


def consume_data(topic, kafka_url):
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=[kafka_url],
        auto_offset_reset='earliest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    print(f"Listening to topic '{topic}' on {kafka_url}...")
    
    flag = False
    # df = initialize_df()

    imputed_mean_table = pd.read_csv('data/imputed_mean_table.csv')
    imputed_mode_table = pd.read_csv('data/imputed_mode_table.csv')
    lookup_table = pd.read_csv('data/lookup_table.csv')
    iqr_threshold_table = pd.read_csv('data/iqr_threshold_table.csv')
    one_hot_categories_table = pd.read_csv('data/one_hot_categories_table.csv')
    scaler_table = pd.read_csv('data/scaler_table.csv')



    while True:
        message = consumer.poll(timeout_ms=2000)
        print('Polling...')

        if flag:
            break

        if message:
            for _, messages in message.items():
                if flag:
                    break

                for msg in messages:
                    print(msg)
                    if msg.value == 'EOF':
                        flag = True
                        break
                    else:
                        new_row = pd.DataFrame([msg.value], columns=COLUMNS)
                        new_row, lookup_table, imputed_mode_table, imputed_mean_table, iqr_threshold_table, one_hot_categories_table, scaler_table = clean(new_row, lookup_table, imputed_mode_table, imputed_mean_table, iqr_threshold_table, one_hot_categories_table, scaler_table, True)
                        print(new_row)
                        save_data(new_row, 'data/fintech_data_22_52_14669_clean.csv', mode='a', index=True)
                        append_to_db(new_row)
    
    consumer.close()


