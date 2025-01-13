from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('postgresql://root:root@pgdatabase:5432/fintech_etl')

def save_to_db(cleaned: pd.DataFrame, lookup_table: pd.DataFrame, imputed_mode_table: pd.DataFrame, imputed_mean_table: pd.DataFrame, iqr_threshold_table: pd.DataFrame, one_hot_categories_table: pd.DataFrame, scaler_table: pd.DataFrame):
    if(engine.connect()):
        print('Connected to Database')
        try:
            print('Writing cleaned dataset to database')
            cleaned.to_sql('fintech_data_MET_1_52_14669_clean', con=engine, if_exists='fail')

            print('Writing lookup table to database')
            lookup_table.to_sql('lookup_fintech_data_MET_1_52_14669', con=engine, if_exists='fail')

            print('Writing imputed mode table to database')
            imputed_mode_table.to_sql('imputed_mode_fintech_data_MET_1_52_14669', con=engine, if_exists='fail')

            print('Writing imputed mean table to database')
            imputed_mean_table.to_sql('imputed_mean_fintech_data_MET_1_52_14669', con=engine, if_exists='fail')

            print('Writing iqr threshold table to database')
            iqr_threshold_table.to_sql('iqr_threshold_fintech_data_MET_1_52_14669', con=engine, if_exists='fail')

            print('Writing one hot categories table to database')
            one_hot_categories_table.to_sql('one_hot_categories_fintech_data_MET_1_52_14669', con=engine, if_exists='fail')

            print('Writing scaler table to database')
            scaler_table.to_sql('scaler_fintech_data_MET_1_52_14669', con=engine, if_exists='fail')

            print('Done writing to database')
        except ValueError as vx:
            print('Cleaned Table already exists.')
        except Exception as ex:
            print(ex)
    else:
        print('Failed to connect to Database')