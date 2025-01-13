import pandas as pd
import numpy as np
import requests

# Data visualization
import matplotlib.pyplot as plt
from matplotlib import rcParams

from bs4 import BeautifulSoup

from sklearn.preprocessing import MinMaxScaler

# Cool plotting style
plt.style.use('ggplot')
rcParams['figure.figsize'] = 12, 6

# lookup_table = pd.DataFrame(columns=['Column', 'Original', 'Imputed'])
# imputed_mode_table = pd.DataFrame(columns=['Column', 'Group', 'Imputed_Mode'])
# imputed_mean_table = pd.DataFrame(columns=['Column', 'Group', 'Imputed_Mean'])
# iqr_threshold_table = pd.DataFrame(columns=['Column', 'Lower_Threshold', 'Upper_Threshold'])


def clean_column_name(column_name: str):
    """
    Cleans and formats a column name string.
    This function converts the input string to lowercase, strips leading and trailing whitespace,
    replaces spaces with underscores, and removes any characters that are not alphanumeric or underscores.
    Args:
        column_name (str): The original column name to be cleaned.
    Returns:
        str: The cleaned and formatted column name.
    """
    formatted_name = column_name.lower()
    formatted_name = formatted_name.strip()
    formatted_name = formatted_name.replace(' ', '_')
    formatted_name = ''.join(e for e in formatted_name if e.isalnum() or e == '_')

    return formatted_name

def clean_column_names(df: pd.DataFrame):
    """
    Cleans the column names of a DataFrame by applying the clean_column_name function to each column name.

    Args:
        df (pd.DataFrame): The DataFrame whose column names need to be cleaned.

    Returns:
        pd.DataFrame: The DataFrame with cleaned column names.
    """
    df.columns = [clean_column_name(column) for column in df.columns]
    return df

def set_index(df: pd.DataFrame, column_name: str):
    """
    Set the specified column as the index of the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    column_name (str): The name of the column to set as the index.

    Returns:
    pd.DataFrame: The DataFrame with the specified column set as the index.
    """
    df.set_index(column_name, inplace=True)
    return df

def drop_duplicates(df: pd.DataFrame):
    """
    Remove duplicate rows from a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame from which to drop duplicate rows.

    Returns:
    pd.DataFrame: The DataFrame with duplicate rows removed.
    """
    df.drop_duplicates(inplace=True)
    return df

ExpectedDataTypes = {
    "customer_id": "object",
    "emp_title": "object",
    "emp_length": "object",
    "home_ownership": "object",
    "annual_inc": "float64",
    "annual_inc_joint": "float64",
    "verification_status": "object",
    "zip_code": "object",
    "addr_state": "object",
    "avg_cur_bal": "float64",
    "tot_cur_bal": "float64",
    "loan_id": "int64",
    "loan_status": "object",
    "loan_amount": "float64",
    "state": "object",
    "funded_amount": "float64",
    "term": "object", 
    "int_rate": "float64",
    "grade": "int64",
    "issue_date": "object",
    "pymnt_plan": "bool",
    "type": "object",
    "purpose": "object",
    "description": "object"
}

def parse_column_data_types(df: pd.DataFrame, expected_data_types=ExpectedDataTypes):
    """
    Converts the columns of a DataFrame to the specified data types.

    Parameters:
    df (pd.DataFrame): The DataFrame whose columns need to be converted.
    expected_data_types (dict): A dictionary where keys are column names and values are the expected data types.

    Returns:
    pd.DataFrame: The DataFrame with columns converted to the specified data types.

    Raises:
    ValueError: If a column cannot be converted to the specified data type.

    Prints:
    Success message for each column successfully converted.
    Error message for each column that could not be converted.
    Warning message for each column specified in expected_data_types but not found in the DataFrame.
    """
    for column, expected_type in expected_data_types.items():
        if column in df.columns:
            try:
                # Convert column to the expected data type
                df[column] = df[column].astype(expected_type)
                print(f'Success: Column {column} converted to {expected_type}')
            except ValueError as e:
                print(f'Error: Could not convert column {column} to {expected_type}. Error: {e}')
        else:
            print(f'Warning: Column {column} not found in DataFrame')
    return df

def standardize_values_proper_case(df: pd.DataFrame, columns: list):
    """
    Standardizes the values in the specified columns of a DataFrame to proper case.
    Proper case means that the first letter of each word is capitalized and the rest are in lowercase.
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to be standardized.
    columns (list): A list of column names in the DataFrame whose values need to be standardized.
    Returns:
    pd.DataFrame: The DataFrame with the specified columns' values standardized to proper case.
    """
    for column in columns:
        # if value is null or nan skip
        df[column] = df[column].apply(lambda x: ' '.join([word.capitalize() for word in str(x).split()]) if pd.notnull(x) else x)
    
    return df

def change_column_values_to_mapped_values(df: pd.DataFrame, column: str, mapping_dict: dict):
    """
    This function replaces values in a specified column of a DataFrame according to a mapping dictionary.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The name of the column in the DataFrame whose values are to be replaced.
    mapping_dict (dict): A dictionary where keys are the original values in the specified column,
                         and values are the new values to replace the originals.

    Returns:
    pd.DataFrame: The modified DataFrame with the specified column's values changed according to the mapping dictionary.
    """
    df[column] = df[column].map(mapping_dict)
    return df

type_column_map = {
    "Individual": "Individual",
    "Joint": "Joint",
    "Joint App": "Joint",
    "Direct_pay": "Direct Pay"
}

# def check_for_non_standard_missing_values(df: pd.DataFrame, columns: list):
#     """
#     Checks for non-standard missing values in the specified columns of a dataframe, replaces them with NaN,
#     prints the found values and their counts, and returns the modified DataFrame.

#     Args:
#     df (pd.DataFrame): The dataframe to check and replace non-standard missing values.
#     columns (list): List of column names to check for non-standard missing values.

#     Returns:
#     pd.DataFrame: DataFrame with non-standard missing values replaced by NaN.
#     """
    
#     non_standard_missing_values = ["na", "n/a", "missing", "none", "nan", "null", "nil"]
#     missing_values_dict = {}
    
#     for column in columns:
#         found_values = set()
#         count = 0
        
#         for value in non_standard_missing_values:
#             matches = df[df[column].astype(str).str.lower() == value]
#             found_values.update(matches[column].unique())
#             count += len(matches)
        
        
#         if found_values:
#             missing_values_dict[column] = {'values': list(found_values), 'count': count}
#             df[column] = df[column].replace(non_standard_missing_values, np.nan, regex=True)
    
#     print(missing_values_dict)
    
#     return df

def check_for_non_standard_missing_values(df: pd.DataFrame, columns: list):
    """
    Checks for non-standard missing values in the specified columns of a dataframe, replaces them with NaN
    for exact matches only, prints the found values and their counts, and returns the modified DataFrame.

    Args:
    df (pd.DataFrame): The dataframe to check and replace non-standard missing values.
    columns (list): List of column names to check for non-standard missing values.

    Returns:
    pd.DataFrame: DataFrame with non-standard missing values replaced by NaN.
    """
    
    non_standard_missing_values = ["na", "n/a", "missing", "none", "nan", "null", "nil"]
    missing_values_dict = {}

    for column in columns:
        if df[column].dtype == 'object':
            found_values = set()
            count = 0
            
            for value in non_standard_missing_values:
                matches = df[df[column].astype(str).str.lower() == value]
                found_values.update(matches[column].unique())
                count += len(matches)
                
            if found_values:
                missing_values_dict[column] = {'values': list(found_values), 'count': count}
                df.loc[df[column].astype(str).str.lower().isin(non_standard_missing_values), column] = np.nan
    
    print("Non-standard missing values found and replaced:")
    print(missing_values_dict)
    
    return df


########## Handling Missing Values ##########
def impute_with_value(df: pd.DataFrame, column: str, value, lookup_table: pd.DataFrame, stream: bool):
    """
    Impute missing values in a specified column with a given value.

    Parameters:
    df (pd.DataFrame): The dataframe containing the column to impute.
    column (str): The name of the column in which to impute missing values.
    value: The value to use for imputation.

    Returns:
    pd.DataFrame: The dataframe with missing values in the specified column imputed.
    """
    df[column].fillna(value, inplace=True)

    # update the lookup table
    if not stream:
        lookup_table.loc[len(lookup_table)] = [column, 'Missing', value]
    
    return df, lookup_table

# def impute_with_group_mode(df: pd.DataFrame, target_column: str, group_by_column: str):
#     """
#     Impute missing values in the target column based on the mode of the target column
#     within groups defined by another column. If there are no modes found in the group,
#     it will use the overall mode of the target column.

#     Parameters:
#     df (pd.DataFrame): The dataframe containing the columns to impute.
#     target_column (str): The name of the column with missing values to impute.
#     group_by_column (str): The column to group by and determine the mode for imputation.

#     Returns:
#     pd.DataFrame: The dataframe with missing values in the target column imputed.
#     """

#     mode_imputer = lambda x: x.mode()[0] if not x.mode().empty else None
#     mode_map = df.groupby(group_by_column)[target_column].apply(mode_imputer).to_dict()

#     df[target_column] = df.apply(
#         lambda row: mode_map.get(row[group_by_column], None) if pd.isna(row[target_column]) else row[target_column],
#         axis=1
#     )
#     overall_mode = df[target_column].mode()[0] if not df[target_column].mode().empty else None
#     df[target_column].fillna(overall_mode, inplace=True)

#     return df

def impute_with_group_mode(df: pd.DataFrame, target_column: str, group_by_column: str, imputed_mode_table: pd.DataFrame, stream: bool):
    """
    Impute missing values in the target column based on the mode within groups defined by another column.
    Falls back to the overall mode if the group mode is missing.

    Parameters:
    df (pd.DataFrame): The dataframe containing the columns to impute.
    target_column (str): The name of the column with missing values to impute.
    group_by_column (str): The column to group by and determine the mode for imputation.
    imputed_mode_table (pd.DataFrame): The global DataFrame to store or retrieve modes.
    stream (bool): If False, calculate and update modes in imputed_mode_table; if True, use imputed_mode_table.

    Returns:
    pd.DataFrame: The dataframe with missing values in the target column imputed.
    pd.DataFrame: Updated imputed_mode_table with modes.
    """
    if not stream:
        mode_map = df.groupby(group_by_column)[target_column].apply(lambda x: x.mode()[0] if not x.mode().empty else None).to_dict()
        overall_mode = df[target_column].mode()[0] if not df[target_column].mode().empty else None

        for group, mode in mode_map.items():
            imputed_mode_table = pd.concat([imputed_mode_table, pd.DataFrame({'Column': [target_column], 'Group': [group], 'Imputed_Mode': [mode]})])

        imputed_mode_table = pd.concat([imputed_mode_table, pd.DataFrame({
            'Column': [target_column],
            'Group': ['Overall'],
            'Imputed_Mode': [overall_mode]
        })])

    mode_map = imputed_mode_table[imputed_mode_table['Column'] == target_column].set_index('Group')['Imputed_Mode'].to_dict()
    overall_mode = mode_map.get('Overall')

    df[target_column] = df.apply(
        lambda row: mode_map.get(row[group_by_column], overall_mode) 
                    if pd.isna(row[target_column]) 
                    else row[target_column],
        axis=1
    )

    df[target_column].fillna(overall_mode, inplace=True)

    return df, imputed_mode_table



# def impute_with_group_mean(df: pd.DataFrame, target_column: str, group_by_column: str):
#     """
#     Impute missing values in the target column based on the mean of the target column
#     within groups defined by another column.

#     Parameters:
#     df (pd.DataFrame): The dataframe containing the columns to impute.
#     target_column (str): The name of the column with missing values to impute.
#     group_by_column (str): The column to group by to calculate the mean for imputation.

#     Returns:
#     pd.DataFrame: The dataframe with missing values in the target column imputed.
#     """
#     # Calculate the mean for the target column grouped by the group_by_column
#     mean_map = df.groupby(group_by_column)[target_column].mean().to_dict()

#     # Fill missing values in the target column with the mean from the corresponding group
#     df[target_column] = df.apply(
#         lambda row: mean_map.get(row[group_by_column]) if pd.isna(row[target_column]) else row[target_column],
#         axis=1
#     )

#     return df

def impute_with_group_mean(df: pd.DataFrame, target_column: str, group_by_column: str, imputed_mean_table:pd.DataFrame, stream: bool):

    if not stream:
        mean_map = df.groupby(group_by_column)[target_column].mean().to_dict()
        
        for group, mean in mean_map.items():
            imputed_mean_table = pd.concat([imputed_mean_table, pd.DataFrame({'Column': [target_column], 'Group': [group], 'Imputed_Mean': [mean]})])

    mean_map = imputed_mean_table[imputed_mean_table['Column'] == target_column].set_index('Group')['Imputed_Mean'].to_dict()

    df[target_column] = df.apply(lambda row: mean_map.get(row[group_by_column]) if pd.isna(row[target_column]) else row[target_column], axis=1)
    return df, imputed_mean_table


def apply_log_transformation(df, column_name, lookup_table: pd.DataFrame):
    """
    Applies log or log1p transformation to a specified column depending on whether
    the column contains zero or negative values. 
    
    Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        column_name (str): The column to log-transform and visualize.
        
    Returns:
        df (pd.DataFrame): The dataframe with the log-transformed column.
    """

    filled = False
    if column_name in lookup_table['Column'].values:
        if lookup_table[lookup_table['Column'] == column_name]['Original'].values[0] == 'Missing':
            df[column_name] = df[column_name].replace(lookup_table[lookup_table['Column'] == column_name]['Imputed'].values[0], np.nan)
            filled = True

    if (df[column_name] <= 0).any():
        print(f"Column '{column_name}' contains non-positive values. Applying log1p transformation (log(1 + x)) to handle zeros.")
        df[column_name] = np.log1p(df[column_name])
    else:
        print(f"Applying log transformation to column '{column_name}'")
        df[column_name] = np.log(df[column_name])

    if filled:
        df[column_name] = df[column_name].replace(np.nan, lookup_table[lookup_table['Column'] == column_name]['Imputed'].values[0])

    return df

def reverse_log_transformation(df, column_name):
    """
    Reverses a log transformation by exponentiating the values in the specified column.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The column on which to reverse the log transformation.
        
    Returns:
        pd.DataFrame: The DataFrame with the specified column transformed back to its original scale.
    """
    
    if (df[column_name] <= np.log1p(df[column_name].dropna())).all():
        df[column_name] = np.expm1(df[column_name])
        print(f"Reversed log1p transformation on '{column_name}' with expm1.")
    else:
        df[column_name] = np.exp(df[column_name])
        print(f"Reversed log transformation on '{column_name}' with exp.")
    
    return df

def apply_sqrt_transformation(df, column_name):
    """
    Applies square root transformation to a specified column
    
    Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        column_name (str): The column to apply the square root transformation and visualize.
        
    Returns:
        df (pd.DataFrame): The dataframe with the square root-transformed column.
    """
    # Ensure that the column contains non-negative values (sqrt requires non-negative values)
    if (df[column_name] < 0).any():
        print(f"Column '{column_name}' contains negative values, which are not suitable for square root transformation.")
        return
    
    # Create a new column for the square root-transformed data
    df[column_name] = np.sqrt(df[column_name])

    return df

def reverse_sqrt_transformation(df, column_name):
    """
    Reverses a square root transformation by squaring the values in the specified column.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The column on which to reverse the square root transformation.
        
    Returns:
        pd.DataFrame: The DataFrame with the specified column transformed back to its original scale.
    """
    
    # Reverse the transformation by squaring the values
    df[column_name] = df[column_name] ** 2
    print(f"Reversed square root transformation on '{column_name}' by squaring the values.")
    
    return df

def apply_reciprocal_transformation(df, column_name):
    """
    Applies reciprocal transformation to a specified column
    
    Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        column_name (str): The column to apply the reciprocal transformation and visualize.
        
    Returns:
        df (pd.DataFrame): The dataframe with the reciprocal-transformed column.
    """
    # Ensure the column does not contain zero (reciprocal of zero is undefined)
    if (df[column_name] == 0).any():
        print(f"Column '{column_name}' contains zero values, which are not suitable for reciprocal transformation.")
        return
    
    # Create a new column for the reciprocal-transformed data
    df[column_name] = 1 / df[column_name]

    return df

# def apply_capping_iqr(df, column_name):
#     """
#     Caps the outliers in a specified column by setting values below the lower percentile to the lower threshold
#     and values above the upper percentile to the upper threshold.
    
#     Parameters:
#         df (pd.DataFrame): The dataframe containing the data.
#         column_name (str): The column to apply capping on.
        
#     Returns:
#         df (pd.DataFrame): The dataframe with the capped column.
#     """

#     Q1, Q3, IQR = df[column_name].quantile(0.25), df[column_name].quantile(0.75), df[column_name].quantile(0.75) - df[column_name].quantile(0.25)
#     lower_threshold, upper_threshold = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    
#     # Cap the values at the specified thresholds
#     df[column_name] = df[column_name].clip(lower=lower_threshold, upper=upper_threshold)

#     return df

def apply_capping_iqr(df: pd.DataFrame, column_name: str, iqr_threshold_table: pd.DataFrame, stream: bool):
    """
    Apply capping to a specified column in a DataFrame based on the Interquartile Range (IQR).
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    column_name (str): The name of the column to apply capping to.
    iqr_threshold_table (pd.DataFrame): A DataFrame containing the IQR thresholds for columns.
    stream (bool): A flag indicating whether to use precomputed thresholds from iqr_threshold_table (True) 
                   or to compute new thresholds (False).
    Returns:
    pd.DataFrame: The DataFrame with the specified column capped based on IQR thresholds.
    pd.DataFrame: The updated iqr_threshold_table with new thresholds if stream is False.
    """

    if not stream:
        Q1, Q3 = df[column_name].quantile(0.25), df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_threshold, upper_threshold = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

        iqr_threshold_table = pd.concat([iqr_threshold_table, pd.DataFrame({'Column': [column_name], 'Lower_Threshold': [lower_threshold], 'Upper_Threshold': [upper_threshold]})])

    thresholds = iqr_threshold_table[iqr_threshold_table['Column'] == column_name]
    lower_threshold = thresholds['Lower_Threshold'].values[0]
    upper_threshold = thresholds['Upper_Threshold'].values[0]

    df[column_name] = df[column_name].clip(lower=lower_threshold, upper=upper_threshold)
    return df, iqr_threshold_table


def apply_capping_threshold(df, column_name, lower_threshold, upper_threshold):
    """
    Caps the outliers in a specified column by setting values below the lower threshold to the lower threshold
    and values above the upper threshold to the upper threshold.
    
    Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        column_name (str): The column to apply capping on.
        lower_threshold (float): The lower threshold for capping.
        upper_threshold (float): The upper threshold for capping.
        
    Returns:
        df (pd.DataFrame): The dataframe with the capped column.
    """
    # Cap the values at the specified thresholds
    df[column_name] = df[column_name].clip(lower=lower_threshold, upper=upper_threshold)

    return df

def add_month_number(df: pd.DataFrame, date_column: str, new_column_name: str = 'month_number'):
    """
    Adds a month number column to the DataFrame based on the specified date column.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the date column.
    date_column (str): The name of the column with date values to convert to datetime and extract month numbers from.
    new_column_name (str): The name for the new column to store month numbers. Default is 'month_number'.

    Returns:
    pd.DataFrame: The updated DataFrame with a new column containing the month number (1-12).
    """
    
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df[new_column_name] = df[date_column].dt.month
    
    return df

def create_salary_can_cover_column(df: pd.DataFrame, income_column: str, loan_column: str):
    """
    Adds a 'salary_can_cover' column to the DataFrame.
    
    This column contains a boolean value (True = 1, False = 0) that indicates
    if the annual income can cover the loan amount.

    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    income_column (str): The name of the column representing annual income.
    loan_column (str): The name of the column representing loan amount.

    Returns:
    pd.DataFrame: The modified DataFrame with a new 'Salary Can Cover' column.
    """
    df['salary_can_cover'] = (df[income_column] >= df[loan_column]).astype(bool)
    return df

def one_encode_grade(df: pd.DataFrame, column: str):
    """
    Encodes the numeric grade column into letter grades (A-G) based on specified ranges.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the grade column.
    column (str): The name of the column to encode.
    
    Returns:
    pd.DataFrame: The modified DataFrame with an added 'letter_grade' column.
    """
    grade_mapping = {
        'A': range(1, 6),
        'B': range(6, 11),
        'C': range(11, 16),
        'D': range(16, 21),
        'E': range(21, 26),
        'F': range(26, 31),
        'G': range(31, 36)
    }
    
    def map_to_letter(grade):
        for letter, grade_range in grade_mapping.items():
            if grade in grade_range:
                return letter
        return None 

    df['letter_grade'] = df[column].apply(map_to_letter)

    return df

def calculate_monthly_installment(df: pd.DataFrame, principal_column: str, interest_rate_column: str, term_column: str, lookup_table: pd.DataFrame):
    """
    Calculates the monthly installment for each loan based on the provided formula.

    Parameters:
    df (pd.DataFrame): The DataFrame containing loan information.
    principal_column (str): The name of the column for loan principal/amount.
    interest_rate_column (str): The name of the column for interest rate (annual).
    term_column (str): The name of the column for loan term (in months, as strings).

    Returns:
    pd.DataFrame: The modified DataFrame with a new column for monthly installments.
    """
    df = reverse_sqrt_transformation(df, principal_column)
    df = reverse_log_transformation(df, interest_rate_column)
    
    df['num_payments'] = df[term_column].str.extract('(\d+)').astype(int)
    df['monthly_interest_rate'] = df[interest_rate_column] / 12

    df['monthly_installment'] = (
        df[principal_column] *
        df['monthly_interest_rate'] *
        (1 + df['monthly_interest_rate']) ** df['num_payments'] /
        ((1 + df['monthly_interest_rate']) ** df['num_payments'] - 1)
    )
    
    df.drop(columns=['monthly_interest_rate', 'num_payments'], inplace=True)

    df = apply_sqrt_transformation(df, principal_column)
    df = apply_log_transformation(df, interest_rate_column, lookup_table)

    return df

def scrape_state_codes(url: str):
    """
    Scrapes state codes and their corresponding state names from a given URL.
    Args:
        url (str): The URL of the webpage containing the state codes table.
    Returns:
        dict: A dictionary where the keys are state alpha codes and the values are state names.
    Raises:
        Exception: If the webpage fails to load (status code is not 200).
    Example:
        >>> url = "http://example.com/state-codes"
        >>> state_codes = scrape_state_codes(url)
        >>> print(state_codes)
        {'CA': 'California', 'TX': 'Texas', ...}
    """
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to load page: {response.status_code}")
    soup = BeautifulSoup(response.text, 'html.parser')

    state_map = {}
    table = soup.find('table', class_='table')
    rows = table.find_all('tr')

    for row in rows:
        columns = row.find_all('td')
        if len(columns) > 1:  # Ensure there are enough columns
            state_name = columns[0].text.strip()
            alpha_code = columns[2].text.strip()
            state_map[alpha_code] = state_name

    return state_map

def add_state_name(df: pd.DataFrame, alpha_column: str):
    """
    Add a new column 'state_name' to the DataFrame based on the alpha code mapping.

    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    alpha_column (str): The name of the column containing the alpha codes.

    Returns:
    pd.DataFrame: The modified DataFrame with a new 'state_name
    """

    url = 'https://www23.statcan.gc.ca/imdb/p3VD.pl?Function=getVD&TVD=53971'
    state_codes = scrape_state_codes(url)
    # Add a new column to the DataFrame by mapping the alpha code to the state name
    df['state_name'] = df[alpha_column].map(state_codes)
    
    return df

# def one_hot_encode(df, columns_to_encode):
#     """
#     Performs one-hot encoding on specified columns and updates a lookup table.
    
#     Parameters:
#     - df (pd.DataFrame): The original DataFrame to encode.
#     - columns_to_encode (list of str): List of column names to one-hot encode.
    
#     Returns:
#     - pd.DataFrame: The DataFrame with one-hot encoded columns.
#     """

#     for column in columns_to_encode:

#         dummies = pd.get_dummies(df[column], prefix=column)
#         dummies.columns = [f"{clean_column_name(column)}" for column in dummies.columns]
        
#         df = pd.concat([df, dummies], axis=1)
#         df.drop(column, axis=1, inplace=True)
    
#     return df

def one_hot_encode(df, columns_to_encode, one_hot_categories_table, stream=False):
    """
    Performs one-hot encoding on specified columns and updates a one-hot categories lookup table.
    
    Parameters:
    - df (pd.DataFrame): The original DataFrame to encode.
    - columns_to_encode (list of str): List of column names to one-hot encode.
    - one_hot_categories_table (pd.DataFrame): Lookup table for storing one-hot encoded categories.
    - stream (bool): If False, updates one_hot_categories_table; if True, uses existing entries for encoding.

    Returns:
    - pd.DataFrame: The DataFrame with one-hot encoded columns.
    - pd.DataFrame: Updated one_hot_categories_table with encoding details.
    """
    
    for column in columns_to_encode:
        
        if not stream:
            dummies = pd.get_dummies(df[column], prefix=column)
            
            for category in dummies.columns:
                if not ((one_hot_categories_table['Column'] == column) &
                        (one_hot_categories_table['Encoded_Column'] == category)).any():
                    one_hot_categories_table = pd.concat([
                        one_hot_categories_table,
                        pd.DataFrame({'Column': [column], 
                                      'Category': [category.replace(f"{column}_", "")], 
                                      'Encoded_Column': [category]})
                    ])
            
            df = pd.concat([df, dummies], axis=1)
            df.drop(column, axis=1, inplace=True)
        
        else:
            categories = one_hot_categories_table[one_hot_categories_table['Column'] == column]
            for _, row in categories.iterrows():
                encoded_column = row['Encoded_Column']
                category = row['Category']
                
                df[encoded_column] = (df[column] == category).astype(bool)
            
            df.drop(column, axis=1, inplace=True)

    return df, one_hot_categories_table


# def ordinal_label_encode_column(df: pd.DataFrame, column: str, mapping: dict):
#     """
#     Encodes a categorical column with ordinal labels based on a specified mapping.

#     Parameters:
#     df (pd.DataFrame): The DataFrame containing the column to encode.
#     column (str): The name of the column to encode.
#     mapping (dict): A dictionary mapping original values to ordinal labels.

#     Returns:
#     pd.DataFrame: The modified DataFrame with the column encoded as ordinal labels.
#     """
    
#     df[column] = df[column].map(mapping)

#     for key, value in mapping.items():
#         lookup_table.loc[len(lookup_table)] = [column, key, value]

#     return df

# def label_encode_alphabetically(df: pd.DataFrame, column: str):
#     """
#     Label encodes a column alphabetically based on unique values.

#     Parameters:
#     df (pd.DataFrame): The DataFrame containing the column to encode.
#     column (str): The name of the column to encode.

#     Returns:
#     pd.DataFrame: The modified DataFrame with the column encoded alphabetically.
#     """
    
#     mapping = {value: idx + 1 for idx, value in enumerate(sorted(df[column].unique()))}
    
#     df = ordinal_label_encode_column(df, column, mapping)
    
#     return df
def ordinal_label_encode_column(df: pd.DataFrame, column: str, mapping: dict, lookup_table: pd.DataFrame, stream: bool = False):
    """
    Encodes a categorical column with ordinal labels based on a specified mapping.
    Updates the lookup table with new mappings if `stream` is False.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the column to encode.
    column (str): The name of the column to encode.
    mapping (dict): A dictionary mapping original values to ordinal labels.
    lookup_table (pd.DataFrame): Lookup table to track mappings for the column.
    stream (bool): If True, only use existing mappings in `lookup_table`; otherwise, update mappings.

    Returns:
    pd.DataFrame: The modified DataFrame with the column encoded as ordinal labels.
    pd.DataFrame: Updated lookup_table with new mappings if `stream` is False.
    """
    
    if not stream:
        for key, value in mapping.items():
            if not ((lookup_table['Column'] == column) & (lookup_table['Original'] == key)).any():
                lookup_table = pd.concat([lookup_table, pd.DataFrame({'Column': [column], 'Original': [key], 'Imputed': [value]})])

    encoding_map = {row['Original']: row['Imputed'] for _, row in lookup_table[lookup_table['Column'] == column].iterrows()}
    encoding_map.update(mapping)

    df[column] = df[column].map(encoding_map)

    return df, lookup_table


def label_encode_alphabetically(df: pd.DataFrame, column: str, lookup_table: pd.DataFrame, stream: bool = False):
    """
    Label encodes a column alphabetically based on unique values. Updates the lookup table if not streaming.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the column to encode.
    column (str): The name of the column to encode.
    lookup_table (pd.DataFrame): Lookup table to track mappings for the column.
    stream (bool): If True, only use existing mappings in `lookup_table`; otherwise, update mappings.

    Returns:
    pd.DataFrame: The modified DataFrame with the column encoded alphabetically.
    pd.DataFrame: Updated lookup_table with new mappings if `stream` is False.
    """
    
    if not stream:
        unique_values = sorted(df[column].unique())
        mapping = {value: idx + 1 for idx, value in enumerate(unique_values)}
    else:
        mapping = {row['Original']: row['Imputed'] for _, row in lookup_table[lookup_table['Column'] == column].iterrows()}

    df, lookup_table = ordinal_label_encode_column(df, column, mapping, lookup_table, stream)

    return df, lookup_table


emp_length_mapping = {
    '10+ years': 10,
    '9 years': 9,
    '8 years': 8,
    '7 years': 7,
    '6 years': 6,
    '5 years': 5,
    '4 years': 4,
    '3 years': 3,
    '2 years': 2,
    '1 year': 1,
    '< 1 year': 0.5
}

letter_grade_mapping = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7
}

# def scale_columns(df: pd.DataFrame, columns: list):
#     """
#     Scale specified columns in the DataFrame using Min-Max scaling.

#     Parameters:
#     df (pd.DataFrame): The DataFrame containing the columns to scale.
#     columns (list): The list of columns to scale.

#     Returns:
#     pd.DataFrame: The DataFrame with the specified columns scaled.
#     """
#     scaler = MinMaxScaler()

#     df[columns] = scaler.fit_transform(df[columns])

#     return df


def scale_columns(df: pd.DataFrame, columns: list, scaler_table: pd.DataFrame, stream: bool):
    """
    Scale specified columns in the DataFrame using Min-Max scaling, with streaming support.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the columns to scale.
    columns (list): The list of columns to scale.
    scaler_table (pd.DataFrame): DataFrame to hold min and max values for each column (used in streaming).
    stream (bool): If False, calculate and store scaling parameters; if True, use existing scaling parameters.

    Returns:
    pd.DataFrame: The DataFrame with the specified columns scaled.
    pd.DataFrame: Updated scaler_table with scaling parameters.
    """

    if not stream:
        for column in columns:
            scaler = MinMaxScaler()
            df[[column]] = scaler.fit_transform(df[[column]])

            min_val, max_val = scaler.data_min_[0], scaler.data_max_[0]
            scaler_table = pd.concat([
                scaler_table, 
                pd.DataFrame({'Column': [column], 'Min': [min_val], 'Max': [max_val]})
            ])
    else:
        for column in columns:
            column_min = scaler_table.loc[scaler_table['Column'] == column, 'Min'].values[0]
            column_max = scaler_table.loc[scaler_table['Column'] == column, 'Max'].values[0]

            df[column] = (df[column] - column_min) / (column_max - column_min)

    return df, scaler_table


def handling_inconsistent_data(df: pd.DataFrame):

    print("Cleaning column names")
    df = clean_column_names(df)

    print("Dropping duplicates")
    df = drop_duplicates(df)

    print("Parsing column data types")
    df = parse_column_data_types(df)

    print("Standardizing values to proper case")
    columns_to_standardize = ['emp_title', 'home_ownership', 'verification_status', 'type', 'description']
    df = standardize_values_proper_case(df, columns_to_standardize)

    print("Changing column values to mapped values")
    df = change_column_values_to_mapped_values(df, 'type', type_column_map)

    return df

def imputation(df: pd.DataFrame, lookup_table: pd.DataFrame, imputed_mode_table: pd.DataFrame, imputed_mean_table: pd.DataFrame, stream: bool):
    print("Checking for non-standard missing values")
    df = check_for_non_standard_missing_values(df, df.columns)
    print(df.isnull().sum())

    print("Imputing the annual_inc_joint column")
    df, lookup_table = impute_with_value(df, 'annual_inc_joint', 0, lookup_table, stream)
    print(df.isnull().sum())

    print("Impute the empl_length column")
    df, imputed_mode_table = impute_with_group_mode(df, "emp_length", "annual_inc", imputed_mode_table, stream)
    print(df.isnull().sum())

    print("Impute the emp_title column")
    df, imputed_mode_table = impute_with_group_mode(df, 'emp_title', 'emp_length', imputed_mode_table, stream)

    print("Impute the int_rate column")
    df, imputed_mean_table = impute_with_group_mean(df, 'int_rate', 'state', imputed_mean_table, stream)

    print("Impute the description column")
    df, imputed_mode_table = impute_with_group_mode(df, 'description', 'purpose', imputed_mode_table, stream)

    return df, lookup_table, imputed_mode_table, imputed_mean_table

def handling_outliers(df: pd.DataFrame, lookup_table: pd.DataFrame, iqr_threshold_table: pd.DataFrame, stream: bool):
    print("Applying the capping IQR method to the annual_inc column")
    df, iqr_threshold_table = apply_capping_iqr(df, 'annual_inc', iqr_threshold_table, stream)

    print("Applying the log transformation to the annual_inc_joint column")
    df = apply_log_transformation(df, 'annual_inc_joint', lookup_table)

    print("Applying the log transformation to the avg_cur_bal column")
    df = apply_log_transformation(df, 'avg_cur_bal', lookup_table)

    print("Applying the reciprocal transformation to the tot_cur_bal column")
    df = apply_log_transformation(df, 'tot_cur_bal', lookup_table)

    print("Applying the square root transformation to the loan_amount column")
    df = apply_sqrt_transformation(df, 'loan_amount')

    print("Applying the square root transformation to the funded_amount column")
    df = apply_sqrt_transformation(df, 'funded_amount')

    print("Applying the log transformation to the int_rate column")
    df = apply_log_transformation(df, 'int_rate', lookup_table)

    return df, iqr_threshold_table

def feature_engineering(df: pd.DataFrame, lookup_table: pd.DataFrame):
    print("Adding the month number column")
    df = add_month_number(df, 'issue_date', 'month_number')

    print("Add the salary can cover column")
    df = create_salary_can_cover_column(df, 'annual_inc', 'loan_amount')

    print("Adding the letter grade column")
    df = one_encode_grade(df, 'grade')

    print("Adding the monthly_installment column")
    df = calculate_monthly_installment(df, 'loan_amount', 'int_rate', 'term', lookup_table)

    print("Adding the state name column")
    df = add_state_name(df, 'state')

    return df

def encoding(df: pd.DataFrame, lookup_table: pd.DataFrame, one_hot_categories_table: pd.DataFrame, stream: bool):
    print("Label encoding the emp_length column")
    df, lookup_table = ordinal_label_encode_column(df, 'emp_length', emp_length_mapping, lookup_table, stream)

    print("One hot encoding the home ownership column")
    df, one_hot_categories_table = one_hot_encode(df, ['home_ownership'], one_hot_categories_table, stream)

    print("One hot encoding the verification_status column")
    df, one_hot_categories_table = one_hot_encode(df, ['verification_status'], one_hot_categories_table, stream)

    print("Label encoding the addr_state column")
    df, lookup_table = label_encode_alphabetically(df, 'addr_state', lookup_table, stream)

    print("One hot encoding the loan_status column")
    df, one_hot_categories_table = one_hot_encode(df, ['loan_status'], one_hot_categories_table, stream)

    print("One hot encoding the state column")
    df, lookup_table = label_encode_alphabetically(df, 'state', lookup_table, stream)

    print("One hot encoding the term column")
    df, one_hot_categories_table = one_hot_encode(df, ['term'], one_hot_categories_table, stream)

    print("Label encoding the letter_grade column")
    df, lookup_table = ordinal_label_encode_column(df, 'letter_grade', letter_grade_mapping, lookup_table, stream)
    
    print("One hot encoding the type column")
    df, one_hot_categories_table = one_hot_encode(df, ['type'], one_hot_categories_table, stream)

    print("One hot encoding the purpose column")
    df, one_hot_categories_table = one_hot_encode(df, ['purpose'], one_hot_categories_table, stream)

    df = clean_column_names(df)

    return df, lookup_table, one_hot_categories_table

def scaling(df: pd.DataFrame, scaler_table: pd.DataFrame, stream: bool):
    print("Scaling the annual_inc, annual_inc_joint, avg_cur_bal, tot_cur_bal, loan_amount, funded_amount, monthly_installment columns")
    df, scaler_table = scale_columns(df, ['annual_inc', 'annual_inc_joint', 'avg_cur_bal', 'tot_cur_bal', 'loan_amount', 'funded_amount', 'monthly_installment'], scaler_table, stream)

    df = set_index(df, 'loan_id')

    return df, scaler_table


def clean(df: pd.DataFrame, lookup_table: pd.DataFrame, imputed_mode_table: pd.DataFrame, imputed_mean_table: pd.DataFrame, iqr_threshold_table: pd.DataFrame, one_hot_categories_table: pd.DataFrame, scaler_table: pd.DataFrame, stream: bool = False):
    df = handling_inconsistent_data(df)
    df, lookup_table, imputed_mode_table, imputed_mean_table = imputation(df, lookup_table, imputed_mode_table, imputed_mean_table, stream)
    df, iqr_threshold_table = handling_outliers(df, lookup_table, iqr_threshold_table, stream)
    df = feature_engineering(df, lookup_table)
    df, lookup_table, one_hot_categories_table = encoding(df, lookup_table, one_hot_categories_table, stream)
    df, scaler_table = scaling(df, scaler_table, stream)

    return df, lookup_table, imputed_mode_table, imputed_mean_table, iqr_threshold_table, one_hot_categories_table, scaler_table






