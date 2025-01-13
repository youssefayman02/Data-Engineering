from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd

def inverse_scale(df: pd.DataFrame, column: str, scaler_table: pd.DataFrame):
    """
    Inversely scales a specified column in a DataFrame based on precomputed minimum and maximum values.
    Parameters:
    df (pd.DataFrame): The DataFrame containing the column to be inversely scaled.
    column (str): The name of the column to be inversely scaled.
    Returns:
    pd.Series: The inversely scaled values of the specified column.
    """

    min_value = scaler_table[scaler_table['Column'] == column]['Min'].values[0]
    max_value = scaler_table[scaler_table['Column'] == column]['Max'].values[0]

    return df[column] * (max_value - min_value) + min_value

def inverse_sqrt_transform(df: pd.DataFrame, column: str):
    """
    Applies an inverse square root transformation to a specified column in a DataFrame.
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The name of the column to transform.
    Returns:
    pd.Series: A Series with the transformed values.
    """
    return df[column] ** 2

def preprocess_data(df: pd.DataFrame, scaler_table: pd.DataFrame):
    """
    Preprocesses the input DataFrame by performing various transformations and feature engineering steps.
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data to be preprocessed.
    scaler_table (pd.DataFrame): The DataFrame containing scaling information for inverse transformations.
    Returns:
    pd.DataFrame: The preprocessed DataFrame with additional columns and transformations applied.
    Transformations applied:
    - Inverse scaling and inverse square root transformation on 'loan_amount'.
    - Inverse scaling on 'annual_inc'.
    - Derives 'loan_status' from multiple loan status columns.
    - Calculates the percentage of each loan grade and maps it to the 'loan_grade_percentage' column.
    - Extracts 'year' and 'month' from the 'issue_date' column.
    """

    df['loan_amount_original'] = inverse_scale(df, 'loan_amount', scaler_table)
    df['loan_amount_original'] = inverse_sqrt_transform(df, 'loan_amount_original')

    df['annual_inc_original'] = inverse_scale(df, 'annual_inc', scaler_table)

    loan_status_columns = [
        'loan_status_charged_off',
        'loan_status_current',
        'loan_status_default',
        'loan_status_fully_paid',
        'loan_status_in_grace_period',
        'loan_status_late_1630_days',
        'loan_status_late_31120_days'
    ]

    df['loan_status'] = df[loan_status_columns].idxmax(axis=1).str.replace('loan_status_', '', regex=False)

    loan_grade_counts = df['letter_grade'].value_counts(normalize=True) * 100
    df['loan_grade_percentage'] = df['letter_grade'].map(loan_grade_counts)

    df['year'] = pd.to_datetime(df['issue_date']).dt.year
    df['month'] = pd.to_datetime(df['issue_date']).dt.month

    return df

def create_dashboard(filename):
    
    df = pd.read_csv(filename)
    scaler_table = pd.read_csv('/opt/airflow/data/scaler_table.csv')
    df = preprocess_data(df, scaler_table)

    app = Dash(__name__)

    app.title = 'Fintech Loan Dashboard'

    app.layout = html.Div([
    html.H1('Fintech Loan Analysis Dashboard', style={'textAlign': 'center'}),
    html.P("Created by Youssef Ayman | ID: 52-14669", style={'textAlign': 'center'}),

    # 1) What is the distribution of loan amounts across different grades?
    dcc.Graph(
        id='loan-grade-distribution',
        figure=px.box(df, x='letter_grade', 
                      y='loan_amount_original', 
                      title='What is the distribution of loan amounts across different grades?',
                        labels={'loan_amount_original': 'Loan Amount', 'letter_grade': 'Grade'})\
                        .update_layout(
                            height=900,
                            title_font_size=20,
                        )
        ),

    # 2) How does the loan amount relate to annual income across states?
    dcc.Dropdown(
        id='state-dropdown',
        options=[{'label': 'All', 'value': 'All'}] + [{'label': state, 'value': state} for state in sorted(df['state_name'].unique())],
        value='All', # default value
        placeholder='Select a state',
        clearable=False,
        ),

    dcc.Graph(id='loan-amount-vs-annual-income'),


    # 3)  What is the trend of loan issuance over the months (number of loans per month), filtered by year?
    dcc.Dropdown(
        id='year-dropdown',
        options=[{'label': str(year), 'value': year} for year in sorted(df['year'].unique())],
        value=df['year'].min(),
        placeholder='Select a year',
        clearable=False
        ),
    dcc.Graph(id='loan-trend-month-graph'),
    
    # 4)  Which states have the highest average loan amount?
    dcc.Graph(
        id='states-highest-loan-amount',
        figure=px.bar(df.groupby('state_name')['loan_amount_original'].mean().sort_values(ascending=False),
                      title='Which states have the highest average loan amount?',
                      labels={'state_name': 'State', 'value': 'Average Loan Amount'})\
                      .update_layout(
                                        height=900,
                                        title_font_size=20,
                                    )
        ),

    # 5) What is the percentage distribution of loan grades in the dataset?
    dcc.Graph(
        id='loan-grade-percentage',
        figure=px.histogram(df, x='letter_grade',
                            title='What is the percentage distribution of loan grades in the dataset?',
                            labels={'letter_grade': 'Grade', 'count': 'Percentage'},
                            histnorm='percent',
                            text_auto='.2f').update_layout(yaxis=dict(title='Percentage'))\
                            .update_layout(
                                            height=900,
                                            title_font_size=20,
                                        )
        ),

    ])

    @app.callback(
        Output('loan-amount-vs-annual-income', 'figure'),
        Input('state-dropdown', 'value')
    )

    def update_loan_amount_vs_annual_income_graph(state):
        """
        Updates the loan amount vs. annual income graph based on the selected state.
        Parameters:
        state (str): The selected state.
        Returns:
        dict: A dictionary containing the updated graph data.
        """
        if state == 'All':
            filtered_df = df
        else:
            filtered_df = df[df['state_name'] == state]

        fig = px.scatter(filtered_df, 
                        x='annual_inc_original', 
                        y='loan_amount_original', 
                        color='loan_status',
                        title='How does the loan amount relate to annual income across states?',
                        labels={
                                'annual_inc_original': 'Annual Income',
                                'loan_amount_original': 'Loan Amount',
                                'loan_status': 'Loan Status'
                        },
                        hover_data={'loan_status': True})
        
        fig.update_layout(
            legend_title='Loan Status',
            xaxis_title='Annual Income',
            yaxis_title='Loan Amount',
            height=900,
            title_font_size=20
        )

        return fig

    @app.callback(
        Output('loan-trend-month-graph', 'figure'),
        Input('year-dropdown', 'value')
    )

    def update_loan_trend_month_graph(year):
        """
        Updates the loan trend graph based on the selected year.
        Parameters:
        year (int): The selected year.
        Returns:
        dict: A dictionary containing the updated graph data.
        """
        filtered_df = df[df['year'] == year]
        fig = px.line(filtered_df.groupby('month')['loan_amount_original'].sum().reset_index(),
                    x='month', 
                    y='loan_amount_original', 
                    title='What is the trend of loan issuance over the months (total loan amount per month), filtered by year',
                    labels={'loan_amount_original': 'Total Loan Amount', 'month': 'Month'})\
        
        fig.update_layout(xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
        ),
        height=900,
        title_font_size=20)

        return fig
    
    print('Running dashboard... on 8050')
    app.run_server(host='0.0.0.0', debug=False, port=8050, threaded=True)
