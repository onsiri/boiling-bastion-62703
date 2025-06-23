import pandas as pd
from datetime import datetime


def filter_forecast_file(input_path, output_path, target_date='2025-12-31', max_items=10):
    """
    Process forecast file to keep only specified items with data up to target date

    Args:
        input_path (str): Path to input CSV file
        output_path (str): Path to save filtered CSV
        target_date (str): Cutoff date (YYYY-MM-DD format)
        max_items (int): Number of items to keep
    """
    try:
        # Read CSV with date parsing
        df = pd.read_csv(input_path, parse_dates=['ds'])

        # Convert target date to datetime
        target_dt = datetime.strptime(target_date, '%Y-%m-%d')

        # Filter items with data reaching target date
        items_with_full_history = df.groupby('group').filter(
            lambda x: x['ds'].max() >= target_dt
        )['group'].unique()

        # Get first n items meeting criteria
        selected_items = items_with_full_history[:max_items]

        # Filter original dataframe
        filtered_df = df[df['group'].isin(selected_items)]

        # Save filtered data
        filtered_df.to_csv(output_path, index=False)

        return filtered_df

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise

filter_forecast_file(
        input_path='/Users/onsiri/Documents/apps/item_forecasts.csv',
        output_path='/Users/onsiri/Documents/apps/filtered_item_forecasts.csv',
        target_date='2025-12-31',
        max_items=10
    )