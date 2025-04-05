from celery import shared_task


from ai_models.utils import generate_forecast, predict_future_sales
from django.db import connection
import pandas as pd
@shared_task
def async_generate_forecast():
    generate_forecast()

@shared_task
def async_predict_future_sales():
    predict_future_sales(None)  # Pass None if no request object needed

@shared_task
def process_analytics_after_upload():
    async_generate_forecast.delay()
    async_predict_future_sales.delay()

@shared_task
async def async_upload_object_db(model_object_name, df_json):
    df = pd.read_json(df_json)
    table_name = model_class._meta.db_table
    columns = ['ds', 'group', 'prediction', 'prediction_lower', 'prediction_upper']

    # Prepare tuples for all rows
    values = [
        (
            pd.to_datetime(row.ds).date().isoformat(),
            row.country if model_class.__name__ == 'CountrySaleForecast' else row.ItemDescription,
            float(row.prediction),
            float(row.prediction_lower),
            float(row.prediction_upper)
        )
        for row in df.itertuples()
    ]

    # Batch insert using raw SQL
    batch_size = 500  # Reduce batch size for Heroku
    with connection.cursor() as cursor:
        for i in range(0, len(values), batch_size):
            batch = values[i:i + batch_size]
            query = f"""
                    INSERT INTO {table_name} 
                    ({", ".join(columns)})
                    VALUES {", ".join(["%s"] * len(batch))}
                """
            cursor.execute(query, batch)
    print(f'🎉 Total records inserted successfully')
