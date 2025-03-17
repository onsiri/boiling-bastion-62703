from django.core.management.base import BaseCommand
from ai_models.models import Transaction, sale_forecast
from prophet import Prophet
import pandas as pd
from dateutil import parser
from datetime import datetime

import pytz
class Command(BaseCommand):
    help = 'Generate forecast'

    def handle(self, *args, **options):
        # Get all transactions
        transactions = Transaction.objects.all()

        # Create a pandas dataframe from the transactions
        df = pd.DataFrame({
            'ds': [t.TransactionTime for t in transactions],
            'y': [t.CostPerItem * t.NumberOfItemsPurchased for t in transactions]
        })


        model = Prophet()

        # Fit the model to the data
        model.fit(df)

        # Generate the next 30 days forecast
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Save the forecast to the SaleForecast model
        for index, row in forecast.iterrows():
            sale_forecast.objects.create(
                ds=row['ds'].strftime('%Y-%m-%d'),
                prediction=row['yhat'],
                prediction_lower=row['yhat_lower'],
                prediction_upper=row['yhat_upper']
            )