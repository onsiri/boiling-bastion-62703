import pandas as pd
from botocore.exceptions import ClientError
from django.db.models.signals import post_save
from django.dispatch import receiver
from keras import Sequential
from keras.src.layers import Embedding, LSTM
from keras.src.utils import to_categorical
from prophet import Prophet
from django.utils import timezone
from sklearn.neighbors import NearestNeighbors
from tensorflow.python.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from .models import Transaction, sale_forecast
from django.db.models import Max, Prefetch, Sum
from datetime import datetime
from django.db.models.signals import post_save
from django.dispatch import receiver
from ai_models.models import Transaction, NextItemPrediction, Item, CustomerDetail
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from django.utils import timezone
from django.db import transaction
from django.db.models import Count
from sklearn.metrics import mean_absolute_error, mean_squared_error
import boto3
from django.conf import settings
import gc
import os
from django.apps import apps
from django.db import transaction

def generate_presigned_url(file_name):
    s3 = boto3.client(
        's3',
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_S3_REGION_NAME
    )

    presigned_url = s3.generate_presigned_url(
        ClientMethod='put_object',
        Params={
            'Bucket': settings.AWS_STORAGE_BUCKET_NAME,
            'Key': f'uploads/{file_name}'
        },
        ExpiresIn=3600
    )
    return presigned_url


def generate_forecast():
    # 1. Stream data from database in chunks
    print('# 1. Stream data from database in chunks')
    latest_transaction = Transaction.objects.order_by('-TransactionTime').first()
    if not latest_transaction:
        return

    # 2. Database-side aggregation (group by country and date)
    print('# 2. Database-side aggregation (by country and date)')
    daily_sales = (
        Transaction.objects
        .values('TransactionTime__date', 'Country')  # Add Country to grouping
        .annotate(total_sales=Sum('NumberOfItemsPurchased'))
        .order_by('TransactionTime__date', 'Country')
    )

    # 3. Optimized DataFrame creation (one row per country-date)
    print('# 3. Optimized DataFrame creation')
    df = pd.DataFrame.from_records(
        daily_sales,
        columns=['TransactionTime__date', 'Country', 'total_sales']
    )
    df = df.rename(columns={
        'TransactionTime__date': 'ds',
        'total_sales': 'y'
    })
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values(['Country', 'ds'])  # Sort by country and date

    # 4. Initialize Prophet model once (reused across countries)
    print('# 4. Prophet configuration')
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='additive',
        mcmc_samples=0,
        uncertainty_samples=0
    )

    # 5. Batch delete old forecasts
    print('# 5. Delete old forecasts')
    with transaction.atomic():
        sale_forecast.objects.all().delete()

    # 6. Forecast for each country (memory-efficient loop)
    print('# 6. Forecast by country')
    batch_size = 1000
    forecasts = []

    for country in df['Country'].unique():
        country_df = df[df['Country'] == country][['ds', 'y']]

        # Skip countries with no historical data
        if country_df['y'].sum() == 0:
            continue

        # Train model
        try:
            model.fit(country_df)
        except Exception as e:
            print(f"Failed for {country}: {str(e)}")
            continue

        # Generate forecast
        future = model.make_future_dataframe(periods=365, freq='D')
        forecast = model.predict(future)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        # Clip negatives to 0
        forecast['yhat'] = forecast['yhat'].clip(lower=0)
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)

        # Add to bulk insert batch
        for _, row in forecast.iterrows():
            forecasts.append(sale_forecast(
                ds=row['ds'].date(),
                prediction=row['yhat'],
                prediction_lower=row['yhat_lower'],
                prediction_upper=row['yhat_upper'],
                country=country,  # Ensure your model has this field
                accuracy_score=0
            ))

            # Batch insert to reduce database calls
            if len(forecasts) >= batch_size:
                with transaction.atomic():
                    sale_forecast.objects.bulk_create(forecasts)
                    forecasts = []

    # Insert remaining forecasts
    if forecasts:
        with transaction.atomic():
            sale_forecast.objects.bulk_create(forecasts)

    # 7. Cleanup
    print('# 7. Memory cleanup')
    del df, forecasts, daily_sales
    gc.collect()


def predict_future_sales(request):
    try:
        BATCH_SIZE = 1024
        SEQUENCE_LENGTH = 10
        EPOCHS = 5

        # Data preparation
        transactions = Transaction.objects.select_related('user').values(
            'user_id', 'ItemCode'
        ).order_by('user_id', 'TransactionTime')

        if not transactions.exists():
            return  # Add proper early return handling

        items = Item.objects.all().values('ItemCode', 'ItemDescription', 'CostPerItem')
        item_df = pd.DataFrame(items)
        item_map = item_df.set_index('ItemCode').to_dict('index')

        label_encoder = LabelEncoder()
        label_encoder.fit(item_df['ItemCode'])

        # Model setup (initialize once)
        model = Sequential([
            Embedding(input_dim=len(label_encoder.classes_), output_dim=64),
            LSTM(128, return_sequences=False),
            Dense(len(label_encoder.classes_), activation='softmax')
        ])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Prepare all sequences first
        all_sequences = []
        df = pd.DataFrame(transactions)
        df['ItemCodeEncoded'] = label_encoder.transform(df['ItemCode'])
        user_sequences = df.groupby('user_id')['ItemCodeEncoded'].apply(list).to_dict()

        # Epoch training loop
        for epoch in range(EPOCHS):
            predictions = []

            for user_id, sequence in user_sequences.items():
                # Sequence processing
                if len(sequence) < SEQUENCE_LENGTH:
                    continue

                # Sliding window with randomization
                windows = [
                    sequence[i:i + SEQUENCE_LENGTH]
                    for i in range(len(sequence) - SEQUENCE_LENGTH)
                ]
                np.random.shuffle(windows)

                # Batch processing
                for i in range(0, len(windows), BATCH_SIZE):
                    batch = windows[i:i + BATCH_SIZE]
                    X = pad_sequences([seq[:-1] for seq in batch], maxlen=SEQUENCE_LENGTH - 1)
                    y = np.array([seq[-1] for seq in batch])

                    # Train and predict
                    model.train_on_batch(X, y)
                    preds = model.predict(X, verbose=0)

                    # Store predictions
                    for seq, pred in zip(batch, preds):
                        pred_idx = np.argmax(pred)
                        prob = pred[pred_idx]
                        item_code = label_encoder.inverse_transform([pred_idx])[0]

                        predictions.append(NextItemPrediction(
                            UserId=user_id,
                            PredictedItemCode=item_code,
                            PredictedItemDescription=item_map.get(item_code, {}).get('ItemDescription', ''),
                            PredictedItemCost=item_map.get(item_code, {}).get('CostPerItem', 0),
                            Probability=float(prob),
                            PredictedAt=timezone.now()
                        ))

            # Periodic saving
            if predictions:
                with transaction.atomic():
                    NextItemPrediction.objects.bulk_create(
                        predictions,
                        batch_size=1000,
                        update_conflicts=True,
                        update_fields=[
                            'PredictedItemCode',
                            'PredictedItemDescription',
                            'PredictedItemCost',
                            'Probability',
                            'PredictedAt'
                        ],
                        unique_fields=['UserId', 'PredictedAt']
                    )
                predictions = []

        return True  # Or return appropriate response

    except Exception as e:
        if request:  # Handle CLI vs web context
            print(f"Critical error: {str(e)}")
        raise ValueError("Critical error: {str(e)}")

    finally:
        from keras import backend as K
        K.clear_session()  # Critical for TensorFlow
        gc.collect()



def get_customer_transaction_data():
    """
    Perform left join between CustomerDetail and Transaction
    Returns QuerySet of existing customers with their transactions
    """
    return CustomerDetail.objects.filter(
        existing_customer='Yes'
    ).prefetch_related(
        Prefetch('transactions',
                 queryset=Transaction.objects.all(),
                 to_attr='transaction_list')
    ).annotate(
        transaction_count=Count('transactions')
    )


def prepare_prediction_data():
    """Prepare customer data for prediction pipeline"""
    customers = CustomerDetail.objects.select_related('predictions').all()
    return pd.DataFrame.from_records(
        customers.values(
            'UserId',
            'existing_customer',
            'country',
            'age',
            'gender',
            'income'
        )
    )




def get_customer_history(user_id):
    """Get a customer's transaction history"""
    return Transaction.objects.filter(user__UserId=user_id).order_by('-TransactionTime')

#@receiver(post_save, sender=Transaction)
#def handle_new_transaction(sender, instance, **kwargs):
#    """Handle new transaction signal"""
 #   generate_forecast()


def import_forecasts_from_s3(filename):
    """Downloads CSV from S3 and returns DataFrame"""
    # 1. Configure AWS credentials
    print('# 1. Configure AWS credentials')
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    S3_BUCKET = 'dsinsight-3904-0253-1082-us-east-1'
    s3_path = filename  # Use parameter as local path

    # 2. Download from S3
    print('# 2. Download from S3')
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    try:
        s3.download_file(S3_BUCKET, filename, s3_path)
    except ClientError as e:
        print(f"S3 download error: {e}")
        return None

    # 3. Read CSV with cleanup
    print('# 3. Read CSV with cleanup')
    try:
        df = pd.read_csv(s3_path, parse_dates=['ds'])

        # Validate required columns
        #required_columns = ['ds', 'country', 'prediction', 'prediction_lower', 'prediction_upper']
        #if not all(col in df.columns for col in required_columns):
        #    raise ValueError("CSV missing required columns")
        #print('# 4. Read CSV with cleanup successfully')
        return df
    finally:
        if os.path.exists(s3_path):
            os.remove(s3_path)





def upload_object_db(model_object_name, df):
    """Uploads DataFrame to specified model with progress tracking"""
    print('[1/4] Getting model class...')
    try:
        model_class = apps.get_model('ai_models', model_object_name)
    except LookupError as e:
        raise ValueError(f"Model error: {str(e)}")

    print('[2/4] Clearing existing records...')
    with transaction.atomic():
        deleted_count = model_class.objects.all().delete()[0]
        print(f'✅ Deleted {deleted_count} existing records')

    print('[3/4] Preparing batch inserts...')
    batch_size = 1000
    total_records = len(df)
    forecasts = []

    # Check if DataFrame is empty
    if df.empty:
        raise ValueError("No data to upload!")

    # Only proceed if model is sale_forecast
    if model_class.__name__ == 'sale_forecast':
        for idx, row in enumerate(df.itertuples(), 1):
            try:
                # Convert ds to date (handle timezone if needed)
                ds_date = pd.to_datetime(row.ds).date()
                forecasts.append(model_class(
                    ds=ds_date,
                    prediction=float(row.yhat),
                    prediction_lower=float(row.yhat_lower),
                    prediction_upper=float(row.yhat_upper),
                    accuracy_score=0.0  # Default value
                ))
            except Exception as e:
                print(f"Error in row {idx}: {str(e)}")
                raise

            # Batch insertion
            if len(forecasts) >= batch_size:
                model_class.objects.bulk_create(forecasts)
                forecasts = []

        # Insert remaining records
        if forecasts:
            model_class.objects.bulk_create(forecasts)

        # Only proceed if model is CountrySaleForecast
    if model_class.__name__ == 'CountrySaleForecast':
        for idx, row in enumerate(df.itertuples(), 1):
            try:
                # Convert ds to date (handle timezone if needed)
                ds_date = pd.to_datetime(row.ds).date()
                forecasts.append(model_class(
                        ds=ds_date,
                        group=row.country,
                        prediction=float(row.prediction),
                        prediction_lower=float(row.prediction_lower),
                        prediction_upper=float(row.prediction_upper)
                    ))
            except Exception as e:
                print(f"Error in row {idx}: {str(e)}")
                raise

                # Batch insertion
            if len(forecasts) >= batch_size:
                model_class.objects.bulk_create(forecasts)
                forecasts = []

            # Insert remaining records
        if forecasts:
            model_class.objects.bulk_create(forecasts)

    # Only proceed if model is ItemSaleForecast
    if model_class.__name__ == 'ItemSaleForecast':
        for idx, row in enumerate(df.itertuples(), 1):
            try:
                # Convert ds to date (handle timezone if needed)
                ds_date = pd.to_datetime(row.ds).date()
                forecasts.append(model_class(
                    ds=ds_date,
                    group=row.ItemDescription,
                    prediction=float(row.prediction),
                    prediction_lower=float(row.prediction_lower),
                    prediction_upper=float(row.prediction_upper)
                ))
            except Exception as e:
                print(f"Error in row {idx}: {str(e)}")
                raise

                # Batch insertion
            if len(forecasts) >= batch_size:
                model_class.objects.bulk_create(forecasts)
                forecasts = []

            # Insert remaining records
        if forecasts:
            model_class.objects.bulk_create(forecasts)

    print(f'🎉 Total {total_records} records inserted successfully')