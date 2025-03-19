import pandas as pd
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
from django.db.models import Max, Prefetch
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

def generate_forecast():
    # Get the latest transaction time
    latest_transaction_time = Transaction.objects.aggregate(Max('TransactionTime'))['TransactionTime__max']
    if not latest_transaction_time:
        return  # No transactions yet, nothing to forecast

    # Convert Transaction data to a DataFrame suitable for Prophet
    transactions = Transaction.objects.values('TransactionTime', 'NumberOfItemsPurchased')
    df = pd.DataFrame.from_records(transactions)
    df.rename(columns={'TransactionTime': 'ds', 'NumberOfItemsPurchased': 'y'}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'])

    # Ensure the DataFrame is sorted by date
    df.sort_values('ds', inplace=True)

    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(df)

    # Create a DataFrame to hold the future dates for the forecast
    future_dates = model.make_future_dataframe(periods=30, freq='D')

    # Make the forecast
    forecast = model.predict(future_dates)

    # Extract the forecast data and save to the sale_forecast table
    for index, row in forecast.tail(30).iterrows():
        sale_forecast.objects.create(
            ds=row['ds'].strftime('%Y-%m-%d'),
            prediction=row['yhat'],
            prediction_lower=row['yhat_lower'],
            prediction_upper=row['yhat_upper'],
            uploaded_at=datetime.now()
        )


def predict_future_sales():
    try:
        # Batch processing parameters
        BATCH_SIZE = 1024
        SEQUENCE_LENGTH = 10
        EPOCHS = 5

        # Get transactions with efficient database query
        transactions = Transaction.objects.select_related('user').values(
            'user_id', 'ItemCode'
        ).order_by('user_id', 'TransactionTime')

        if not transactions.exists():
            print("No transactions to process")
            return

        print("Starting prediction processing")

        # Single database query to get all items
        items = Item.objects.all().values('ItemCode', 'ItemDescription', 'CostPerItem')
        item_df = pd.DataFrame(items)
        item_map = item_df.set_index('ItemCode').to_dict('index')

        # Prepare label encoder with all possible items
        label_encoder = LabelEncoder()
        all_item_codes = item_df['ItemCode'].unique()
        label_encoder.fit(all_item_codes)

        # Prepare data in pandas with vectorized operations
        df = pd.DataFrame(transactions)
        df['ItemCodeEncoded'] = label_encoder.transform(df['ItemCode'])

        # Precompute user sequences in a dictionary
        user_sequences = df.groupby('user_id')['ItemCodeEncoded'].apply(list).to_dict()

        # Model initialization (single instance)
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

        predictions = []
        user_batch = []
        sequence_batch = []

        # Batch processing of users
        for user_id, sequence in user_sequences.items():
            if len(sequence) < SEQUENCE_LENGTH:
                continue

            # Extract sequences using sliding window approach
            for i in range(len(sequence) - SEQUENCE_LENGTH):
                seq = sequence[i:i + SEQUENCE_LENGTH]
                sequence_batch.append(seq[:-1])
                user_batch.append((user_id, seq[-1]))

            if len(sequence_batch) >= BATCH_SIZE:
                # Convert to numpy arrays
                X = pad_sequences(sequence_batch, maxlen=SEQUENCE_LENGTH - 1)
                y = np.array([item[1] for item in user_batch])

                # Train on batch
                model.train_on_batch(X, y)

                # Generate predictions
                preds = model.predict(X, batch_size=BATCH_SIZE, verbose=0)
                predicted_indices = np.argmax(preds, axis=1)

                # Process batch predictions
                for (user_id, _), pred_idx in zip(user_batch, predicted_indices):
                    item_code = label_encoder.inverse_transform([pred_idx])[0]
                    predictions.append(NextItemPrediction(
                        UserId=user_id,
                        PredictedItemCode=item_code,
                        PredictedItemDescription=item_map[item_code]['ItemDescription'],
                        PredictedItemCost=item_map[item_code]['CostPerItem'],
                        Probability=np.max(preds),
                        PredictedAt=timezone.now()
                    ))

                sequence_batch = []
                user_batch = []

        # Process remaining items in last batch
        if sequence_batch:
            X = pad_sequences(sequence_batch, maxlen=SEQUENCE_LENGTH - 1)
            y = np.array([item[1] for item in user_batch])
            model.train_on_batch(X, y)
            preds = model.predict(X, verbose=0)
            predicted_indices = np.argmax(preds, axis=1)

            for (user_id, _), pred_idx in zip(user_batch, predicted_indices):
                item_code = label_encoder.inverse_transform([pred_idx])[0]
                predictions.append(NextItemPrediction(
                    UserId=user_id,
                    PredictedItemCode=item_code,
                    PredictedItemDescription=item_map[item_code]['ItemDescription'],
                    PredictedItemCost=item_map[item_code]['CostPerItem'],
                    Probability=np.max(preds),
                    PredictedAt=timezone.now()
                ))

        # Bulk create predictions
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
                unique_fields=['UserId']
            )

        print(f"Successfully processed {len(predictions)} predictions")

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise



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

@receiver(post_save, sender=Transaction)
def handle_new_transaction(sender, instance, **kwargs):
    """Handle new transaction signal"""
    generate_forecast()

