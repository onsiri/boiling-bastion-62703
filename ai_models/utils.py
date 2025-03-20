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

