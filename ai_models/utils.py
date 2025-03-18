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
        # Get transactions from the last 5 minutes to avoid processing old data
        time_threshold = timezone.now() - timezone.timedelta(minutes=5)
        recent_transactions = Transaction.objects.all()#filter(uploaded_at__gte=time_threshold)
        if not recent_transactions.exists():
            print("No recent transactions to process")
            return

        print("Starting prediction for recent upload")

        # Convert to DataFrame
        df = pd.DataFrame(list(recent_transactions.values('user_id', 'ItemCode')))

        # Encode ItemCode to integers
        label_encoder = LabelEncoder()
        df['ItemCodeEncoded'] = label_encoder.fit_transform(df['ItemCode'])

        # Group sequences by user
        unique_users = df['user_id'].unique()

        # Define the model outside the loop
        sequence_length = 10
        model = Sequential([
            Embedding(
                input_dim=len(label_encoder.classes_),
                output_dim=50
            ),
            LSTM(100),
            Dense(len(label_encoder.classes_), activation='softmax')
        ])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

        for user in unique_users:
            user_df = df[df['user_id'] == user]
            if len(user_df) < sequence_length:
                continue

            # Get sequence of transactions for this user
            user_sequence = user_df['ItemCodeEncoded'].tolist()

            # Pad sequences for the LSTM model
            padded_sequence = np.array([user_sequence[-sequence_length:]]).reshape(1, -1)

            # Prepare training data
            X = padded_sequence[:, :-1]
            y = np.array([user_sequence[-1]])

            # Train the model
            model.fit(X, y, epochs=5, batch_size=32, verbose=0)

            # Generate predictions
            prediction = model.predict(X, verbose=0)
            predicted_index = np.argmax(prediction[0])
            predicted_item_code = label_encoder.inverse_transform([predicted_index])[0]
            probability = np.max(prediction[0])

            item = Item.objects.get(ItemCode=predicted_item_code)
            item_description = item.ItemDescription
            item_cost = item.CostPerItem

            # Save the prediction
            NextItemPrediction.objects.update_or_create(
                UserId=str(user),
                defaults={
                    'PredictedItemCode': predicted_item_code,
                    'Probability': probability,
                    'PredictedItemDescription': item_description,
                    'PredictedItemCost': item_cost,
                    'PredictedAt': timezone.now()
                }
            )
        print(f"prediction has completed ")
    except Exception as e:
        print(f"An error occurred: {str(e)}")



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

