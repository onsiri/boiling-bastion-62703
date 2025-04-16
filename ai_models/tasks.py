from celery import shared_task
from keras import Sequential
from keras.src.layers import Embedding, LSTM
from keras.src.utils import pad_sequences
from sklearn.preprocessing import LabelEncoder
from ai_models.models import Transaction, NextItemPrediction, Item
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from django.db import connection
import pandas as pd
from django.utils import timezone
import logging
from .management.commands.generate_predictions import PredictionPipeline
from celery.utils.log import get_task_logger
from django.apps import apps
from django.core.management.color import no_style
logger = get_task_logger(__name__)
from django.db import transaction, connections
from io import StringIO

@shared_task(bind=True)
def generate_predictions_task(self):
    try:
        logger.info("Starting prediction pipeline")
        pipeline = PredictionPipeline()
        pipeline.run()
        return "Predictions generated successfully"
    except Exception as e:
        logger.error(f"Task failed: {str(e)}")
        raise self.retry(exc=e, countdown=60)  # Retry after 60 seconds



@shared_task
def async_predict_future_sales():
    try:
        # Move your existing predict_future_sales logic here
        # Ensure no request objects are used
        print(f"[{timezone.now()}] Task started")
        BATCH_SIZE = 512
        SEQUENCE_LENGTH = 5
        EPOCHS = 2

        # Data preparation
        print('#1 ata preparation')
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
        print('#2 Model setup (initialize once)')
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
        print('#3 Prepare all sequences first')
        df = pd.DataFrame(transactions)
        df['ItemCodeEncoded'] = label_encoder.transform(df['ItemCode'])
        user_sequences = df.groupby('user_id')['ItemCodeEncoded'].apply(list).to_dict()

        # Epoch training loop
        print('#4 Epoch training loop')
        print('#5 truncate first existing records')
        with transaction.atomic():
            deleted_count = NextItemPrediction.objects.all().delete()[0]
            print(f'✅ Deleted NextItemPrediction: {deleted_count} existing records')
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
            print('#6 Periodic saving')
            if predictions:
                with transaction.atomic():
                    NextItemPrediction.objects.bulk_create(predictions, batch_size=1000)
                predictions = []
        print(f"[{timezone.now()}] Predictions saved: {len(predictions)} items")
        return "SUCCESS"
    except Exception as e:
        logger.error(f"Task failed: {e}")
        raise

@shared_task
def process_analytics_after_upload():
    #async_generate_forecast.delay()
    async_predict_future_sales.delay()


@shared_task
def async_upload_object_db(model_path, df_json):
    try:
        model = apps.get_model(model_path)  # Must be "app_label.ModelName"
        df = pd.read_json(df_json)

        # Verify exact column names
        df = df.rename(columns={
            'ItemDescription': 'group'  # If DF uses different column name
        })

        # Prepare data for bulk insertion
        columns = ['ds', 'group', 'prediction', 'prediction_lower', 'prediction_upper']
        df = df[columns].copy()
        df['ds'] = pd.to_datetime(df['ds']).dt.date  # Convert to date objects

        # Reset DB sequence
        conn = connections['default']
        sequence_sql = conn.ops.sequence_reset_sql(no_style(), [model])
        with conn.cursor() as cursor:
            for sql in sequence_sql:
                cursor.execute(sql)

        # Batch insert directly using low-level SQL
        with transaction.atomic():
            batch_size = 10000  # Adjusted for large datasets
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i + batch_size]
                model.objects.bulk_create(
                    [model(**row) for _, row in batch.iterrows()],
                    batch_size=batch_size,
                    ignore_conflicts=True
                )

        print(f'🔥 Success: {len(df)} records inserted')

    except Exception as e:
        print(f'💥 Catastrophic failure: {str(e)}')
        raise