from tempfile import NamedTemporaryFile
from psycopg2.extras import execute_values
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
from django.db import transaction, connections
from io import StringIO
import datetime
import gc
import io
from django.utils.timezone import now
import numpy as np

logger = get_task_logger(__name__)

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
def async_upload_object_db(model_path, df):
    """Memory-safe CSV upload with PostgreSQL reserved keyword handling"""
    try:
        model = apps.get_model(model_path)

        # 1. Column validation and renaming
        df = df.rename(columns={'ItemDescription': 'group'})
        required_cols = ['ds', 'group', 'prediction', 'prediction_lower', 'prediction_upper']

        if not set(required_cols).issubset(df.columns):
            missing = set(required_cols) - set(df.columns)
            raise ValueError(f"Missing columns: {missing}")

        # 2. Clean data in chunks
        chunksize = 10000  # Process in 10k row chunks
        total_inserted = 0

        with connection.cursor() as cursor:
            # 3. Stream directly to PostgreSQL using binary COPY
            cursor.execute("SET SESSION statement_timeout = 0;")

            for chunk in pd.read_csv(
                    df.to_csv(index=False, header=True, encoding='utf-8'),
                    chunksize=chunksize,
                    parse_dates=['ds'],
                    infer_datetime_format=True,
                    dtype={'group': 'category'}
            ):
                # Clean chunk data
                chunk = chunk.dropna()
                if chunk.empty:
                    continue

                # 4. Use CSV format with proper quoting
                csv_buffer = chunk.to_csv(
                    index=False,
                    header=False,
                    encoding='utf-8',
                    columns=required_cols,
                    date_format='%Y-%m-%d'
                )

                # 5. Execute COPY with proper column quoting
                copy_sql = f"""
                    COPY {model._meta.db_table} 
                    (ds, "group", prediction, prediction_lower, prediction_upper)
                    FROM STDIN WITH (
                        FORMAT CSV,
                        DELIMITER ',',
                        NULL '',
                        ENCODING 'UTF8'
                    )
                """

                cursor.copy_expert(copy_sql, io.StringIO(csv_buffer))
                total_inserted += len(chunk)

                # 6. Explicit memory cleanup
                del chunk, csv_buffer
                gc.collect()

        logger.info(f'🔥 Success: {total_inserted} records inserted')
        return total_inserted

    except Exception as e:
        logger.error(f'💥 Catastrophic failure: {str(e)}', exc_info=True)
        raise