import pandas as pd
from django.db import transaction
from ..models import NextItemPrediction, CustomerDetail, NewCustomerRecommendation
from ..ml.recommender import PurchasePredictor
import numpy as np
from datetime import datetime, timedelta

class PredictionPipeline:
    def __init__(self):
        self.predictor = PurchasePredictor()
        self.features = None

    def run(self):
        """Main pipeline execution"""
        print("\n[1/4] Preparing data...")
        self.features = self.predictor.prepare_data()
        print("[2/4] Training model...")
        self.predictor.train_model(self.features)

        print("[3/4] Fetching new customers...")
        new_customers = CustomerDetail.objects.filter(
            existing_customer=False
        )
        print(f"Found {new_customers.count()} new customers to process")

        with transaction.atomic():
            #NextItemPrediction.objects.all().delete()
            self._process_predictions(new_customers)

    def _process_predictions(self, customers):
        """Process predictions for new customers"""
        predictions = []
        total = customers.count()
        for i, customer in enumerate(customers, 1):
            if i % 100 == 0 or i == total:
                print(f"Processing customer {i}/{total} ({i / total:.1%})")

            cust_features = self._get_customer_features(customer)
            neighbor_indices = self.predictor.predict(cust_features)

            customer_predictions = self._create_prediction_records(customer, neighbor_indices)
            if customer_predictions:
                predictions.extend(customer_predictions)

        print(f"\nSaving {len(predictions)} predictions to database...")
        NewCustomerRecommendation.objects.bulk_create(predictions)
        print("Save completed!")

    def _get_customer_features(self, customer):
        """Create feature vector for new customer"""
        # Create a DataFrame with matching feature names
        new_data = pd.DataFrame([[customer.country, customer.gender]],
                                columns=['country', 'gender'])

        encoded = self.predictor.encoder.transform(new_data)

        return np.concatenate([
            encoded[0],
            [customer.age, customer.income],
            np.zeros(len(self.predictor.item_list))
        ])

    def _create_prediction_records(self, customer, indices):
        """Generate prediction records from similar customers"""
        print(f"\nProcessing customer: {customer.UserId}")
        print(f"Neighbor indices received: {indices}")
        # Convert numpy indices to Python integers
        indices = [int(idx) for idx in indices]

        # Get existing customer IDs from the prepared features
        existing_customers = CustomerDetail.objects.filter(existing_customer=True)
        neighbor_ids = [existing_customers[idx].id for idx in indices]

        # Get neighbor customers and their transactions
        neighbor_customers = CustomerDetail.objects.filter(
            id__in=neighbor_ids
        ).prefetch_related('transactions')

        # Collect items from neighbors
        all_items = []
        for neighbor in neighbor_customers:
            for t in neighbor.transactions.all():
                # Explicitly verify transaction structure
                if not all([t.ItemCode, t.ItemDescription, t.CostPerItem]):
                    print(f"Skipping invalid transaction {t.TransactionId}")
                    continue

                all_items.append((
                    str(t.ItemCode).strip(),
                    str(t.ItemDescription).strip(),
                    float(t.CostPerItem)
                ))

            # Add debug print
        print(f"First 3 items sample: {all_items[:3]}")
        print(f"Total items collected from neighbors: {len(all_items)}")
        # Get unique items with their latest details
        item_details = {}
        for code, desc, cost in all_items:
            item_details[code] = (desc, cost)

        # Get most common items (top 3)
        item_counts = pd.Series(all_items).value_counts().head(3)
        print(f"Item counts:\n{item_counts}")

        # Create prediction records
        predictions = []
        for item_code, count in item_counts.items():
            print(f"Processing item {item_code} (count: {count})")
            desc, cost = item_details.get(item_code, ("Unknown", 0.00))
            predictions.append(NewCustomerRecommendation(
                user=customer,
                item_code=item_code,
                recommendation_type='PL',  # 'PL' for Personalized
                confidence_score=(count / len(all_items)) * 100,
                generation_date=datetime.now(),
                expiry_date=datetime.now() + timedelta(days=30)  # adjust expiry date as needed
            ))
        print(f"Generated {len(predictions)} predictions for {customer.UserId}")
        return predictions