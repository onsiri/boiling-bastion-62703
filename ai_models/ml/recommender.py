import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ..models import CustomerDetail, Transaction


class PurchasePredictor:
    def __init__(self):
        self.model = None
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.item_list = []
        self.scaler = StandardScaler()
        self.customer_ids = []
    def prepare_data(self):
        """Prepares customer data with consistent 2D arrays"""
        try:
            # Get existing customers with transactions
            existing_customers = CustomerDetail.objects.filter(
                existing_customer='Yes'
            ).prefetch_related('transactions')

            if not existing_customers.exists():
                print("No existing customers found")
                return None

            # Collect data
            customer_data = []
            self.customer_ids = []
            all_items = set()

            for cust in existing_customers:
                items = [t.ItemCode for t in cust.transactions.all()]
                customer_data.append({
                    'country': cust.country,
                    'gender': cust.gender,
                    'age': cust.age,
                    'income': cust.income,
                    'items': items
                })
                all_items.update(items)

            # Create item mapping
            self.item_list = sorted(all_items)
            item_index = {item: idx for idx, item in enumerate(self.item_list)}

            # Encode features
            country_gender_df = pd.DataFrame(customer_data)[['country', 'gender']]
            encoded = self.encoder.fit_transform(country_gender_df)

            # Create feature matrix
            features = []
            for cust in customer_data:
                item_vec = np.zeros(len(self.item_list))
                for item in cust['items']:
                    item_vec[item_index[item]] += 1

                features.append(np.concatenate([
                    encoded[len(features)],
                    [cust['age'], cust['income']],
                    item_vec
                ]))
            print(f"Data preparation complete")
            return np.array(features)

        except Exception as e:
            print(f"Data preparation failed: {str(e)}")
            return None

    def train_model(self, features):
        """Trains the recommendation model"""
        if features is None or len(features) == 0:
            print("Skipping training - no valid data")
            return

        print(f"Training on {len(features)} samples")
        self.model = NearestNeighbors(
            n_neighbors=10,
            metric='cosine',
            algorithm='auto'
        )
        self.model.fit(features)
        print("Model training completed")

    def predict(self, customer_features):
        """Generates predictions for new customers"""
        if self.model is None:
            return []

        print(" Finding nearest neighbors...")
        _, indices = self.model.kneighbors([customer_features])
        print(f" Found {len(indices[0])} neighbors")
        return indices[0]