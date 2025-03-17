# ai_models/signals.py
from django.db import transaction
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Transaction
from .utils import predict_future_sales
from django.db.models.signals import post_migrate
from django.dispatch import receiver


@receiver(post_save, sender=Transaction)
def schedule_prediction(sender, instance, created, **kwargs):
    if created:
        # Schedule predictions to run once after all rows are saved
        if not hasattr(transaction, 'prediction_scheduled'):
            transaction.prediction_scheduled = True
            transaction.on_commit(predict_future_sales)
@receiver(post_migrate)
def initialize_predictor(sender, **kwargs):
    if sender.name == 'myapp':
        from .ml.predictor import PurchasePredictor
        # Initialize without DB access
        PurchasePredictor()