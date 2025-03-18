from django.db.models import Max
from django.utils import timezone
from django.db.models.signals import post_save
from django.dispatch import receiver
from prophet import Prophet
import pandas as pd
from django.db import models
from datetime import datetime, timedelta
from django.core.validators import MinValueValidator, MaxValueValidator  # Add this import

class UploadBatch(models.Model):
    uploaded_at = models.DateTimeField(auto_now_add=True)

class Item(models.Model):
    class Meta:
        app_label = 'ai_models'
    ItemCode = models.CharField(max_length=50, unique=True)
    ItemDescription = models.CharField(max_length=200)
    CostPerItem = models.DecimalField(max_digits=10, decimal_places=2)
    uploaded_at = models.DateTimeField(auto_now=True, null=True)
    def __str__(self):
        return self.ItemCode

class sale_forecast(models.Model):
    ds =  models.CharField(max_length=500)
    prediction =  models.DecimalField(max_digits=10, decimal_places=2)
    prediction_lower =  models.DecimalField(max_digits=10, decimal_places=2)
    prediction_upper = models.DecimalField(max_digits=10, decimal_places=2)
    uploaded_at = models.DateTimeField(auto_now=True, null=True)

    def __str__(self):
        return self.ds


class CustomerDetail(models.Model):
    class Meta:
        app_label = 'ai_models'
    UserId = models.CharField(max_length=100, unique=True)
    existing_customer = models.BooleanField()
    country = models.CharField(max_length=100)
    age = models.IntegerField()
    gender = models.CharField(max_length=10)
    income = models.IntegerField()
    occupation = models.CharField(max_length=100)
    education_level = models.CharField(max_length=100)
    uploaded_at = models.DateTimeField(auto_now=True, null=True)
    def __str__(self):
        return f"{self.UserId} "

class Transaction(models.Model):
    class Meta:
        app_label = 'ai_models'
    user = models.ForeignKey(
        CustomerDetail,
        on_delete=models.CASCADE,
        related_name='transactions',
        db_column='UserId'
    )
    TransactionId = models.CharField(max_length=100)
    TransactionTime = models.CharField(max_length=500)
    ItemCode = models.CharField(max_length=100)
    ItemDescription = models.TextField()
    NumberOfItemsPurchased = models.IntegerField()
    CostPerItem = models.DecimalField(max_digits=10, decimal_places=2)
    Country = models.CharField(max_length=100)
    uploaded_at = models.DateTimeField(auto_now=True, null=True)

    def __str__(self):
        return f"{self.TransactionId} - {self.user.UserId}"


class NextItemPrediction(models.Model):
    class Meta:
        app_label = 'ai_models'
    UserId = models.CharField(max_length=100)
    PredictedItemCode = models.CharField(max_length=100)
    PredictedItemDescription = models.CharField(max_length=200, null=True)
    PredictedItemCost = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    Probability = models.FloatField()
    PredictedAt = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.UserId} - {self.PredictedItemCode} ({self.Probability:.2f})"

class NewCustomerRecommendation(models.Model):
    class RecommendationType(models.TextChoices):
        PERSONALIZED = 'PL', 'Personalized'
        TRENDING = 'TR', 'Trending'
        POPULAR = 'PP', 'Popular'

    user = models.ForeignKey(
        CustomerDetail,
        on_delete=models.CASCADE,
        related_name='recommendations',
        db_column='UserId'
    )
    item_code = models.CharField(max_length=100)
    recommendation_type = models.CharField(
        max_length=2,
        choices=RecommendationType.choices,
        default=RecommendationType.PERSONALIZED
    )
    confidence_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    generation_date = models.DateTimeField(auto_now_add=True)
    expiry_date = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Recommendation validity period"
    )
    recommendation_metadata = models.JSONField(
        default=dict,
        help_text="Stores model version, neighbor count, and feature weights"
    )

    class Meta:
        indexes = [
            models.Index(fields=['generation_date', '-confidence_score']),
            models.Index(fields=['item_code', 'recommendation_type']),
        ]
        ordering = ['-confidence_score']
        verbose_name = 'New Customer Recommendation'
        verbose_name_plural = 'New Customer Recommendations'

    def __str__(self):
        return f"{self.user.UserId} - {self.get_recommendation_type_display()} ({self.confidence_score:.0%})"