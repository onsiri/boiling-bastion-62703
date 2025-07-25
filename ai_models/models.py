from django.db import models
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
    ds = models.DateField()
    prediction = models.DecimalField(max_digits=10, decimal_places=2)
    prediction_lower = models.DecimalField(max_digits=10, decimal_places=2)
    prediction_upper = models.DecimalField(max_digits=10, decimal_places=2)
    uploaded_at = models.DateTimeField(auto_now=True)
    accuracy_score = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)

    def __str__(self):
        return f"{self.ds}"


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
    TransactionId = models.CharField(max_length=255)
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
    item_code = models.CharField(max_length=1000)
    recommendation_type = models.CharField(
        max_length=2,
        choices=RecommendationType.choices,
        default=RecommendationType.PERSONALIZED
    )
    confidence_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(100.0)]
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


    class BulkUpload(models.Model):
        s3_key = models.CharField(max_length=255)
        uploaded_at = models.DateTimeField(auto_now_add=True)
        processed = models.BooleanField(default=False)

class CountrySaleForecast(models.Model):
    class Meta:
        indexes = [
            models.Index(fields=['group', 'ds']),
            models.Index(fields=['ds'])
        ]
    group = models.CharField(max_length=100)  # Country name
    ds = models.DateField()
    prediction = models.FloatField()
    prediction_lower = models.FloatField()
    prediction_upper = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

class ItemSaleForecast(models.Model):
    ds = models.DateField()
    group = models.CharField(max_length=200)
    prediction = models.FloatField()
    prediction_lower = models.FloatField()
    prediction_upper = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'ai_models_itemsaleforecast'
        constraints = [
            models.UniqueConstraint(
                name='unique_group_ds',
                fields=['group', 'ds'],
                violation_error_message='Duplicate forecast entry'
            )
        ]
        indexes = [
            models.Index(fields=['group', 'ds']),
            models.Index(fields=['ds'])
        ]
