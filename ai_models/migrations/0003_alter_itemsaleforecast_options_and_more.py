# Generated by Django 5.1.7 on 2025-04-22 20:43

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ai_models', '0002_remove_nextitemprediction_unique_user_id_and_predicted_at'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='itemsaleforecast',
            options={'managed': False},
        ),
        migrations.AlterField(
            model_name='newcustomerrecommendation',
            name='confidence_score',
            field=models.FloatField(validators=[django.core.validators.MinValueValidator(0.0), django.core.validators.MaxValueValidator(100.0)]),
        ),
        migrations.AlterField(
            model_name='newcustomerrecommendation',
            name='item_code',
            field=models.CharField(max_length=1000),
        ),
    ]
