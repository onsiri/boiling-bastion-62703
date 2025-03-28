from django.utils import timezone
from django.db.models.signals import post_migrate
from django.dispatch import receiver

from ai_models.apps import AiModelsConfig


def my_post_migrate_handler(sender, **kwargs):
    # Use relative import
    from .models import Transaction

    # Create sample transaction if none exists
    Transaction.objects.get_or_create(
        transaction_id=1,
        defaults={
            'uploaded_at': timezone.now(),
            # Add other required fields for your model
            # Example: 'amount': 0.0, 'description': 'Initial transaction'
        }
    )


# Connect the signal using the app config class name
post_migrate.connect(my_post_migrate_handler, sender=AiModelsConfig)