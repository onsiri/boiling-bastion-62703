from datetime import timezone

from django.dispatch import receiver
from django.db.models.signals import post_migrate
from .models import Transaction

@receiver(post_migrate, sender="ai_models")  # Only trigger for this app
def my_post_migrate_handler(sender, **kwargs):
    # Use valid fields from your Transaction model
    Transaction.objects.get_or_create(
        transaction_id=1,
        defaults={"uploaded_at": timezone.now()}
    )