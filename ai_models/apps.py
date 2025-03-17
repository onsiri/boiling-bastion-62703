from django.apps import AppConfig
from django.db.models.signals import post_migrate
from django.dispatch import receiver

class AiModelsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ai_models'

    def ready(self):
        # Use signals to defer database access
        post_migrate.connect(self.my_post_migrate_handler, sender=self)

    @receiver(post_migrate)
    def my_post_migrate_handler(sender, **kwargs):
        # Your database access code here
        from .models import Transaction
        Transaction.objects.create(name="Initial Data")
