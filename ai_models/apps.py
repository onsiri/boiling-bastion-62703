from django.apps import AppConfig
from django.db.models.signals import post_migrate
from django.dispatch import receiver

class AiModelsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "ai_models"

    def ready(self):
        # Connect the signal handler using a decorator
        from . import signals  # Import signals module to register handlers

