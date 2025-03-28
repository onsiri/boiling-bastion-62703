from django.apps import AppConfig
from django.db.models.signals import post_migrate
from django.dispatch import receiver

class AiModelsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "ai_models"

    def ready(self):
        # Import signals inside ready() to avoid circular imports
        from . import signals  # This registers your signal handlers

