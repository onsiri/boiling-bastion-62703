from django.apps import AppConfig
from django.db.models.signals import post_migrate
from django.dispatch import receiver

class AiModelsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ai_models'

    def ready(self):
        # Use signals to defer database access
        post_migrate.connect(self.my_post_migrate_handler, sender=self)

    def my_post_migrate_handler(sender, **kwargs):
        pass  # Remove this initialization logic


