from django.core.management.base import BaseCommand
from ai_models.pipelines.prediction_pipeline import PredictionPipeline

class Command(BaseCommand):
    help = 'Generates purchase predictions for new customers'

    def handle(self, *args, **options):
        pipeline = PredictionPipeline()
        pipeline.run()
        self.stdout.write(self.style.SUCCESS('Successfully generated predictions'))