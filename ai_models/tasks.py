from celery import shared_task
from ai_models.utils import generate_forecast, predict_future_sales

@shared_task
def async_generate_forecast():
    generate_forecast()

@shared_task
def async_predict_future_sales():
    predict_future_sales(None)  # Pass None if no request object needed

@shared_task
def process_analytics_after_upload():
    async_generate_forecast.delay()
    async_predict_future_sales.delay()