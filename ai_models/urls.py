from django.urls import path
from django.contrib import admin
from . import views
from django.views.generic import RedirectView

app_name = 'ai_models'  # Unique namespace for this app's admin
urlpatterns = [
    #path('top-30-sale-forecast/', views.top_30_sale_forecast, name='top_30_sale_forecast'),
    path('sales-forecast-dashboard/', views.sales_forecast_dashboard, name='sales_forecast_dashboard'),
    path('future-sales/', views.future_sale_prediction, name='future_sale_prediction'),
    path('generate-recommendations/', views.generate_recommendations, name='generate_recommendations'),
    path('check-task-status/', views.check_task_status, name='check_task_status'),
    path('new-customer-recommendations/', views.customer_recommendations, name='customer_recommendations'),
    path('ask/', views.ask_ai, name='ask_ai'),
]




