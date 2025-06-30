from django.urls import path
from . import views
from .views import sales_forecast_view, personalization_view, new_customer_rec_view


app_name = 'dashboard'

urlpatterns = [
    path('', views.pivot_dashboard, name='pivot_dashboard'),
    path('sales-forecast/', sales_forecast_view, name='sales_forecast'),
    path('sales/', sales_forecast_view, name='sales_chart'),
    path('personalization/', personalization_view, name='personalization'),
    path('new_customer_rec/', new_customer_rec_view, name='new_customer_rec'),
    path('sales-forecast/partial/', views.sales_forecast_partial, name='sales_forecast_partial'),
]