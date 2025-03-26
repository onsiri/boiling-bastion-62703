from django.urls import path
from . import views
from .views import sales_forecast_view

urlpatterns = [
    path('', views.pivot_dashboard, name='pivot_dashboard'),
    #path('data/', views.pivot_data, name='pivot_data'),
    #path('sales-dashboard/', views.dash_view, name='dash-view'),
    #path('debug-dash-apps/', views.debug_view),
    path('sales-forecast/', sales_forecast_view, name='sales_forecast'),
    path('sales/', sales_forecast_view, name='sales_chart'),
]