from django.urls import path
from . import views

urlpatterns = [
    path('', views.pivot_dashboard, name='pivot_dashboard'),
    path('data/', views.pivot_data, name='pivot_data'),
]