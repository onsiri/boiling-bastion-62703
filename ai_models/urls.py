from django.urls import path
from django.contrib import admin
from . import views

from django.conf import settings
from django.conf.urls.static import static
app_name = 'ai_models_admin'  # Unique namespace for this app's admin
urlpatterns = [
    path('admin/', admin.site.urls),
    path('top-30-sale-forecast/', views.top_30_sale_forecast, name='top_30_sale_forecast'),
    path('future_sale/', views.future_sale_prediction, name='future_sale'),
]

