
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.views.generic.base import RedirectView
from django.conf.urls.static import static
urlpatterns = [
    path('admin/', admin.site.urls),
    path("accounts/", include("django.contrib.auth.urls")),
    path("", include("pages.urls")),
    path("accounts/", include("accounts.urls")),
    path("Products/", include("products.urls")),
    path('models/', include(('ai_models.urls', 'ai_models'), namespace='ai_models')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
if not settings.DEBUG:
    urlpatterns += [
        path('', RedirectView.as_view(url='https://boiling-bastion-62703-1fb7e4016adf.herokuapp.com/')),
    ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)