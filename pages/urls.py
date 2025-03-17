from django.urls import path

from .views import HomePageView, AboutPageView, BlogCreateView

urlpatterns = [
    path("", HomePageView.as_view(), name="home"),
    path("about/", AboutPageView.as_view(), name="about"),
    path("post/new/", BlogCreateView.as_view(), name="contact_us"),
]
