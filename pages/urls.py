from django.urls import path

from .views import HomePageView, AboutPageView, BlogCreateView, TicketDetailView

urlpatterns = [
    path("", HomePageView.as_view(), name="home"),
    path("about/", AboutPageView.as_view(), name="about"),
    path('contact/', BlogCreateView.as_view(), name='contact_us'),
    path('ticket/<int:pk>/', TicketDetailView.as_view(), name='ticket_detail'),
]
