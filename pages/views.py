from django.shortcuts import render

from django.views.generic import TemplateView, CreateView
from .models import Ticket

class HomePageView(TemplateView):
    template_name = "home.html"

class AboutPageView(TemplateView):
    template_name = "about.html"

    def get_context_data(self, **kwargs):  # new
        context = super().get_context_data(**kwargs)
        context["contact_address"] = "123 Main Street"
        context["phone_number"] = "555-555-5555"
        return context
class BlogCreateView(CreateView):  # new
    model = Ticket
    template_name = "contact_us.html"
    fields = ["title", "author", "body"]