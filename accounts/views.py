from django.shortcuts import render
from .decorators import allow_unauthenticated
from django.urls import reverse_lazy
from django.views import generic
from django import forms
from .user_forms import CustomUserCreationForm
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import CreateView, TemplateView

@allow_unauthenticated
class SignupPageView(generic.CreateView):
    form_class = CustomUserCreationForm
    success_url = reverse_lazy("login")
    template_name = "registration/signup.html"

# Add this new protected view (example)
class HomePageView(LoginRequiredMixin, TemplateView):
    template_name = "home.html"  # Requires authentication to access
