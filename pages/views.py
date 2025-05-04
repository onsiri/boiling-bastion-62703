from django.views.generic import TemplateView, CreateView, DetailView
from .models import Ticket
from .forms import TicketForm
from django.urls import reverse_lazy
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
    form_class = TicketForm  # Use your custom form
    template_name = "contact_us.html"
    def get_success_url(self):
        return reverse_lazy('ticket_detail', kwargs={'pk': self.object.pk})

class TicketDetailView(DetailView):
    model = Ticket
    template_name = "ticket_detail.html"