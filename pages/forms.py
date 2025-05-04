from django import forms
from .models import Ticket  # Import from current app

class TicketForm(forms.ModelForm):
    class Meta:
        model = Ticket
        fields = ['title', 'author', 'body']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control', 'style': 'max-width: 500px'}),
            'author': forms.TextInput(attrs={'class': 'form-control', 'style': 'max-width: 500px'}),
            'body': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 5,
                'style': 'min-width: 100%'
            }),
        }