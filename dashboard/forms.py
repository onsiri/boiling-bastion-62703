from django import forms

PLOT_CHOICES = [
    ('line', 'Line Chart'),
    ('bar', 'Bar Chart'),
    ('scatter', 'Scatter Plot'),
    ('pie', 'Pie Chart'),
]

class PlotTypeForm(forms.Form):
    plot_type = forms.ChoiceField(
        choices=PLOT_CHOICES,
        initial='line',
        widget=forms.Select(attrs={'class': 'form-control'})
    )