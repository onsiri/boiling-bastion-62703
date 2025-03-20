from django.shortcuts import render
from django.http import JsonResponse
from django.core import serializers

from ai_models.models import sale_forecast

def pivot_dashboard(request):
    # Fetch and serialize data directly in the view
    dataset = sale_forecast.objects.all()
    serialized_data = serializers.serialize('json', dataset)
    return render(request, 'pivot_dashboard.html', {
        'sales_data': serialized_data  # Pass data to template
    })
def pivot_data(request):
    dataset = sale_forecast.objects.all()
    data = serializers.serialize('json', dataset)
    return JsonResponse(data, safe=False)