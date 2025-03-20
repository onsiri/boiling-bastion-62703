from django.http import JsonResponse
from django.shortcuts import render
from ai_models.models import sale_forecast
from django.core import serializers

def dashboard_with_pivot(request):
    return render(request, 'dashboard_with_pivot.html', {})

def pivot_data(request):
    dataset = sale_forecast.objects.all()
    data = serializers.serialize('json', dataset)
    return JsonResponse(data, safe=False)