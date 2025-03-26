from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.core import serializers

from ai_models.models import sale_forecast
from django.http import JsonResponse
from django.core import serializers
import boto3
from datetime import datetime

from django_project import settings


def export_to_s3(request):
    # Serialize data
    sales_data = serializers.serialize('json', sale_forecast.objects.all())

    # Configure S3 client
    s3 = boto3.client('s3',
                      aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY)

    # Create timestamped filename
    filename = f"sales_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    try:
        s3.put_object(
            Bucket=settings.AWS_STORAGE_BUCKET_NAME,
            Key=f"raw_data/{filename}",
            Body=sales_data,
            ContentType='application/json'
        )
        return JsonResponse({'status': 'success', 'filename': filename})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})
def pivot_dashboard(request):
    # Fetch and serialize data directly in the view
    sales_data_dataset = sale_forecast.objects.all()
    serialized_data = serializers.serialize('json', sales_data_dataset)
    return render(request, 'pivot_dashboard.html', {
        'sales_data': serialized_data  # Pass data to template
    })
def pivot_data(request):
    dataset = sale_forecast.objects.all()
    data = serializers.serialize('json', dataset)
    return JsonResponse(data, safe=False)

def dash_view(request):
    return render(request, 'sales_dash/dash.html')

from django_plotly_dash.models import StatelessApp

def debug_view(request):
    apps = StatelessApp.objects.all()
    return HttpResponse(f"Registered Dash Apps: {list(apps.values_list('slug', flat=True))}")


from django.shortcuts import render
from plotly.offline import plot
import plotly.graph_objs as go
from ai_models.models import sale_forecast

from django.shortcuts import render
from plotly.offline import plot
import plotly.graph_objects as pgo
from ai_models.models import sale_forecast  # EXACT model name


def sales_forecast_view(request):
    # Get all unique countries for the dropdown
    countries = sale_forecast.objects.values_list('country', flat=True).distinct()
    selected_country = request.GET.get('country', 'All')
    chart_type = request.GET.get('chart_type', 'line')

    # Filter data based on selection
    if selected_country and selected_country != 'All':
        data = sale_forecast.objects.filter(country=selected_country).order_by('ds')
    else:
        data = sale_forecast.objects.all().order_by('ds')

    # Prepare data
    dates = [item.ds for item in data]
    predictions = [float(item.prediction) for item in data]
    lower_bounds = [float(item.prediction_lower) for item in data]
    upper_bounds = [float(item.prediction_upper) for item in data]

    # Create selected chart type
    fig = pgo.Figure()

    if chart_type == 'line':
        fig.add_trace(pgo.Scatter(
            x=dates, y=predictions,
            mode='lines+markers',
            name='Prediction'
        ))
    elif chart_type == 'bar':
        fig.add_trace(pgo.Bar(
            x=dates, y=predictions,
            name='Prediction'
        ))
    elif chart_type == 'scatter':
        fig.add_trace(pgo.Scatter(
            x=dates, y=predictions,
            mode='markers',
            name='Prediction'
        ))

    # Add confidence interval if line chart
    if chart_type == 'line':
        fig.add_trace(pgo.Scatter(
            x=dates + dates[::-1],  # x then reversed x
            y=upper_bounds + lower_bounds[::-1],  # upper then reversed lower
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))

    fig.update_layout(
        title=f'Sales Forecast - {selected_country}',
        xaxis_title='Date',
        yaxis_title='Sales'
    )

    plot_html = plot(fig, output_type='div')

    context = {
        'plot': plot_html,
        'countries': ['All'] + list(countries),
        'selected_country': selected_country,
        'chart_types': ['line', 'bar', 'scatter'],
        'selected_chart_type': chart_type
    }

    return render(request, 'sales_chart.html', context)