from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.core import serializers

from ai_models.models import CountrySaleForecast
from django.http import JsonResponse
from django.core import serializers
import boto3
from datetime import datetime

from django_project import settings
from django.shortcuts import render
from plotly.offline import plot
import plotly.graph_objects as pgo
from ai_models.models import CountrySaleForecast, ItemSaleForecast
from django.db.models import Sum
from django.db.models.functions import TruncMonth

def export_to_s3(request):
    # Serialize data
    sales_data = serializers.serialize('json', CountrySaleForecast.objects.all())

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
    sales_data_dataset = CountrySaleForecast.objects.all()
    serialized_data = serializers.serialize('json', sales_data_dataset)
    return render(request, 'pivot_dashboard.html', {
        'sales_data': serialized_data  # Pass data to template
    })
def pivot_data(request):
    dataset = CountrySaleForecast.objects.all()
    data = serializers.serialize('json', dataset)
    return JsonResponse(data, safe=False)

def dash_view(request):
    return render(request, 'sales_dash/dash.html')

from django_plotly_dash.models import StatelessApp

def debug_view(request):
    apps = StatelessApp.objects.all()
    return HttpResponse(f"Registered Dash Apps: {list(apps.values_list('slug', flat=True))}")


def sales_forecast_view(request):
    # Country Forecast Data
    country_groups = CountrySaleForecast.objects.values_list('group', flat=True).distinct()
    country_selected = request.GET.get('country_group', 'All')
    country_chart_type = request.GET.get('country_chart_type', 'line')

    # Item Forecast Data
    item_groups = ItemSaleForecast.objects.values_list('group', flat=True).distinct()
    item_selected = request.GET.get('item_group', 'All')
    item_chart_type = request.GET.get('item_chart_type', 'line')

    def create_figure(data, chart_type, title):
        fig = pgo.Figure()

        dates = [item.ds for item in data]
        predictions = [float(item.prediction) for item in data]
        lowers = [float(item.prediction_lower) for item in data]
        uppers = [float(item.prediction_upper) for item in data]

        if chart_type == 'line':
            fig.add_trace(pgo.Scatter(x=dates, y=predictions, mode='lines+markers', name='Prediction'))
            fig.add_trace(pgo.Scatter(
                x=dates + dates[::-1],
                y=uppers + lowers[::-1],
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            ))
        elif chart_type == 'bar':
            fig.add_trace(pgo.Bar(x=dates, y=predictions, name='Prediction'))
        elif chart_type == 'scatter':
            fig.add_trace(pgo.Scatter(x=dates, y=predictions, mode='markers', name='Prediction'))
        elif chart_type == 'pie':
            fig.add_trace(pgo.Pie(
                labels=dates,
                values=predictions,
                textinfo='label+percent',
                insidetextorientation='radial'
            ))

        fig.update_layout(title=title)
        return plot(fig, output_type='div')

    # Country Chart
    country_data = CountrySaleForecast.objects.all()
    if country_selected != 'All':
        country_data = country_data.filter(group=country_selected)
    country_plot = create_figure(
        country_data.order_by('ds'),
        country_chart_type,
        f"Country Forecast - {country_selected or 'All'}"
    )

    # Item Chart
    item_data = ItemSaleForecast.objects.all()
    if item_selected != 'All':
        item_data = item_data.filter(group=item_selected)
    item_plot = create_figure(
        item_data.order_by('ds'),
        item_chart_type,
        f"Item Forecast - {item_selected or 'All'}"
    )
    # Calculate MoM Growth Rate for Countries
    monthly_totals = CountrySaleForecast.objects.annotate(
        month=TruncMonth('ds')
    ).values('month').annotate(
        total_prediction=Sum('prediction')
    ).order_by('-month')

    mom_growth_rate = None
    if len(monthly_totals) >= 2:
        current_month = monthly_totals[0]['total_prediction']
        previous_month = monthly_totals[1]['total_prediction']
        if previous_month != 0:  # Avoid division by zero
            mom_growth_rate = ((current_month - previous_month) / previous_month) * 100

    top_items = ItemSaleForecast.objects.values('group').annotate(
        total_sales=Sum('prediction')
    ).order_by('-total_sales')[:3]
    context = {
        'country_plot': country_plot,
        'item_plot': item_plot,
        'country_groups': ['All'] + list(country_groups),
        'item_groups': ['All'] + list(item_groups),
        'country_selected': country_selected,
        'item_selected': item_selected,
        'chart_types': ['line', 'bar', 'scatter', 'pie'],
        'country_chart_type': country_chart_type,
        'item_chart_type': item_chart_type,
    }
    context.update({
        'mom_growth_rate': mom_growth_rate,
        'top_items': top_items,
    })
    return render(request, 'dashboard/sales_forecast.html', context)