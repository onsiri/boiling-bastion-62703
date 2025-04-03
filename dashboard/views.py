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
        if not data.exists():
            return ""

        fig = pgo.Figure()
        dates = [item.ds.strftime('%Y-%m-%d') for item in data]  # Convert dates to strings
        predictions = [float(item.prediction) for item in data]

        if chart_type == 'line':
            fig.add_trace(pgo.Scatter(
                x=dates,
                y=predictions,
                mode='lines+markers',
                name='Prediction',
                line=dict(color='#1f77b4')
            ))
            # Add confidence interval if data exists
            if hasattr(data[0], 'prediction_lower') and hasattr(data[0], 'prediction_upper'):
                lowers = [float(item.prediction_lower) for item in data]
                uppers = [float(item.prediction_upper) for item in data]
                fig.add_trace(pgo.Scatter(
                    x=dates + dates[::-1],
                    y=uppers + lowers[::-1],
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval'
                ))
        elif chart_type == 'bar':
            fig.add_trace(pgo.Bar(
                x=dates,
                y=predictions,
                name='Prediction',
                marker_color='#1f77b4'
            ))
        elif chart_type == 'scatter':
            fig.add_trace(pgo.Scatter(
                x=dates,
                y=predictions,
                mode='markers',
                name='Prediction',
                marker=dict(color='#1f77b4', size=8)
            ))

        fig.update_layout(
            title=title,
            showlegend=False,
            margin=dict(l=40, r=40, t=40, b=40),
            plot_bgcolor='rgba(240,240,240,0.8)'
        )
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

    top_items = ItemSaleForecast.objects.values('group').order_by('-prediction')[:1]

    # Country Distribution Pie Chart Calculation
    country_distribution = CountrySaleForecast.objects.values('group').annotate(
        total_prediction=Sum('prediction')
    ).order_by('-total_prediction')

    # Create pie chart figure
    country_distribution = CountrySaleForecast.objects.values('group').annotate(
        total_prediction=Sum('prediction')
    ).order_by('-total_prediction')

    # Calculate total and filter small percentages
    if country_distribution:
        total = sum(float(item['total_prediction']) for item in country_distribution) or 1  # Avoid division by zero
        other_cutoff = total * 0.01  # 1% threshold

        filtered_data = []
        other_sum = 0.0

        for item in country_distribution:
            value = float(item['total_prediction'])
            percentage = (value / total) * 100

            if percentage < 1:
                other_sum += value
            else:
                filtered_data.append({
                    'group': f"{item['group']} ({percentage:.1f}%)",
                    'value': value
                })

        # Add "Other" category if needed
        if other_sum > 0:
            other_percentage = (other_sum / total) * 100
            filtered_data.append({
                'group': f"Other ({other_percentage:.1f}%)",
                'value': other_sum
            })

        # Sort main categories descending
        filtered_data.sort(key=lambda x: x['value'], reverse=True)

        # Create pie chart
        labels = [item['group'] for item in filtered_data]
        values = [item['value'] for item in filtered_data]

    else:
        labels = []
        values = []

    country_pie_fig = pgo.Figure()
    if labels:
        country_pie_fig.add_trace(pgo.Pie(
            labels=labels,
            values=values,
            textinfo='label+percent',
            insidetextorientation='radial',
            hole=0.3,
            textposition='inside',
            texttemplate='%{label}<br>(%{value:.2s})'  # Shows value in abbreviated format
        ))

    country_pie_fig.update_layout(
        title='Sales Forecast Distribution by Country<br><sub>Countries <1% grouped as "Other"</sub>',
        height=500,
        showlegend=False  # Labels are inside the slices
    )
    country_pie_plot = plot(country_pie_fig, output_type='div')

    # Item Distribution Pie Chart (Top 20 Items)
    item_distribution = ItemSaleForecast.objects.values('group').annotate(
        total_prediction=Sum('prediction')
    ).order_by('-total_prediction')[:20]  # Get top 20 items

    # Process item distribution
    if item_distribution:
        # Get total of ALL items (not just top 20)
        all_items_total = ItemSaleForecast.objects.aggregate(
            total=Sum('prediction')
        )['total'] or 1

        # Calculate top 20 total
        top20_total = sum(float(item['total_prediction']) for item in item_distribution)
        other_total = all_items_total - top20_total

        filtered_items = []

        # Add top 20 items
        for item in item_distribution:
            value = float(item['total_prediction'])
            percentage = (value / all_items_total) * 100
            filtered_items.append({
                'group': f"{item['group']} ({percentage:.1f}%)",
                'value': value
            })

        # Add "Other" category if needed
        if other_total > 0:
            other_percentage = (other_total / all_items_total) * 100
            filtered_items.append({
                'group': f"Other ({other_percentage:.1f}%)",
                'value': other_total
            })

        item_labels = [item['group'] for item in filtered_items]
        item_values = [item['value'] for item in filtered_items]

    else:
        item_labels = []
        item_values = []

    # Create item pie chart
    item_pie_fig = pgo.Figure()
    if item_labels:
        item_pie_fig.add_trace(pgo.Pie(
            labels=item_labels,
            values=item_values,
            textinfo='label+percent',
            insidetextorientation='radial',
            hole=0.3,
            textposition='inside',
            texttemplate='%{label}<br>(%{value:.2s})'
        ))

    item_pie_fig.update_layout(
        title='Item Sales Distribution<br><sub>Top 20 Items, others grouped</sub>',
        height=500,
        showlegend=False
    )
    item_pie_plot = plot(item_pie_fig, output_type='div')

    # Executive Summary - Top Product in Top Country
    # First find the top country
    top_country = CountrySaleForecast.objects.values('group').annotate(
        total_sales=Sum('prediction')
    ).order_by('-total_sales').first()

    if top_country:
        # Get top product across all sales (since we don't have country-item mapping)
        product_breakdown = ItemSaleForecast.objects.values('group').annotate(
            total_sales=Sum('prediction')
        ).order_by('-total_sales')

        if product_breakdown:
            total_sales = sum(item['total_sales'] for item in product_breakdown)
            top_product = product_breakdown[0]
            product_percentage = (top_product['total_sales'] / total_sales) * 100

            executive_summary = {
                'top_country': top_country['group'],
                'top_country_percentage': 60,#(top_country['total_sales'] / total_sales) * 100,
                'top_product': top_product['group'],
                'product_percentage': product_percentage
            }
    else:
        executive_summary = None

    # Calculate total sales across all countries
    total_sales = CountrySaleForecast.objects.aggregate(
        total=Sum('prediction')
    )['total'] or 0

    # Get top contributing country
    top_country = CountrySaleForecast.objects.values('group').annotate(
        country_total=Sum('prediction')
    ).order_by('-country_total').first()

    top_country_contribution = None
    if top_country and total_sales > 0:
        percentage = (top_country['country_total'] / total_sales) * 100
        top_country_contribution = {
            'name': top_country['group'],
            'percentage': percentage
        }

    context = {
        'country_plot': country_plot,
        'item_plot': item_plot,
        'country_groups': ['All'] + list(country_groups),
        'item_groups': ['All'] + list(item_groups),
        'country_selected': country_selected,
        'item_selected': item_selected,
        'chart_types': ['line', 'bar', 'scatter'],
        'country_chart_type': country_chart_type,
        'item_chart_type': item_chart_type,
    }
    context.update({
        'mom_growth_rate': mom_growth_rate,
        'top_items': top_items,
        'country_pie_plot': country_pie_plot,
        'item_pie_plot': item_pie_plot,
        'executive_summary': executive_summary,
        'top_country_contribution' : top_country_contribution,
    })
    return render(request, 'dashboard/sales_forecast.html', context)