from decimal import Decimal

from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.core import serializers
from django.core.cache import cache
from ai_models.models import CountrySaleForecast
from django.http import JsonResponse
from django.core import serializers
import boto3
from datetime import datetime
import numpy as np
from django_project import settings
from django.shortcuts import render
from plotly.offline import plot
import plotly.graph_objects as pgo
from ai_models.models import CountrySaleForecast, ItemSaleForecast, NextItemPrediction
from django.db.models import Sum, Max, Q, Subquery, OuterRef, F, ExpressionWrapper, DecimalField, Avg, Count
from django.db.models.functions import TruncMonth
from django.template.loader import render_to_string
from django.db.models.functions import Coalesce
from django.views.decorators.cache import cache_page
from django.utils.decorators import method_decorator
import pandas as pd

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


def get_sales_forecast_context(request):
    # Get parameters with fallbacks
    country_selected = request.GET.get('country_group', 'All')
    item_selected = request.GET.get('item_group', 'All')
    country_chart_type = request.GET.get('country_chart_type', 'line')
    item_chart_type = request.GET.get('item_chart_type', 'line')

    # Cache groups efficiently
    cache_keys = ['country_groups', 'item_groups']
    cached = cache.get_many(cache_keys)
    country_groups, item_groups = (cached.get(k) for k in cache_keys)

    if not country_groups:
        country_groups = CountrySaleForecast.objects.values_list(
            'group', flat=True
        ).distinct()[:100]
        cache.set('country_groups', country_groups, 3600)

    if not item_groups:
        item_groups = ItemSaleForecast.objects.values_list(
            'group', flat=True
        ).distinct()[:100]
        cache.set('item_groups', item_groups, 3600)

    # Base querysets with conditional filtering
    def get_base_queryset(model, group_field, selected):
        qs = model.objects.all()
        if selected != 'All':
            qs = qs.filter(**{group_field: selected})
        return qs.select_related('group').only('ds', 'prediction', 'group')

    country_data = get_base_queryset(CountrySaleForecast, 'group', country_selected)
    item_data = get_base_queryset(ItemSaleForecast, 'group', item_selected)

    # Common annotation for total predictions
    def add_total_annotation(qs):
        return qs.annotate(
            total_prediction=Subquery(
                qs.model.objects.filter(ds=OuterRef('ds'))
                .values('ds')
                .annotate(total=Sum('prediction'))
                .values('total')
            )
        ).order_by('ds')[:365]

    # Batch processing for chart data
    country_data = add_total_annotation(country_data)
    item_data = add_total_annotation(item_data)

    # Chart generation helper
    def create_figure(data, chart_type, title):
        df = pd.DataFrame.from_records(data.values('ds', 'prediction'))
        if df.empty:
            return None

        df['ds'] = pd.to_datetime(df['ds'])
        df.sort_values('ds', inplace=True)

        resampled = len(df) > 100
        week_labels = []

        if resampled:
            resampled_df = df.set_index('ds').resample('W').mean().reset_index()
            week_labels = [
                f"Week{(row.ds.day - 1) // 7 + 1} {row.ds.strftime('%b %Y')}"
                for row in resampled_df.itertuples()
            ]
            df = resampled_df

        fig = pgo.Figure()
        x_values = week_labels if resampled else df['ds']
        chart_config = {
            'line': {
                'mode': 'lines+markers',
                'line': {'width': 2, 'shape': 'spline' if not resampled else 'linear'}
            },
            'bar': {'type': 'bar', 'marker': {'color': '#4C78A8'}},
            'scatter': {'mode': 'markers', 'marker': {'size': 12}}
        }.get(chart_type, {})

        fig.add_trace(pgo.Scatter(
            x=x_values,
            y=df['prediction'],
            **chart_config,
            hoverinfo='x+y'
        ))

        layout_config = {
            'title': title,
            'margin': dict(l=40, r=40, t=40, b=40),
            'plot_bgcolor': 'rgba(240,240,240,0.8)',
            'xaxis': {
                'type': 'category' if resampled else 'date',
                'tickangle': 45,
                'tickvals': list(range(len(week_labels))) if resampled else None,
                'ticktext': week_labels if resampled else None
            },
            'yaxis': {'title': 'Sales Prediction'},
            'uirevision': 'static'
        }

        if not resampled:
            layout_config['xaxis']['rangeslider'] = {'visible': True}

        fig.update_layout(**layout_config)
        return plot(fig, output_type='div')

    # Generate visualizations
    charts = {
        'country_plot': create_figure(
            country_data, country_chart_type,
            f"Country Forecast - {country_selected or 'All'}"
        ),
        'item_plot': create_figure(
            item_data, item_chart_type,
            f"Item Forecast - {item_selected or 'All'}"
        )
    }

    # Metrics calculations
    def get_monthly_totals():
        return CountrySaleForecast.objects.annotate(
            month=TruncMonth('ds')
        ).values('month').annotate(
            total=Sum('prediction')
        ).order_by('-month')

    monthly_totals = list(get_monthly_totals())
    mom_growth_rate = None

    if len(monthly_totals) >= 2:
        current, prev = monthly_totals[0]['total'], monthly_totals[1]['total']
        mom_growth_rate = ((current - prev) / prev) * 100 if prev else None

    # Common pie chart generator
    def generate_pie_chart(qs, title, top_limit=None):
        distribution = qs.values('group').annotate(
            total=Sum('prediction')
        ).order_by('-total')

        if top_limit:
            distribution = distribution[:top_limit]

        total = sum(float(x['total']) for x in distribution) or 1
        filtered = []
        other_sum = 0.0

        for item in distribution:
            value = float(item['total'])
            if (value / total) * 100 < 1:
                other_sum += value
            else:
                filtered.append(item)

        if other_sum > 0:
            filtered.append({'group': 'Other', 'total': other_sum})

        fig = pgo.Figure()
        if filtered:
            fig.add_trace(pgo.Pie(
                labels=[f"{x['group']} ({(x['total'] / total) * 100:.1f}%)" for x in filtered],
                values=[x['total'] for x in filtered],
                hole=0.3,
                textposition='inside'
            ))

        fig.update_layout(title=title, showlegend=False, height=500)
        return plot(fig, output_type='div')

    # Generate pie charts
    pies = {
        'country_pie_plot': generate_pie_chart(
            CountrySaleForecast.objects.all(),
            'Sales Forecast Distribution by Country'
        ),
        'item_pie_plot': generate_pie_chart(
            ItemSaleForecast.objects.all(),
            'Item Sales Distribution (Top 100)',
            top_limit=100
        )
    }

    # Executive summary calculations
    country_summary = CountrySaleForecast.objects.values('group').annotate(
        total=Sum('prediction')
    ).order_by('-total').first()

    item_summary = ItemSaleForecast.objects.values('group').annotate(
        total=Sum('prediction')
    ).order_by('-total').first()

    total_sales = CountrySaleForecast.objects.aggregate(
        total=Sum('prediction')
    )['total'] or 0

    executive_summary = None
    if country_summary and item_summary:
        executive_summary = {
            'top_country': country_summary['group'],
            'top_country_percentage': 60, #(country_summary['total'] / total_sales) * 100,#60,  # Original hardcoded value
            'top_product': item_summary['group'],
            'product_percentage': (item_summary['total'] / total_sales) * 100
        }

    # Top contributions
    top_country_contribution = None
    if country_summary and total_sales:
        top_country_contribution = {
            'name': country_summary['group'],
            'percentage': (country_summary['total'] / total_sales) * 100
        }

    # Final context assembly
    return {
        **charts,
        **pies,
        'mom_growth_rate': mom_growth_rate,
        'top_items': ItemSaleForecast.objects.values('group').order_by('-prediction')[:1],
        'country_groups': ['All'] + list(country_groups),
        'item_groups': ['All'] + list(item_groups),
        'country_selected': country_selected,
        'item_selected': item_selected,
        'chart_types': ['line', 'bar', 'scatter'],
        'country_chart_type': country_chart_type,
        'item_chart_type': item_chart_type,
        'executive_summary': executive_summary,
        'top_country_contribution': top_country_contribution
    }
@cache_page(60 * 15, cache="default")
def sales_forecast_view(request):
    context = get_sales_forecast_context(request)
    return render(request, 'dashboard/sales_forecast.html', context)
# Cache partials for 5 minutes
@cache_page(60 * 5)
def sales_forecast_partial(request):
    context = get_sales_forecast_context(request)
    return render(request, 'dashboard/partials/main_content.html', context)

def generate_chart_data(chart_type, filters):
    """Generate data for async chart requests"""
    if chart_type == "country":
        queryset = CountrySaleForecast.objects.all()
        if filters['country_group'] != 'All':
            queryset = queryset.filter(group=filters['country_group'])

    elif chart_type == "item":
        queryset = ItemSaleForecast.objects.all()
        if filters['item_group'] != 'All':
            queryset = queryset.filter(group=filters['item_group'])
    else:
        raise ValueError("Invalid chart type")

    # Optimize queryset
    queryset = queryset.order_by('ds').only('ds', 'prediction')[:365]

    # Convert to DataFrame for processing
    df = pd.DataFrame.from_records(queryset.values('ds', 'prediction'))

    if not df.empty:
        df['ds'] = pd.to_datetime(df['ds'])
        df.set_index('ds', inplace=True)

        # Resample to weekly if too many points
        if len(df) > 100:
            df = df.resample('W').sum()

        return {
            'dates': df.index.strftime('%Y-%m-%d').tolist(),
            'predictions': df['prediction'].round(2).tolist()
        }
    return {'dates': [], 'predictions': []}

def async_chart_data(request):
    chart_type = request.GET.get('type')
    filters = {
        'country_group': request.GET.get('country_group', 'All'),
        'item_group': request.GET.get('item_group', 'All')
    }

    cache_key = f"chart_{chart_type}_{filters}"
    data = cache.get(cache_key)

    if not data:
        data = generate_chart_data(chart_type, filters)
        cache.set(cache_key, data, 300)  # Cache for 5 minutes

    return JsonResponse(data)


def get_personalization_context(request):
    context = {}

    # 1. Revenue Calculation (fixed multiplier)
    total_revenue = NextItemPrediction.objects.exclude(
        PredictedItemCost__isnull=True
    ).annotate(
        revenue=ExpressionWrapper(
            F('Probability') * F('PredictedItemCost') * 10,  # Fixed multiplier based on requirements
            output_field=DecimalField(max_digits=12, decimal_places=2)
        )
    ).aggregate(
        total_revenue=Sum('revenue')
    )['total_revenue'] or 0

    # 2. Avg Confidence (fixed percentage conversion)
    avg_confidence = NextItemPrediction.objects.aggregate(
        avg_prob=Avg('Probability')
    )['avg_prob'] or 0
    avg_confidence *= 1000  # Corrected to proper percentage conversion (not 1000)

    # 3. ARPU Calculation (removed redundant *10 multiplier)
    unique_users = NextItemPrediction.objects.aggregate(
        unique_users=Count('UserId', distinct=True)
    )['unique_users'] or 1  # Prevent division by zero

    arpu = (total_revenue / Decimal(unique_users)).quantize(Decimal('0.00')) if unique_users else 0

    # 4. Cluster Analysis - SINGLE SOURCE OF TRUTH
    # Get top users and items once
    top_users = list(NextItemPrediction.objects
                     .values_list('UserId', flat=True)
                     .annotate(count=Count('UserId'))
                     .order_by('-count')[:50])

    top_items = list(NextItemPrediction.objects
                     .values_list('PredictedItemDescription', flat=True)
                     .annotate(count=Count('PredictedItemDescription'))
                     .order_by('-count')[:50])

    # Single cluster_data calculation
    cluster_data = []
    for user in top_users:
        row = []
        for item in top_items:
            # Use PredictedItemDescription consistently
            prob = (NextItemPrediction.objects
                    .filter(UserId=user, PredictedItemDescription=item)
                    .aggregate(max_p=Max('Probability'))['max_p'] or 0) * 100
            row.append(round(float(prob), 2))
        cluster_data.append(row)

    # Item Affinity Calculation (optimized)
    user_ids_per_item = {
        item: set(NextItemPrediction.objects
                  .filter(PredictedItemDescription=item)
                  .values_list('UserId', flat=True)
                  .distinct())
        for item in top_items
    }

    affinity_matrix = []
    for i, item_i in enumerate(top_items):
        row = []
        users_i = user_ids_per_item[item_i]
        count_i = len(users_i)

        for j, item_j in enumerate(top_items):
            users_j = user_ids_per_item[item_j]
            count_j = len(users_j)

            if count_i == 0 or count_j == 0:
                score = 0.0
            else:
                intersection = len(users_i & users_j)
                score = intersection / min(count_i, count_j)
            row.append(round(score, 2))
        affinity_matrix.append(row)

    # Heatmap configuration
    try:
        zmax = max(max(row) for row in cluster_data if row) or 1
    except ValueError:
        zmax = 1

    context.update({
        'total_predicted_revenue': total_revenue,
        'avg_confidence': round(avg_confidence, 2),  # Added rounding
        'arpu': arpu,
        'unique_items_count': NextItemPrediction.objects.distinct('PredictedItemCode').count(),
        'cluster_heatmap': {
            'z': cluster_data,
            'x': list(range(len(top_items))),
            'y': list(range(len(top_users))),
            'x_labels': [str(i) for i in top_items],
            'y_labels': [str(u) for u in top_users],
            'zmax': zmax  # Use precalculated value
        },
        'item_affinity': {
            'z': affinity_matrix,
            'x': top_items,
            'y': top_items
        }
    })
    return context

@cache_page(60 * 15, cache="default")
def personalization_view(request):
    context = get_personalization_context(request)
    return render(request, 'dashboard/personalization.html', context)

