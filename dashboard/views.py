from decimal import Decimal
from django.db.models import Case, When, IntegerField
from django.utils import timezone
from datetime import timedelta
from django.http import HttpResponse
from django.core.cache import cache
from django.http import JsonResponse
from django.core import serializers
import boto3
from datetime import datetime
from ai_models.views import logger
from django_project import settings
from django.shortcuts import render
from plotly.offline import plot
import plotly.graph_objects as pgo
from ai_models.models import CountrySaleForecast, ItemSaleForecast, NextItemPrediction, sale_forecast, NewCustomerRecommendation, CustomerDetail
from django.db.models import Sum, Max, Q, Subquery, OuterRef, F, ExpressionWrapper, DecimalField, Avg, Count
from django.db.models.functions import TruncMonth
from django.views.decorators.cache import cache_page, never_cache
import pandas as pd
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.utils.safestring import mark_safe
import json

import logging
logger = logging.getLogger(__name__)

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
        queryset = model.objects.all()

        # Special case: Exclude Brazil for "All" countries selection
        if model == CountrySaleForecast and selected == "All":
            queryset = queryset.exclude(group="Brazil")  # Adjust field name if needed

        if selected != "All":
            queryset = queryset.filter(**{group_field: selected})

        return queryset.values("ds").annotate(
            prediction=Sum("prediction")
        ).order_by("ds")

    country_data = get_base_queryset(CountrySaleForecast, "group", country_selected)
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

        # Handle different chart types
        if chart_type == 'bar':
            fig.add_trace(pgo.Bar(
                x=x_values,
                y=df['prediction'],
                marker={'color': '#4C78A8'},
                hoverinfo='x+y'
            ))
            # Specific layout adjustments for bar charts
            fig.update_layout(
                yaxis=dict(
                    autorange=True,
                    rangemode='tozero'  # Ensure bars start from zero
                )
            )
        elif chart_type == 'scatter':
            fig.add_trace(pgo.Scatter(
                x=x_values,
                y=df['prediction'],
                mode='markers',
                marker={'size': 12},
                hoverinfo='x+y'
            ))
        else:  # Default to line chart
            fig.add_trace(pgo.Scatter(
                x=x_values,
                y=df['prediction'],
                mode='lines+markers',
                line={'width': 2, 'shape': 'spline' if not resampled else 'linear'},
                hoverinfo='x+y'
            ))

        # Common layout settings
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

        # Create a new figure
        fig = pgo.Figure()

        x_values = week_labels if resampled else df['ds']

        # The key fix: Create different trace types based on chart_type
        if chart_type == 'line':
            fig.add_trace(pgo.Scatter(
                x=x_values,
                y=df['prediction'],
                mode='lines+markers',
                line={'width': 2, 'shape': 'spline' if not resampled else 'linear'},
                hoverinfo='x+y'
            ))
        elif chart_type == 'bar':
            fig.update_layout(
                yaxis=dict(
                    autorange=True,  # Force autorange for bar charts
                    rangemode='tozero'  # Ensure y-axis starts at zero for bars
                ))
        else:  # Default to scatter
            fig.add_trace(pgo.Scatter(
                x=x_values,
                y=df['prediction'],
                mode='markers',
                marker={'size': 12},
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


def get_combined_context(request):
    # Existing dashboard context
    dashboard_context = get_sales_forecast_context(request)

    # 30-day forecast context
    sort_by = request.GET.get('sort_by', 'ds')
    sort_order = request.GET.get('sort_order', 'desc')
    forecasts = sale_forecast.objects.all().order_by(f'{"-" if sort_order == "desc" else ""}{sort_by}')

    paginator = Paginator(forecasts, 30)
    page_number = request.GET.get('page', 1)

    try:
        page_obj = paginator.page(page_number)
    except (PageNotAnInteger, EmptyPage):
        page_obj = paginator.page(1)

    return {
        **dashboard_context,
        'page_obj': page_obj,
        'sort_by': sort_by,
        'sort_order': sort_order,
    }
@cache_page(60 * 15, cache="default")
def sales_forecast_view(request):
    context = get_combined_context(request)
    return render(request, 'dashboard/sales_forecast.html', context)

@cache_page(60 * 5)
def sales_forecast_partial(request):
    context = get_combined_context(request)
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
    avg_confidence *= 1000   # Corrected to proper percentage conversion (not 1000)

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


def split_item_code(item_code):
    """Clean parentheses and quotes from item codes"""
    clean_code = item_code.strip(" ()'")  # Remove surrounding parentheses/quotes
    return [part.strip(" '") for part in clean_code.split(',')]

COUNTRY_COORDINATES = {
    # Existing entries
    'United Kingdom': (55.3781, -3.4360),
    'Germany': (51.1657, 10.4515),
    'France': (46.6035, 1.8883),
    'Ireland': (53.4129, -8.2439),
    'EIRE': (53.1759, -8.1526),  # Ireland
    'Spain': (40.4637, -3.7492),
    'Netherlands': (52.1326, 5.2913),
    'Switzerland': (46.8182, 8.2275),
    'Belgium': (50.5039, 4.4699),
    'Portugal': (39.3999, -8.2245),
    'Australia': (-25.2744, 133.7751)
}

def get_new_customer_rec_context(request):
    context = {}
    now = timezone.now()

    # Existing metrics calculation
    total_recommendations = NewCustomerRecommendation.objects.count()

    # Average confidence score
    avg_confidence = NewCustomerRecommendation.objects.aggregate(
        avg_confidence=Avg('confidence_score')
    )['avg_confidence'] or 0
    avg_confidence_percent = round(avg_confidence * 100, 1)

    # Status counts
    status_counts = NewCustomerRecommendation.objects.aggregate(
        active=Count(
            Case(
                When(
                    Q(expiry_date__gt=now) |
                    Q(expiry_date__isnull=True, generation_date__gte=now - timezone.timedelta(days=30)),
                    then=1
                ),
                output_field=IntegerField()
            )
        ),
        expired=Count(
            Case(
                When(
                    Q(expiry_date__lte=now) |
                    Q(expiry_date__isnull=True, generation_date__lt=now - timezone.timedelta(days=30)),
                    then=1
                ),
                output_field=IntegerField()
            )
        )
    )

    # Most Recommended Item
    most_recommended = NewCustomerRecommendation.objects.values('item_code') \
        .annotate(count=Count('id')) \
        .order_by('-count') \
        .first()

    item_info = {'code': None, 'name': None, 'count': 0}
    if most_recommended:
        try:
            item_parts = most_recommended['item_code'].split(',', 1)
            item_info = {
                'code': item_parts[0].strip(),
                'name': item_parts[1].strip() if len(item_parts) > 1 else "N/A",
                'count': int(most_recommended['count'])
            }
        except (KeyError, ValueError, AttributeError) as e:
            logger.error(f"Error processing most recommended item: {str(e)}")

    # Geospatial Data Processing
    geospatial_data = []
    try:
        country_stats = (
            CustomerDetail.objects
            .exclude(country__isnull=True)
            .annotate(recommendation_count=Count('recommendations'))
            .filter(recommendation_count__gt=0)
            .values('country')
            .annotate(total=Count('recommendations'))
            .order_by('-total')[:10]
        )

        country_mapping = {
            'EIRE': 'Ireland',
            'UK': 'United Kingdom',
            'Nederland': 'Netherlands'
        }

        for stat in country_stats:
            original_country = stat['country']
            country = country_mapping.get(original_country, original_country)
            coordinates = COUNTRY_COORDINATES.get(country.title()) or (0, 0)

            if not coordinates:
                continue

            # Get top items
            top_items = (
                NewCustomerRecommendation.objects
                .filter(user__country=original_country)  # Already correct - uses FK relation
                .values('item_code')
                .annotate(count=Count('id'))
                .order_by('-count')[:5]
            )

            processed_items = []
            for item in top_items:
                try:
                    code_part, name_part = item['item_code'].split(',', 1)
                    processed_items.append({
                        'code': code_part.strip(),
                        'name': name_part.strip(),
                        'count': item['count']
                    })
                except (ValueError, KeyError):
                    processed_items.append({
                        'code': item.get('item_code', 'N/A'),
                        'name': 'Unknown Item',
                        'count': item.get('count', 0)
                    })

            geospatial_data.append({
                'country': original_country,
                'latitude': float(coordinates[0]),
                'longitude': float(coordinates[1]),
                'total_recommendations': stat['total'],
                'top_items': processed_items
            })

    except Exception as e:
        logger.error(f"Geospatial data error: {str(e)}")
        geospatial_data = []

    context.update({
        'total_recommendations': total_recommendations,
        'avg_confidence': avg_confidence_percent,
        'active_recommendations': status_counts.get('active', 0),
        'expired_recommendations': status_counts.get('expired', 0),
        'most_recommended_code': item_info['code'],
        'most_recommended_name': item_info['name'],
        'most_recommended_count': item_info['count'],
        'geospatial_data': geospatial_data,  # Remove json.dumps() and mark_safe
        'default_lat': 51.5074 if geospatial_data else 37.0902,
        'default_lng': 0.1278 if geospatial_data else -95.7129
    })

    return context
@never_cache
def new_customer_rec_view(request):
    context = get_new_customer_rec_context(request)
    return render(request, 'dashboard/new_customer_rec.html', context)



