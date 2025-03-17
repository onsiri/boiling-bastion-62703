from django.shortcuts import render
from .models import sale_forecast, NextItemPrediction
from django.db.models import F, OuterRef, Subquery, Max
from datetime import datetime
from django.core.exceptions import FieldError
from django.http import JsonResponse
from urllib.parse import urlencode

def top_30_sale_forecast(request):
    sort_by = request.GET.get('sort_by', 'uploaded_at')
    sort_order = request.GET.get('sort_order', 'desc')

    order_by_field = sort_by
    if sort_order == 'asc':
        order_by_field = f'{sort_by}'
    else:
        order_by_field = f'-{sort_by}'

    try:
        top_30_items = sale_forecast.objects.annotate(
            date_ds=F('ds')
        ).order_by(order_by_field)[:30]
    except FieldError:
        top_30_items = sale_forecast.objects.order_by('-uploaded_at')[:30]

    # Convert 'ds' to datetime for proper sorting if needed
    if sort_by == 'ds':
        top_30_items = sorted(top_30_items, key=lambda x: datetime.strptime(x.ds, '%Y-%m-%d') if x.ds else None)

    return render(request, 'ai_models/top_30_sale_forecast.html', {
        'top_30_items': top_30_items,
        'sort_by': sort_by,
        'sort_order': sort_order,
    })



def future_sale_prediction(request):
    # Get and clean filter parameters
    filters = {
        'user_id_filter': request.GET.get('user_id_filter', '').strip(),
        'description_filter': request.GET.get('description_filter', '').strip(),
        'min_probability': request.GET.get('min_probability'),
        'max_probability': request.GET.get('max_probability'),
        'min_cost': request.GET.get('min_cost'),
        'max_cost': request.GET.get('max_cost'),
        'start_date': request.GET.get('start_date'),
        'end_date': request.GET.get('end_date'),
    }

    # Base query with probability filter
    base_query = NextItemPrediction.objects.filter(Probability__gt=0)

    # Apply filters
    if filters['user_id_filter']:
        base_query = base_query.filter(UserId__iexact=filters['user_id_filter'])

    if filters['description_filter']:
        base_query = base_query.filter(PredictedItemDescription__icontains=filters['description_filter'])

    # Numeric filters with validation
    try:
        if filters['min_probability']:
            base_query = base_query.filter(Probability__gte=float(filters['min_probability']))
        if filters['max_probability']:
            base_query = base_query.filter(Probability__lte=float(filters['max_probability']))
        if filters['min_cost']:
            base_query = base_query.filter(PredictedItemCost__gte=float(filters['min_cost']))
        if filters['max_cost']:
            base_query = base_query.filter(PredictedItemCost__lte=float(filters['max_cost']))
    except ValueError:
        pass  # Handle invalid number formats silently

    # Date filters
    if filters['start_date']:
        base_query = base_query.filter(PredictedAt__gte=filters['start_date'])
    if filters['end_date']:
        base_query = base_query.filter(PredictedAt__lte=filters['end_date'])

    # Sorting configuration
    valid_sort_fields = ['UserId', 'PredictedAt', 'Probability',
                        'PredictedItemDescription', 'PredictedItemCost']
    sort_by = request.GET.get('sort_by', 'Probability')  # Changed default to visible column
    sort_order = request.GET.get('sort_order', 'desc')

    # Validate sorting parameters
    if sort_by not in valid_sort_fields:
        sort_by = 'Probability'
        sort_order = 'desc'

    # Create order_by expression
    order_prefix = '-' if sort_order == 'desc' else ''
    order_by = f'{order_prefix}{sort_by}'

    # Apply sorting with Probability as secondary sort
    ordered = base_query.order_by(order_by, '-Probability')

    context = {
        'future_sale': ordered,
        'sort_by': sort_by,
        'sort_order': sort_order,
        'filters': filters,
        'request': request
    }

    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return render(request, 'ai_models/future_sale_table.html', context)

    return render(request, 'ai_models/future_sale.html', context)