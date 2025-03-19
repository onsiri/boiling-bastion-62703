from django.shortcuts import render
from .models import sale_forecast, NextItemPrediction, NewCustomerRecommendation
from django.db.models import F, OuterRef, Subquery, Max
from datetime import datetime
from django.core.exceptions import FieldError
from urllib.parse import urlencode
from django.views.decorators.http import require_POST
from django.http import JsonResponse, HttpResponse
from django.core.management import call_command
from django.core.paginator import Paginator
from django.views.generic import ListView
from django.db.models import F
import csv
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


def new_customer_recommendations(request):
    # Filter parameters
    filters = {
        'user_filter': request.GET.get('user_filter', '').strip(),
        'item_filter': request.GET.get('item_filter', '').strip(),
        'rec_type': request.GET.get('rec_type', ''),
        'min_confidence': request.GET.get('min_confidence'),
        'max_confidence': request.GET.get('max_confidence'),
        'start_date': request.GET.get('start_date'),
        'end_date': request.GET.get('end_date'),
        'expiry_date': request.GET.get('expiry_date'),
    }

    base_query = NewCustomerRecommendation.objects.all()

    # Apply filters
    if filters['user_filter']:
        base_query = base_query.filter(user__UserId__iexact=filters['user_filter'])

    if filters['item_filter']:
        base_query = base_query.filter(item_code__icontains=filters['item_filter'])

    if filters['rec_type']:
        base_query = base_query.filter(recommendation_type=filters['rec_type'])

    # Numeric filters
    try:
        if filters['min_confidence']:
            base_query = base_query.filter(confidence_score__gte=float(filters['min_confidence']))
        if filters['max_confidence']:
            base_query = base_query.filter(confidence_score__lte=float(filters['max_confidence']))
    except ValueError:
        pass

    # Date filters
    if filters['start_date']:
        base_query = base_query.filter(generation_date__gte=filters['start_date'])
    if filters['end_date']:
        base_query = base_query.filter(generation_date__lte=filters['end_date'])
    if filters['expiry_date']:
        base_query = base_query.filter(expiry_date__lte=filters['expiry_date'])

    # Sorting
    valid_sort_fields = [
        'user__UserId', 'item_code', 'confidence_score',
        'generation_date', 'expiry_date', 'recommendation_type'
    ]
    sort_by = request.GET.get('sort_by', '-confidence_score')
    sort_order = 'desc' if sort_by.startswith('-') else 'asc'
    sort_field = sort_by.lstrip('-')

    if sort_field not in valid_sort_fields:
        sort_by = '-confidence_score'
        sort_field = 'confidence_score'
        sort_order = 'desc'

    # Final ordering
    ordered = base_query.order_by(sort_by, '-generation_date')

    context = {
        'recommendations': ordered,
        'filters': filters,
        'sort_by': sort_field,
        'sort_order': sort_order,
        'rec_types': NewCustomerRecommendation.RecommendationType.choices,
        'request': request
    }

    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return render(request, 'ai_models/new_customer_recommendations_table.html', context)

    return render(request, 'ai_models/new_customer_recommendations.html', context)

@require_POST
def generate_recommendations(request):
    try:
        call_command('generate_predictions')
        return JsonResponse({'status': 'success', 'message': 'Recommendations generated successfully'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


def customer_recommendations(request):
    recommendations = NewCustomerRecommendation.objects.all().order_by('-confidence_score')

    # Add your filter logic here
    if 'export' in request.GET:
        # Apply the same filters but get 1000 records
        queryset = recommendations[:1000]

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="customer_recommendations.csv"'

        writer = csv.writer(response)
        writer.writerow([
            'Customer ID', 'Item Code', 'Type',
            'Confidence Score', 'Generated Date', 'Expiry Date'
        ])

        for rec in queryset:
            writer.writerow([
                rec.user.UserId,
                rec.item_code,
                rec.get_recommendation_type_display(),
                rec.confidence_score,
                rec.generation_date.strftime("%Y-%m-%d %H:%M"),
                rec.expiry_date.strftime("%Y-%m-%d %H:%M") if rec.expiry_date else ''
            ])

        return response
    paginator = Paginator(recommendations, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, 'ai_models/new_customer_recommendations.html', {
        'page_obj': page_obj,  # Make sure this key is used
        'recommendations': page_obj.object_list,
    })
