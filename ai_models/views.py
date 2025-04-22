from django.shortcuts import render
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

from .models import sale_forecast, NextItemPrediction, NewCustomerRecommendation
from django.db.models import F, OuterRef, Subquery, Max
from datetime import datetime, timedelta
from django.core.exceptions import FieldError
from urllib.parse import urlencode
from django.views.decorators.http import require_POST
from django.http import JsonResponse, HttpResponse
from django.core.management import call_command
from django.http import HttpResponse
import csv
from django.utils import timezone
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from .tasks import generate_predictions_task
from celery.result import AsyncResult
import logging
logger = logging.getLogger(__name__)
from django.db.models import Min, Max
from django.utils.dateparse import parse_date

def top_30_sale_forecast(request):
    # Calculate date range (tomorrow + 30 days)
    today = timezone.now().date()
    start_date = today + timedelta(days=1)
    end_date = today + timedelta(days=30)

    # Base query with date filter
    base_query = sale_forecast.objects.filter(
        ds__gte=start_date,
        ds__lte=end_date
    )

    # Default sorting values
    default_sort = 'ds'  # Forecast date
    default_order = 'asc'  # Chronological order

    # Handle CSV export
    if request.GET.get('export') == 'csv':
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="30_days_Sales_Forecast_{today}.csv"'
        response['X-Content-Type-Options'] = 'nosniff'  # Add this
        response['Cache-Control'] = 'no-cache'  # Add this

        writer = csv.writer(response)
        writer.writerow(['Date', 'Prediction', 'Prediction Lower', 'Prediction Upper', 'Uploaded At'])

        sort_by = request.GET.get('sort_by', default_sort)
        sort_order = request.GET.get('sort_order', default_order)
        order_prefix = '-' if sort_order == 'desc' else ''

        try:
            queryset = base_query.order_by(f'{order_prefix}{sort_by}')
            for item in queryset:
                writer.writerow([
                    item.ds.strftime('%Y-%m-%d'),  # Explicitly format date
                    max(0, item.prediction),
                    max(0, item.prediction_lower),
                    max(0, item.prediction_upper),
                    item.uploaded_at.strftime('%Y-%m-%d %H:%M')  # Format datetime
                ])
            return response
        except Exception as e:
            return HttpResponse(f"Error generating CSV: {str(e)}", status=500)

    # Get sort parameters
    sort_by = request.GET.get('sort_by', default_sort)
    sort_order = request.GET.get('sort_order', default_order)

    # Validate sort parameters
    valid_sort_fields = {'ds', 'prediction', 'uploaded_at'}
    if sort_by not in valid_sort_fields:
        sort_by = default_sort
        sort_order = default_order

    # Create order_by expression
    order_prefix = '-' if sort_order == 'desc' else ''
    order_by = f'{order_prefix}{sort_by}'

    try:
        sorted_query = base_query.order_by(order_by)
    except FieldError:
        sorted_query = base_query.order_by(default_sort)

    # Pagination setup
    items_per_page = 20  # One day per row
    paginator = Paginator(sorted_query, items_per_page)
    page_number = request.GET.get('page')

    try:
        page_obj = paginator.page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)

    context = {
        'page_obj': page_obj,
        'sort_by': sort_by,
        'sort_order': sort_order,
        'request': request,
        'date_range': f"{start_date} to {end_date}"
    }

    return render(request, 'ai_models/top_30_sale_forecast.html', context)


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
            base_query = base_query.filter(Probability__gte=float(filters['min_probability'])/100)
        if filters['max_probability']:
            base_query = base_query.filter(Probability__lte=float(filters['max_probability'])/100)
        if filters['min_cost']:
            base_query = base_query.filter(PredictedItemCost__gte=float(filters['min_cost']))
        if filters['max_cost']:
            base_query = base_query.filter(PredictedItemCost__lte=float(filters['max_cost']))
    except ValueError:
        pass  # Handle invalid number formats silently

    # CSV Export Handling
    if request.GET.get('export') == 'csv':
        today = timezone.now().date()
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="Customer_Purchase_Probability_{today}.csv"'
        response['X-Content-Type-Options'] = 'nosniff'
        response['Cache-Control'] = 'no-cache, no-store, must-revalidate'

        writer = csv.writer(response)  # Critical missing line
        writer.writerow([
            'User ID', 'Prediction Date', 'Probability (%)',
            'Item Description', 'Item Cost', 'Prediction Timestamp'
        ])

        try:
            # Sorting configuration for HTML view
            valid_sort_fields = [...]
            sort_by = request.GET.get('sort_by', 'Probability')
            sort_order = request.GET.get('sort_order', 'desc')
            # Validate sorting parameters
            if sort_by not in valid_sort_fields:
                sort_by = 'Probability'
                sort_order = 'desc'
            # Create order_by expression
            order_prefix = '-' if sort_order == 'desc' else ''
            order_by = f'{order_prefix}{sort_by}'
            ordered = base_query.order_by(order_by, '-Probability')
            for item in ordered:
                predicted_date = item.PredictedAt.strftime('%Y-%m-%d') if item.PredictedAt else 'N/A'
                timestamp = item.PredictedAt.strftime('%Y-%m-%d %H:%M') if item.PredictedAt else 'N/A'

                writer.writerow([
                    item.UserId or 'N/A',
                    predicted_date,
                    f"{item.Probability * 100:.2f}" if item.Probability else '0.00',
                    item.PredictedItemDescription or 'N/A',
                    f"${item.PredictedItemCost:.2f}" if item.PredictedItemCost is not None else '\$0.00',
                    timestamp
                ])
            return response
        except Exception as e:
            return HttpResponse(f"Error generating CSV: {str(e)}", status=500)

    # Sorting configuration for HTML view
    valid_sort_fields = ['UserId', 'PredictedAt', 'Probability',
                         'PredictedItemDescription', 'PredictedItemCost']
    sort_by = request.GET.get('sort_by', 'Probability')
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

    # Pagination
    paginator = Paginator(ordered, 20)
    page_number = request.GET.get('page')

    try:
        page_obj = paginator.page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)

    date_range = NextItemPrediction.objects.aggregate(
        min_ds=Min('PredictedAt'),
        max_ds=Max('PredictedAt')
    )

    # Format dates for HTML input
    min_ds = date_range['min_ds'].strftime('%Y-%m-%d') if date_range['min_ds'] else ''
    max_ds = date_range['max_ds'].strftime('%Y-%m-%d') if date_range['max_ds'] else ''

    context = {
        'page_obj': page_obj,
        'sort_by': sort_by,
        'sort_order': sort_order,
        'filters': filters,
        'request': request,
        'min_ds': min_ds,  # Add these
        'max_ds': max_ds   # to context
    }

    if request.headers.get('HX-Request') == 'true' and 'main-container' not in request.GET:
        template = "ai_models/future_sale_table.html"  # Table only
    else:
        template = "dashboard/partials/future_sale_partial.html"  # Full content

    return render(request, template, context)
@csrf_exempt
def generate_recommendations(request):
    if request.method == 'POST':
        task = generate_predictions_task.delay()
        return JsonResponse({'task_id': task.id})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)


def customer_recommendations(request):
    today = timezone.now().date()
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
            base_query = base_query.filter(confidence_score__gte=float(filters['min_confidence']) * 100)
        if filters['max_confidence']:
            base_query = base_query.filter(confidence_score__lte=float(filters['max_confidence']) * 100)
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

    base_query = base_query.order_by(sort_by, '-generation_date')

    # Handle CSV export
    if 'export' in request.GET:
        queryset = base_query[:1000]  # Get top 1000 filtered records
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="New_Customer_Recommendations_{today}.csv"'
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

    # Pagination
    paginator = Paginator(base_query, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'filters': filters,
        'sort_by': sort_field,
        'sort_order': sort_order,
        'rec_types': NewCustomerRecommendation.RecommendationType.choices,
        'request': request
    }

    if request.headers.get('HX-Request') == 'true':
        return render(request, 'ai_models/new_customer_recommendations_table.html', context)

    return render(request, 'ai_models/new_customer_recommendations.html', context)


def check_task_status(request):
    task_id = request.GET.get('task_id')  # Get task_id from query parameters
    if not task_id:
        return JsonResponse({'status': 'error', 'message': 'Missing task_id'}, status=400)

    result = AsyncResult(task_id)
    return JsonResponse({
        'status': result.state,
        'result': result.result if result.ready() else None
    })