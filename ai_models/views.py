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
    active_tab = request.GET.get('active_tab', 'top30-forecast')
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
        'date_range': f"{start_date} to {end_date}",
        'active_tab': active_tab  # Add this
    }

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        template_name = 'dashboard/partials/top_30_sale_forecast_partial.html'
    else:
        template_name = 'dashboard/sales_forecast.html'

    return render(request, template_name, context)

def generate_plot(figure):
    figure.update_layout(
        autosize=True,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return figure.to_html(
        full_html=False,
        config={'responsive': True}
    )

def future_sale_prediction(request):
    # Filter parameters
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

    # Date filters
    if filters['start_date']:
        base_query = base_query.filter(PredictedAt__gte=filters['start_date'])
    if filters['end_date']:
        base_query = base_query.filter(PredictedAt__lte=filters['end_date'])

    # Numeric filters with validation
    try:
        if filters['min_probability']:
            base_query = base_query.filter(Probability__gte=float(filters['min_probability']) / 100)
        if filters['max_probability']:
            base_query = base_query.filter(Probability__lte=float(filters['max_probability']) / 100)
        if filters['min_cost']:
            base_query = base_query.filter(PredictedItemCost__gte=float(filters['min_cost']))
        if filters['max_cost']:
            base_query = base_query.filter(PredictedItemCost__lte=float(filters['max_cost']))
    except ValueError:
        pass  # Handle invalid number formats silently

    # CSV Export Handling
    if request.GET.get('export') == 'csv':
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="Customer_Predictions_{timezone.now().date()}.csv"'

        writer = csv.writer(response)
        writer.writerow([
            'User ID', 'Prediction Date', 'Probability (%)',
            'Item Description', 'Item Cost', 'Prediction Timestamp'
        ])

        try:
            # Sorting logic matching the main view
            valid_sort_fields = ['UserId', 'PredictedItemDescription', 'Probability', 'PredictedItemCost', 'PredictedAt']
            sort_by_param = request.GET.get('sort_by', '-Probability')
            sort_field = sort_by_param.lstrip('-')

            if sort_field not in valid_sort_fields:
                sort_by_param = '-Probability'
                sort_field = 'Probability'

            ordered = base_query.order_by(sort_by_param)

            for item in ordered:
                writer.writerow([
                    item.UserId,
                    item.PredictedAt.date() if item.PredictedAt else 'N/A',
                    f"{item.Probability * 100:.2f}",
                    item.PredictedItemDescription,
                    f"{item.PredictedItemCost:.2f}",
                    item.PredictedAt.strftime('%Y-%m-%d %H:%M') if item.PredictedAt else 'N/A'
                ])
            return response
        except Exception as e:
            return HttpResponse(f"Error generating CSV: {str(e)}", status=500)

    # Sorting configuration
    valid_sort_fields = ['UserId', 'PredictedItemDescription', 'Probability', 'PredictedItemCost', 'PredictedAt']
    sort_by_param = request.GET.get('sort_by', '-Probability')  # Default sort by Probability descending
    sort_field = sort_by_param.lstrip('-')
    sort_order = 'desc' if sort_by_param.startswith('-') else 'asc'

    # Validate sorting parameters
    if sort_field not in valid_sort_fields:
        sort_by_param = '-Probability'
        sort_field = 'Probability'
        sort_order = 'desc'

    # Apply sorting
    ordered = base_query.order_by(sort_by_param)

    # Pagination
    paginator = Paginator(ordered, 20)
    page_number = request.GET.get('page')

    try:
        page_obj = paginator.page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)

    context = {
        'page_obj': page_obj,
        'sort_by': sort_field,
        'sort_order': sort_order,
        'filters': filters,
        'request': request,
        'min_ds': NextItemPrediction.objects.aggregate(Min('PredictedAt'))['PredictedAt__min'] or '',
        'max_ds': NextItemPrediction.objects.aggregate(Max('PredictedAt'))['PredictedAt__max'] or '',
    }

    # Determine template based on request type
    if request.headers.get('HX-Request'):
        # Check if the HTMX request is targeting the main container
        if request.GET.get('main_container') == '1':
            template = "dashboard/partials/future_sale_partial.html"
        else:
            # Default to the table template for HTMX table updates
            template = "ai_models/future_sale_table.html"
    else:
        # Full page request (initial load)
        template = "dashboard/partials/future_sale_partial.html"

    return render(request, template, context)


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
    sort_by = request.GET.get('sort_by', '-confidence_score')
    raw_sort_field = sort_by.lstrip('-')

        # Field mapping with case normalization
    valid_sort_fields = {
            'user': 'user__UserId',
            'user__userid': 'user__UserId',
            'confidence_score': 'confidence_score',
            'recommendation_type': 'recommendation_type',
            'generation_date': 'generation_date',
            'expiry_date': 'expiry_date'
        }

        # Normalize and validate the sort field
    normalized_field = valid_sort_fields.get(raw_sort_field.lower(), 'confidence_score')
    sort_order = 'desc' if sort_by.startswith('-') else 'asc'
    final_sort_by = f'-{normalized_field}' if sort_order == 'desc' else normalized_field

        # Apply sorting
    base_query = base_query.order_by(final_sort_by, '-generation_date')

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
        'sort_by': normalized_field,
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

@csrf_exempt
def generate_recommendations(request):
    if request.method == 'POST':
        task = generate_predictions_task.delay()
        return JsonResponse({'task_id': task.id})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)