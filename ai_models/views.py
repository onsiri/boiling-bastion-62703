from django.shortcuts import render
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.db.models import F, Sum, FloatField, IntegerField, Count
from django.db.models.functions import Cast, Substr, Trim
from .models import sale_forecast, NextItemPrediction, NewCustomerRecommendation, Transaction, CustomerDetail, CountrySaleForecast
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
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from .llm_utils import query_openai, detect_forecast_intent, detect_recommendation_intent, detect_strategic_intent
import re
from django.db import connection

def generate_strategic_recommendations():
    """Generate data-driven recommendations using all available models"""
    recommendations = []

    try:
        # 1. New customer opportunities
        try:
            top_new_customers = CustomerDetail.objects.filter(
                existing_customer=False,
                income__gt=50000
            ).order_by('-income')[:10]

            if top_new_customers.exists():
                rec = "1. Target high-potential new customers:\n" + "\n".join(
                    [f"- (Customer ID: {c.UserId} , Income: ${c.income}, {c.age}y/o {c.gender} in {c.country})"
                     for c in top_new_customers]
                )
                recommendations.append(rec)
        except Exception as e:
            logger.error(f"New customers recommendation failed: {str(e)}")

        # 2. Product recommendations from NextItemPrediction (FIXED)
        try:
            top_product_predictions = NextItemPrediction.objects.values(
                'PredictedItemDescription', 'PredictedItemCode'
            ).annotate(
                total_prob=Sum('Probability'),
                customer_count=Count('UserId')
            ).order_by('-total_prob')[:5]

            for product in top_product_predictions:
                # Get user details through separate query
                user_ids = NextItemPrediction.objects.filter(
                    PredictedItemCode=product['PredictedItemCode']
                ).values_list('UserId', flat=True).distinct()[:5]

                customers = CustomerDetail.objects.filter(
                    UserId__in=user_ids
                ).values('UserId', 'country')

                # Create mapping of UserId to country
                country_map = {c['UserId']: c['country'] for c in customers}

                # Get top predictions with probability
                predictions = NextItemPrediction.objects.filter(
                    PredictedItemCode=product['PredictedItemCode'],
                    UserId__in=user_ids
                ).order_by('-Probability')[:5]

                customer_list = []
                for p in predictions:
                    country = country_map.get(p.UserId, 'unknown')
                    customer_list.append(
                        f"- Customer ID: {p.UserId} ({p.Probability * 100:.1f}% likely) in {country}"
                    )

                if customer_list:
                    rec = f"2. Promote {product['PredictedItemDescription']} (SKU: {product['PredictedItemCode']}) to:\n" + "\n".join(
                        customer_list)
                    recommendations.append(rec)

        except Exception as e:
            logger.error(f"Product recommendations failed: {str(e)}")

        # 3. Upsell opportunities from transaction history
        try:
            frequent_buyers = Transaction.objects.values('user__UserId').annotate(
                total_spent=Sum(F('NumberOfItemsPurchased') * F('CostPerItem')),
                transaction_count=Count('TransactionId')
            ).order_by('-total_spent')[:5]

            if frequent_buyers.exists():
                rec = "3. Focus on high-value customers for upselling:\n" + "\n".join(
                    [
                        f"- Customer ID {b['user__UserId']} (Spent ${b['total_spent']:.2f} across {b['transaction_count']} purchases)"
                        for b in frequent_buyers]
                )
                recommendations.append(rec)
        except Exception as e:
            logger.error(f"Upsell recommendations failed: {str(e)}")

        # 4. Forecast-based recommendations
        try:
            current_year = datetime.now().year
            country_forecasts = CountrySaleForecast.objects.filter(
                ds__year=current_year + 1
            ).order_by('-prediction')[:3]

            if country_forecasts.exists():
                rec = "4. Expand in high-potential markets:\n" + "\n".join(
                    [f"- {f.group} (Projected sales: \${f.prediction:,.0f})"
                     for f in country_forecasts]
                )
                recommendations.append(rec)
        except Exception as e:
            logger.error(f"Forecast recommendations failed: {str(e)}")

    except Exception as e:
        logger.error(f"General recommendation error: {str(e)}", exc_info=True)
        return ["Recommendation system is currently unavailable"]

    return recommendations if recommendations else ["No specific recommendations found in current data"]
@require_POST
def ask_ai(request):
    try:
        prompt = request.POST.get('prompt', '').strip().lower()  # Add .lower()
        response = ""

        # 1. First check strategic business questions
        strategic_intent = detect_strategic_intent(prompt)
        print(f"Strategic Intent RAW Output: {strategic_intent}")
        if strategic_intent.get('needs_strategy') and strategic_intent['confidence'] > 0.7:  # Lowered threshold
            recommendations = generate_strategic_recommendations()
            if recommendations:
                # Add data-backed metrics to recommendations
                response = "Data-Driven Recommendations:\n\n" + "\n\n".join([
                    f"{rec}\n(Confidence: {min(strategic_intent['confidence'] * 100, 95):.1f}%)"
                    for rec in recommendations
                ])
            else:
                # Fallback to model-based suggestions if no data patterns
                response = "Current data suggests focusing on:\n" + \
                           "1. Customer retention strategies\n" + \
                           "2. High-margin product bundles\n" + \
                           "3. Seasonal demand forecasting"
            return JsonResponse({'response': response})

        # 2. Handle sales/forecast requests
        forecast_intent = detect_forecast_intent(prompt)
        if forecast_intent.get('needs_forecast') and forecast_intent['confidence'] > 0.7:
            try:
                year = int(forecast_intent.get('year', 2025))
            except (TypeError, ValueError):
                return JsonResponse({'response': "Please specify a valid year"})

            if year < 2025:
                # Handle historical sales from Transaction data
                transactions = Transaction.objects.annotate(
                    transaction_year=Cast(
                        Substr('TransactionTime', 1, 4),  # Direct year extraction
                        output_field=IntegerField()
                    )
                ).filter(transaction_year=year)

                total = transactions.aggregate(
                    total_revenue=Sum(
                        F('NumberOfItemsPurchased') * F('CostPerItem'),
                        output_field=FloatField()
                    )
                )['total_revenue'] or 0.0

                response = f"Total revenue for {year}: ${total:,.2f}"
            else:
                # Handle forecast data
                forecasts = sale_forecast.objects.filter(ds__year=year)
                total_forecast = sum([f.prediction for f in forecasts]) if forecasts else 0
                response = f"Total sales forecast for {year}: ${total_forecast:,.2f}"

            return JsonResponse({'response': response})  # Early return

        # 3. Only check recommendations if sales check didn't return
        recommendation_intent = detect_recommendation_intent(prompt)
        if recommendation_intent.get('needs_recommendation', False) and recommendation_intent['confidence'] > 0.6:
            # SAFETY: Even after fixes in llm_utils, add redundant checks
            product_name = recommendation_intent['product_name'] or ''
            product_name = product_name.upper().strip()

            if not product_name:
                return JsonResponse({'response': "Please specify a product for recommendations"})

            recommendations = NextItemPrediction.objects.filter(
                PredictedItemDescription__iexact=product_name
            ).order_by('-Probability')[:10]

            if recommendations.exists():
                customer_list = "\n".join(
                    [f"- User {rec.UserId} ({rec.Probability * 100:.1f}% probability)"
                     for rec in recommendations]
                )
                response = f"Customers most likely to purchase {product_name}:\n{customer_list}"
            else:
                response = f"No prediction data available for {product_name}"

            return JsonResponse({'response': response})  # Early return

        # 4. Fallback to general AI (MUST RETURN)
        ai_response = query_openai(prompt)
        response = ai_response if "API Error" not in ai_response else \
            "Currently unavailable. Please try again later."

        return JsonResponse({'response': response})

    except Exception as e:
        logger.error(f"Server Error: {str(e)}", exc_info=True)
        return JsonResponse({'error': "Internal server error"}, status=500)
def top_30_sale_forecast(request):
    print("DEBUG - ENTERING top_30_sale_forecast VIEW")
    active_tab = request.GET.get('active_tab', 'top30-forecast')
    # Calculate date range (tomorrow + 30 days)
    today = timezone.now().date()
    start_date = today + timedelta(days=1)
    end_date = today + timedelta(days=30)

    print(f"DEBUG - TODAY: {today}", flush=True)
    print(f"DEBUG - FILTER RANGE: {start_date} to {end_date}", flush=True)

    # Base query with date filter
    # Check what data exists in the database
    all_forecasts = sale_forecast.objects.all().order_by('ds')
    if all_forecasts.exists():
        print(f"DEBUG - DB HAS DATA: {all_forecasts.count()} records")
        print(f"DEBUG - FIRST DATE: {all_forecasts.first().ds}")
        print(f"DEBUG - LAST DATE: {all_forecasts.last().ds}")

        # Verify if any records match our date range
        matching = sale_forecast.objects.filter(ds__gte=start_date, ds__lte=end_date)
        print(f"DEBUG - MATCHING RECORDS: {matching.count()}")

        # Directly check a specific date to see if it matches
        test_date = start_date
        test_match = sale_forecast.objects.filter(ds=test_date)
        print(f"DEBUG - TEST FOR {test_date}: {test_match.exists()}")

        # Try string-based filtering as a test
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        print(f"DEBUG - TRYING STRING DATES: {start_str} to {end_str}")

        from django.db.models import Q
        test_query = sale_forecast.objects.filter(
            Q(ds__gte=start_date) & Q(ds__lte=end_date)
        )
        print(f"DEBUG - EXPLICIT Q OBJECTS: {test_query.count()}")

    # Base query with date filter
    base_query = sale_forecast.objects.filter(
        ds__gte=start_date,
        ds__lte=end_date
    ).order_by('ds')

    # Default sorting values
    default_sort = 'ds'  # Forecast date
    default_order = 'asc'  # Chronological order

    # Handle CSV export
    if request.GET.get('export') == 'csv':
        response = HttpResponse(content_type='text/csv')
        response[
            'Content-Disposition'] = f'attachment; filename="30_days_Sales_Forecast_{today}.csv"'
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
        'date_range': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        'start_date': start_date,
        'end_date': end_date,
        'active_tab': active_tab
    }

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        template_name = 'dashboard/partials/top_30_sale_forecast_partial.html'
    else:
        template_name = 'dashboard/sales_forecast.html'

    return render(request, template_name, context)

def generate_plot(figure):
    # Add this section to enforce date range
    figure.update_layout(
        xaxis=dict(
            range=['2025-01-01', '2025-12-28'],
            tickformat='%b %Y'  # Shows "Jan 2025", "Feb 2025", etc.
        ),
        # Existing layout config
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


