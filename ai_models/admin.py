import pandas as pd
from django.contrib import admin, messages
from django.db import transaction
from django.http import HttpResponseRedirect, JsonResponse
from django.urls import reverse, path
from django.shortcuts import redirect, render
from .models import Transaction, sale_forecast, NextItemPrediction, Item, CustomerDetail, NewCustomerRecommendation, CountrySaleForecast, ItemSaleForecast
from .utils import generate_forecast, predict_future_sales, import_forecasts_from_s3, upload_object_db
import numpy as np
from .tasks import async_upload_object_db

class BulkActionMixin:
    # Bulk delete configuration
    show_delete_all = True
    delete_confirmation_template = None  # Set a custom template if needed

    # Bulk upload configuration
    show_upload = True
    upload_confirmation_template = None

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                'delete_all/',
                self.admin_site.admin_view(self.delete_all_view),
                name=f'{self.model._meta.app_label}_{self.model._meta.model_name}_delete_all'
            ),
            path(
                'bulk_upload/',
                self.admin_site.admin_view(self.bulk_upload_view),
                name=f'{self.model._meta.app_label}_{self.model._meta.model_name}_bulk_upload'
            ),
        ]
        return custom_urls + urls

    def changelist_view(self, request, extra_context=None):
        extra_context = extra_context or {}
        extra_context.update({
            'show_delete_all': self.show_delete_all,
            'show_upload': self.show_upload,
        })
        return super().changelist_view(request, extra_context)

    def delete_all_view(self, request):
        if request.method == 'POST':
            try:
                deleted_count, _ = self.model.objects.all().delete()
                self.message_user(
                    request,
                    f"Successfully deleted {deleted_count} records",
                    level=messages.SUCCESS
                )
            except Exception as e:
                self.message_user(
                    request,
                    f"Error deleting records: {str(e)}",
                    level=messages.ERROR
                )
            changelist_url = reverse(f'admin:{self.model._meta.app_label}_{self.model._meta.model_name}_changelist')
            return redirect(changelist_url)

        context = {
            **self.admin_site.each_context(request),
            'opts': self.model._meta,
            'title': 'Confirm Bulk Delete'
        }
        return render(
            request,
            self.delete_confirmation_template or 'admin/bulk_delete_confirmation.html',
            context
        )

    def bulk_upload_view(self, request):
        if request.method == 'POST':
            file = request.FILES.get('file')
            if not file:
                self.message_user(request, "No file uploaded", level=40)
                return redirect(reverse(f'admin:{self.model._meta.app_label}_{self.model._meta.model_name}_changelist'))

            try:
                df = self._process_uploaded_file(file)
                return self.process_dataframe(request, df)

            except Exception as e:
                self.message_user(request, f"Error: {str(e)}", level=40)
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({'error': str(e)}, status=400)
                return redirect(reverse(f'admin:{self.model._meta.app_label}_{self.model._meta.model_name}_changelist'))

        context = self.admin_site.each_context(request)
        context.update({
            'opts': self.model._meta,
            'title': 'Bulk Upload'
        })
        return render(request, self.upload_confirmation_template or 'admin/bulk_upload_form.html', context)

    def _handle_bulk_upload(self, request):
        file = request.FILES.get('file')
        if file is None:
            self.message_user(request, "No file uploaded", level=messages.ERROR)
            changelist_url = reverse(
                f'admin:{self.model._meta.app_label}_{self.model._meta.model_name}_changelist'
            )
            return redirect(changelist_url)

        try:
            df = self._process_uploaded_file(file)
            return self.process_dataframe(request, df)
        except Exception as e:
            self.message_user(
                request,
                f"Error processing file: {str(e)}",
                level=messages.ERROR
            )
            changelist_url = reverse(
                f'admin:{self.model._meta.app_label}_{self.model._meta.model_name}_changelist'
            )
            return redirect(changelist_url)

    def _process_uploaded_file(self, file):
        import pandas as pd
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file)
        else:
            raise ValueError("Unsupported file format")

    def _validate_columns(self, df, required_columns):
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

    def process_dataframe(self, request, df):
        raise NotImplementedError("Subclasses must implement process_dataframe()")


# Concrete Admin Classes
class TransactionAdmin(BulkActionMixin, admin.ModelAdmin):
    change_list_template = "admin/ai_models/transaction/change_list.html"
    list_display = ['get_user_id', 'TransactionId', 'TransactionTime', 'ItemDescription',
                    'NumberOfItemsPurchased', 'CostPerItem', 'Country', 'uploaded_at']
    show_delete_all = True
    show_upload = True

    def get_user_id(self, obj):
        return obj.user.UserId

    get_user_id.short_description = 'User ID'
    get_user_id.admin_order_field = 'user__UserId'

    def bulk_upload_view(self, request):
        if request.method == 'POST':
            try:
                file = request.FILES.get('file')
                if not file:
                    return JsonResponse({'error': 'No file uploaded'}, status=400)

                df = self._process_uploaded_file(file)
                return self.process_dataframe(request, df)
            except Exception as e:
                self.message_user(request, f"Error: {str(e)}", level=messages.ERROR)
                return JsonResponse({'error': str(e)}, status=400)
        return super().bulk_upload_view(request)

    def process_dataframe(self, request, df):
        response = None
        try:
            # Validate required columns first
            required_columns = {'UserId', 'TransactionId', 'TransactionTime', 'ItemCode',
                                'ItemDescription', 'NumberOfItemsPurchased', 'CostPerItem', 'Country'}
            self._validate_columns(df, required_columns)

            # Convert UserIds to strings and clean
            df['UserId'] = df['UserId'].astype(str).str.strip()

            # Get existing users in bulk with cleaned UserIds
            existing_users = CustomerDetail.objects.filter(
                UserId__in=df['UserId'].unique()
            ).in_bulk(field_name='UserId')

            # Prepare data
            df['TransactionTime'] = pd.to_datetime(df['TransactionTime']).dt.tz_localize(None)
            df = df.replace({np.nan: None})
            transactions = []
            errors = []
            missing_users = set()

            # Process records with validation
            for row in df.itertuples(index=True):
                try:
                    user_id = row.UserId

                    # Validate user exists
                    if user_id not in existing_users:
                        missing_users.add(user_id)
                        raise ValueError(f"User {user_id} not found")

                    # Validate numeric fields
                    try:
                        num_items = int(row.NumberOfItemsPurchased)
                        cost = float(row.CostPerItem)
                    except ValueError as e:
                        raise ValueError(f"Invalid numeric value: {str(e)}")

                    transactions.append(Transaction(
                        user=existing_users[user_id],  # Use actual CustomerDetail instance
                        TransactionId=str(row.TransactionId).strip(),
                        TransactionTime=row.TransactionTime,
                        ItemCode=str(row.ItemCode).strip(),
                        ItemDescription=str(row.ItemDescription),
                        NumberOfItemsPurchased=num_items,
                        CostPerItem=cost,
                        Country=str(row.Country).strip()
                    ))
                except Exception as e:
                    errors.append(f"Row {row.Index}: {str(e)}")

            # Handle missing users error
            if missing_users:
                error_msg = f"Missing users: {', '.join(sorted(missing_users)[:5])}"
                if len(missing_users) > 5:
                    error_msg += f"... ({len(missing_users) - 5} more)"
                self.message_user(request, error_msg, level=messages.ERROR)

            # Bulk create transactions
            created_count = 0
            if transactions:
                created_objs = Transaction.objects.bulk_create(
                    transactions,
                    batch_size=2000,
                    ignore_conflicts=True
                )
                created_count = len(created_objs)
            messages = []
            if created_count:
                messages.append(f"Successfully imported {created_count} transactions")
            # Handle analytics

            analytics_errors = []
            try:
                print("start generate_forecast function")
                sale_forecasts_df = import_forecasts_from_s3('forecast_next_30.csv')
                upload_object_db('sale_forecast', sale_forecasts_df)
                country_forecasts_df = import_forecasts_from_s3('country_forecasts.csv')
                upload_object_db('CountrySaleForecast', country_forecasts_df)
                item_forecasts_df = import_forecasts_from_s3('item_forecasts.csv')
                #upload_object_db('ItemSaleForecast', item_forecasts_df)
                async_upload_object_db.delay(ItemSaleForecast, item_forecasts_df.to_json())
                print("Upload started in the background!")
            except Exception as e:
                analytics_errors.append(f"Forecast error: {str(e)}")

            try:
                print("start predict_future_sales function")
                predict_future_sales(request)

            except Exception as e:
                analytics_errors.append(f"Sales prediction error: {str(e)}")

            # Prepare response

            if errors:
                messages.append(f"Encountered {len(errors)} errors (first 3: {', '.join(errors[:3])})")
            if analytics_errors:
                messages.append(f"Analytics issues: {', '.join(analytics_errors)}")

            status_level = messages.SUCCESS if created_count else messages.WARNING
            self.message_user(request, ". ".join(messages), level=status_level)

            response_data = {
                'created': created_count,
                'errors': len(errors),
                'warnings': len(analytics_errors),
                'missing_users': list(missing_users)[:10]
            }
            response = JsonResponse(response_data)

        except Exception as e:
            self.message_user(request, f"Processing failed: {str(e)}", level=messages.ERROR)
            response = JsonResponse({'error': str(e)}, status=400)

        finally:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return response or JsonResponse({'status': 'completed'})
            return response or HttpResponseRedirect(reverse('admin:ai_models_transaction_changelist'))


class ItemAdmin(BulkActionMixin, admin.ModelAdmin):
    list_display = ['ItemCode', 'ItemDescription', 'CostPerItem', 'uploaded_at']
    show_delete_all = True  # Set to True
    show_upload = True
    def process_dataframe(self, request, df):
        required_columns = {'ItemCode', 'ItemDescription', 'CostPerItem'}
        self._validate_columns(df, required_columns)
        items = []
        for index, row in df.iterrows():
            try:
                items.append(Item(
                    ItemCode=str(row['ItemCode']).strip(),
                    ItemDescription=str(row['ItemDescription']).strip(),
                    CostPerItem=float(row['CostPerItem'])
                ))
            except Exception as e:
                self.message_user(
                    request,
                    f"Error with item {row.get('ItemCode', 'N/A')}: {str(e)}",  # Fix error message
                    level=40
                )

        if items:
            Item.objects.bulk_create(items)
            self.message_user(request, f"Imported {len(items)} items successfully", level=25)

        return HttpResponseRedirect(reverse('admin:ai_models_item_changelist'))


class CustomerDetailsAdmin(BulkActionMixin, admin.ModelAdmin):
    list_display = ['UserId', 'country', 'age', 'gender', 'income', 'occupation',
                    'education_level', 'existing_customer']
    show_delete_all = True
    show_upload = True

    def get_urls(self):
        return super().get_urls()

    def process_dataframe(self, request, df):
        required_columns = {
            'UserId', 'country', 'age', 'gender', 'income',
            'occupation', 'education_level', 'existing_customer'
        }
        self._validate_columns(df, required_columns)

        customers = []
        for index, row in df.iterrows():
            try:
                customers.append(CustomerDetail(
                    UserId=str(row['UserId']).strip(),
                    country=str(row['country']).strip(),
                    age=int(row['age']),
                    gender=str(row['gender']).strip(),
                    income=float(row['income']),
                    occupation=str(row['occupation']).strip(),
                    education_level=str(row['education_level']).strip(),
                    existing_customer=True if str(row['existing_customer']).strip().lower() in ['yes', 'Yes', '1'] else False
                ))
            except Exception as e:
                self.message_user(
                    request,
                    f"Error creating customer object: {str(e)}",
                    level=40
                )

        try:
            CustomerDetail.objects.bulk_create(customers)
            self.message_user(
                request,
                f"Imported {len(customers)} customers successfully",
                level=25
            )
        except Exception as e:
            self.message_user(
                request,
                f"Error importing customers: {str(e)}",
                level=40
            )
        return HttpResponseRedirect(reverse('admin:ai_models_customerdetail_changelist'))


# Standard Admins
class SaleForecastAdmin(admin.ModelAdmin):
    list_display = ['ds', 'prediction', 'prediction_lower', 'prediction_upper', 'accuracy_score' ,'uploaded_at']


class NextItemPredictionAdmin(admin.ModelAdmin):
    list_display = ['UserId', 'PredictedItemCode', 'PredictedItemDescription',  'PredictedItemCost','Probability','PredictedAt']


class NewCustomerRecommendationAdmin(admin.ModelAdmin):

    list_display = ['user', 'item_code', 'confidence_score', 'generation_date', 'expiry_date']
    raw_id_fields = ['user']

class CountrySaleForecastAdmin(admin.ModelAdmin):

    list_display = ['group', 'ds', 'prediction', 'prediction_lower', 'prediction_upper']

class ItemSaleForecastAdmin(admin.ModelAdmin):

    list_display = ['group', 'ds', 'prediction', 'prediction_lower', 'prediction_upper']

# Registration
admin.site.register(Transaction, TransactionAdmin)
admin.site.register(Item, ItemAdmin)
admin.site.register(CustomerDetail, CustomerDetailsAdmin)
admin.site.register(sale_forecast, SaleForecastAdmin)
admin.site.register(NextItemPrediction, NextItemPredictionAdmin)
admin.site.register(NewCustomerRecommendation, NewCustomerRecommendationAdmin)
admin.site.register(CountrySaleForecast, CountrySaleForecastAdmin)
admin.site.register(ItemSaleForecast, ItemSaleForecastAdmin)

