import pandas as pd
from django.contrib import admin, messages
from django.db import transaction
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse, path
from .models import Transaction, sale_forecast, NextItemPrediction, Item, CustomerDetail, NewCustomerRecommendation
from .utils import generate_forecast, predict_future_sales
from .models import Transaction
from .forms import CustomerDetailsCSVForm, CustomerDetailsForm, ItemsForm
from .resources import CustomerDetailsResource, TransactionResource
from import_export.admin import ImportExportModelAdmin
from django.shortcuts import redirect


class TransactionAdmin(admin.ModelAdmin):
    #resource_class = TransactionResource
    actions = None
    change_list_template = 'admin/ai_models/import_csv.html'
    list_display = ['get_user_id', 'TransactionId', 'TransactionTime', 'ItemDescription',
                    'NumberOfItemsPurchased', 'CostPerItem', 'Country', 'uploaded_at']

    def get_user_id(self, obj):
        return obj.user.UserId  # Access UserId through the ForeignKey

    get_user_id.short_description = 'User ID'
    get_user_id.admin_order_field = 'user__UserId'  # Enable sorting

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                "delete_all/",
                self.admin_site.admin_view(self.delete_all),
                name="transaction_delete_all",
            )
        ]
        return custom_urls + urls

    def delete_all(self, request):
        if not self.has_delete_permission(request):
            return self._dont_delete_related(request)

        try:
            with transaction.atomic():
                transaction.objects.all().delete()
            messages.success(request, "All entries deleted successfully")
        except Exception as e:
            messages.error(request, f"Error deleting entries: {str(e)}")

        return redirect("admin:your_app_transaction_changelist")

    def changelist_view(self, request, extra_context=None):
        if request.method == 'POST' and 'file' in request.FILES:
            file = request.FILES['file']
            try:
                # Read the Excel file
                df = pd.read_excel(file)

                # Validate required columns
                required_columns = ['UserId', 'TransactionId', 'TransactionTime',
                                    'ItemCode', 'ItemDescription', 'NumberOfItemsPurchased',
                                    'CostPerItem', 'Country']
                missing = [col for col in required_columns if col not in df.columns]
                if missing:
                    raise ValueError(f"Missing columns: {', '.join(missing)}")

                # Prepare data for bulk create
                transactions = []
                customer_cache = {}

                for index, row in df.iterrows():
                    try:
                        # Get or cache CustomerDetail
                        user_id = str(row['UserId']).strip()
                        if user_id not in customer_cache:
                            customer = CustomerDetail.objects.get(UserId=user_id)
                            customer_cache[user_id] = customer

                        transactions.append(Transaction(
                            user=customer_cache[user_id],
                            TransactionId=str(row['TransactionId']).strip(),
                            TransactionTime=str(row['TransactionTime']),
                            ItemCode=str(row['ItemCode']).strip(),
                            ItemDescription=str(row['ItemDescription']),
                            NumberOfItemsPurchased=int(row['NumberOfItemsPurchased']),
                            CostPerItem=float(row['CostPerItem']),
                            Country=str(row['Country']).strip()
                        ))
                    except CustomerDetail.DoesNotExist:
                        self.message_user(request,
                                          f"Skipped transaction {row['TransactionId']} - User ID {user_id} not found",
                                          level=40
                                          )
                    except Exception as e:
                        self.message_user(request,
                                          f"Error with transaction {row.get('TransactionId', 'N/A')}: {str(e)}",
                                          level=40
                                          )

                # Bulk create transactions
                if transactions:
                    Transaction.objects.bulk_create(transactions)
                    self.message_user(request,
                                      f"Successfully imported {len(transactions)} transactions",
                                      level=25
                                      )

                    # Generate forecasts
                    generate_forecast()
                    predict_future_sales()

                return HttpResponseRedirect(reverse('admin:ai_models_transaction_changelist'))

            except Exception as e:
                self.message_user(request, f"Import error: {str(e)}", level=40)
                return HttpResponseRedirect(reverse('admin:ai_models_transaction_changelist'))

        extra_context = extra_context or {}
        extra_context['upload_url'] = reverse('admin:ai_models_transaction_changelist')
        return super().changelist_view(request, extra_context=extra_context)

class SaleForecastAdmin(admin.ModelAdmin):
    list_display = ['ds', 'prediction', 'prediction_lower', 'prediction_upper', 'uploaded_at']

class NextItemPredictionAdmin(admin.ModelAdmin):
    list_display = ['UserId', 'PredictedItemDescription','PredictedItemCost', 'Probability', 'PredictedAt']

class ItemAdmin(admin.ModelAdmin):
    actions = None
    change_list_template = 'admin/ai_models/import_csv.html'
    list_display = ['ItemCode', 'ItemDescription', 'CostPerItem', 'uploaded_at']

    def changelist_view(self, request, extra_context=None):
        if request.method == 'POST' and 'file' in request.FILES:
            file = request.FILES['file']
            try:
                df = pd.read_excel(file)
                required_columns = ['ItemCode', 'ItemDescription', 'CostPerItem']
                missing = [col for col in required_columns if col not in df.columns]
                if missing:
                    raise ValueError(f"Missing columns: {', '.join(missing)}")

                items = []
                for index, row in df.iterrows():
                    try:
                        items.append(Item(
                            ItemCode=str(row['ItemCode']).strip(),
                            ItemDescription=str(row['ItemDescription']).strip(),
                            CostPerItem=str(row['CostPerItem']).strip()
                        ))
                    except Exception as e:
                        self.message_user(request, f"Error with row {index + 1}: {str(e)}", level=40)

                if items:
                    Item.objects.bulk_create(items)  # Fixed: Use Item, not CustomerDetail
                    self.message_user(request, f"Successfully imported {len(items)} items", level=25)  # Updated message

                return HttpResponseRedirect(reverse('admin:ai_models_item_changelist'))  # Fixed URL name

            except Exception as e:
                self.message_user(request, f"Import error: {str(e)}", level=40)
                return HttpResponseRedirect(reverse('admin:ai_models_item_changelist'))  # Fixed URL name

        extra_context = extra_context or {}
        extra_context['upload_url'] = reverse('admin:ai_models_item_changelist')  # Fixed URL name
        return super().changelist_view(request, extra_context=extra_context)

class CustomerDetailsAdmin(admin.ModelAdmin):

    actions = None
    change_list_template = 'admin/ai_models/import_csv.html'
    list_display = ['UserId', 'country', 'age', 'gender', 'income', 'occupation', 'education_level',
                    'existing_customer']

    def changelist_view(self, request, extra_context=None):
        if request.method == 'POST' and 'file' in request.FILES:
            file = request.FILES['file']
            try:
                # Read the Excel file
                df = pd.read_excel(file)

                # Validate required columns
                required_columns = ['UserId', 'existing_customer', 'country', 'age', 'gender', 'income', 'occupation',
                                    'education_level']
                missing = [col for col in required_columns if col not in df.columns]
                if missing:
                    raise ValueError(f"Missing columns: {', '.join(missing)}")

                # Prepare data for bulk create
                customers = []
                for index, row in df.iterrows():
                    try:
                        customers.append(CustomerDetail(
                            UserId=str(row['UserId']).strip(),
                            existing_customer=str(row['existing_customer']).strip(),
                            country=str(row['country']).strip(),
                            age=int(row['age']),
                            gender=str(row['gender']).strip(),
                            income=int(row['income']),
                            occupation=str(row['occupation']).strip(),
                            education_level=str(row['education_level']).strip()
                        ))
                    except Exception as e:
                        self.message_user(request, f"Error with customer {row.get('UserId', 'N/A')}: {str(e)}",
                                          level=40)

                # Bulk create customers
                if customers:
                    CustomerDetail.objects.bulk_create(customers)
                    self.message_user(request, f"Successfully imported {len(customers)} customers", level=25)

                return HttpResponseRedirect(reverse('admin:ai_models_customerdetail_changelist'))

            except Exception as e:
                self.message_user(request, f"Import error: {str(e)}", level=40)
                return HttpResponseRedirect(reverse('admin:ai_models_customerdetail_changelist'))

        extra_context = extra_context or {}
        extra_context['upload_url'] = reverse('admin:ai_models_customerdetail_changelist')
        return super().changelist_view(request, extra_context=extra_context)

class NewCustomerRecommendationAdmin(admin.ModelAdmin):
    list_display = ['user', 'item_code', 'confidence_score', 'generation_date', 'expiry_date']

def save_related(self, request, form, formsets, change):
        # Handle bulk uploads within a transaction
    with transaction.atomic():
        super().save_related(request, form, formsets, change)

admin.site.register(Transaction, TransactionAdmin)
admin.site.register(NextItemPrediction, NextItemPredictionAdmin)
admin.site.register(sale_forecast, SaleForecastAdmin)
admin.site.register(Item, ItemAdmin)
admin.site.register(CustomerDetail, CustomerDetailsAdmin)
admin.site.register(NewCustomerRecommendation, NewCustomerRecommendationAdmin)