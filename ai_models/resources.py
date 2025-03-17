from import_export import resources, fields, widgets
from .models import CustomerDetail, Transaction
from django.db.models import signals

class CustomerDetailsResource(resources.ModelResource):
    class Meta:
        model = CustomerDetail
        import_id_fields = ('UserId',)
        skip_unchanged = True
        skip_validation = True
        report_skipped = False
        batch_size = 10000
        use_bulk = True
        fields = ('UserId', 'existing_customer', 'country', 'age', 'gender', 'income', 'occupation', 'education_level')
        export_order = fields



class TransactionResource(resources.ModelResource):
    class Meta:
        model = Transaction
        import_id_fields = ('TransactionId',)
        skip_unchanged = True
        report_skipped = False
        batch_size = 1000000
        use_bulk = True
        fields = ('UserId', 'TransactionId', 'TransactionTime', 'ItemCode', 'ItemDescription',
                  'NumberOfItemsPurchased', 'CostPerItem', 'Country')
        export_order = fields