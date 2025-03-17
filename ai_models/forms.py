from django import forms
from import_export.forms import ImportForm
from .models import CustomerDetail, Transaction

class CustomerDetailsCSVForm(ImportForm):
    pass

class ExcelUploadForm(forms.Form):
    file = forms.FileField()

class CustomerDetailsForm(forms.ModelForm):
    class Meta:
        model = CustomerDetail
        fields = '__all__'

    def clean(self):
        cleaned_data = super().clean()
        print("Cleaned Data:", cleaned_data)  # Check in console
        return cleaned_data

    def save(self, commit=True):
        instance = super().save(commit=False)
        print("Saving instance:", instance.__dict__)  # Debug print
        if commit:
            instance.save()
        return instance

class TransactionForm(forms.ModelForm):
    class Meta:
        model = Transaction
        fields = '__all__'
    def clean(self):
        cleaned_data = super().clean()
        print("Cleaned Data:", cleaned_data)  # Check in console
        return cleaned_data

    def save(self, commit=True):
        instance = super().save(commit=False)
        print("Saving instance:", instance.__dict__)  # Debug print
        if commit:
            instance.save()
        return instance