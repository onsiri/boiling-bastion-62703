from django.shortcuts import render

# Create your views here.
from django.views.generic import ListView

from .models import Product


def product_list(request):
    try:
        products = Product.objects.all()
        return render(request, 'products/product_list.html', {'products': products})
    except Exception as e:
        import traceback
        traceback.print_exc()  # This will log the error to Heroku
        raise
