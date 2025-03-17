from django.shortcuts import render

# Create your views here.
from django.views.generic import ListView

from .models import Product


def product_list(request):
    products = Product.objects.all()
    return render(request, 'products/product_list.html', {'products': products})
