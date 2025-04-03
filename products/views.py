from django.shortcuts import render
from django.views.generic import ListView
from django.shortcuts import get_object_or_404
from .models import Product

def product_detail(request, product_id):
    product = get_object_or_404(Product, id=product_id)
    return render(request, 'products/product_detail.html', {'product': product})
def product_list(request):
    try:
        products = Product.objects.all()
        return render(request, 'products/product_list.html', {'products': products})
    except Exception as e:
        import traceback
        traceback.print_exc()  # This will log the error to Heroku
        raise
