

import random
from django.utils import timezone
from dashboard.models import SalesData

products = ["Laptop", "Phone", "Tablet"]
regions = ["North", "South", "East", "West"]

# Create 50 sample records
for _ in range(50):
    SalesData.objects.create(
        product=random.choice(products),
        region=random.choice(regions),
        sales=random.randint(1000, 5000),
        quantity=random.randint(1, 20),
        date=timezone.now().date()
    )