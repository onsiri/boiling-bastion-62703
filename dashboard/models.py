from django.db import models

class SalesData(models.Model):
    product = models.CharField(max_length=50)
    region = models.CharField(max_length=50)
    sales = models.DecimalField(max_digits=10, decimal_places=2)
    quantity = models.IntegerField()
    date = models.DateField()

    def __str__(self):
        return f"{self.product} - {self.region}"