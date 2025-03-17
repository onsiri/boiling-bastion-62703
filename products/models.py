from django.db import models


def default_image_path():
    return 'product_images/default.jpg'

class Product(models.Model):
    product_description = models.CharField(max_length=2000)
    product_name = models.CharField(max_length=2000)
    image = models.ImageField(upload_to='product_images/', default=default_image_path)

    def __str__(self):
        return self.product_name