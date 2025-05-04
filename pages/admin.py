from django.contrib import admin
from .models import Ticket

class TicketAdmin(admin.ModelAdmin):

    list_display = ['title', 'author', 'body']

# Registration
admin.site.register(Ticket, TicketAdmin)