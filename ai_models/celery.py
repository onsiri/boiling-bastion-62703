import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'your_project.settings')
app = Celery('your_project', broker=os.environ.get('REDIS_URL', 'redis://localhost:6379/0'))
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()