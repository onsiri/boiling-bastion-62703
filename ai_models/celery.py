from __future__ import absolute_import, unicode_literals
import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_project.settings')

app = Celery('django_project')
app.config_from_object('django.conf:settings', namespace='CELERY')

# Optional: Add worker state DB configuration
app.conf.worker_state_db = '/tmp/celery_worker_state.db'  # For Linux/Mac
# app.conf.worker_state_db = 'C:\\celery_worker_state.db'  # For Windows

app.autodiscover_tasks()