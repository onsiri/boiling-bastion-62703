import os
import ssl

from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_project.settings')
app = Celery('django_project', broker=os.environ.get('REDIS_URL', 'redis://localhost:6379/0'))
app.config_from_object('django.conf:settings', namespace='CELERY')
# Heroku Redis TLS fix
app.conf.broker_use_ssl = {'ssl_cert_reqs': ssl.CERT_NONE}
app.conf.redis_backend_use_ssl = {'ssl_cert_reqs': ssl.CERT_NONE}
app.autodiscover_tasks()