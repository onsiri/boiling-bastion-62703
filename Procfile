web: gunicorn django_project.wsgi --log-file -
web: gunicorn django_project.wsgi --timeout 120
worker: celery -A django_project worker --liveness-check=10 --pool=prefork --concurrency=4 --without-heartbeat --loglevel=info
beat: celery -A django_project beat --loglevel=info  # If using periodic tasks