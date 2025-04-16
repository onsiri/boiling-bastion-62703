web: gunicorn django_project.wsgi --log-file -
web: gunicorn django_project.wsgi --timeout 120
web: gunicorn django_project.wsgi --preload --max-requests 500 --max-requests-jitter 50
worker: celery -A django_project worker --loglevel=info --without-mingle --without-gossip --concurrency=2 --pool=prefork
beat: celery -A django_project beat --loglevel=info  # If using periodic tasks