web: gunicorn django_project.wsgi --log-file -
web: gunicorn your_project.wsgi --timeout 120
worker: celery -A django_project worker --loglevel=info