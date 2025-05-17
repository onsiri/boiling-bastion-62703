"""
Django settings for django_project project.
"""

from dotenv import load_dotenv
import dj_database_url
from pathlib import Path
import os
import redis
from urllib.parse import urlparse

# Load environment variables first
load_dotenv()

# Build paths inside the project like this: BASE_DIR / 'subdir'
BASE_DIR = Path(__file__).resolve().parent.parent

# Security settings
SECRET_KEY = os.environ.get('SECRET_KEY', 'django-insecure-dev-key-123')  # Always override in production
DEBUG = True#os.environ.get('DJANGO_DEBUG', 'False') == 'True'
ALLOWED_HOSTS = [
    "boiling-bastion-62703-1fb7e4016adf.herokuapp.com",
    'app.insightsds.com',
    "localhost",
    "127.0.0.1", "0.0.0.0"
]

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.humanize',

    # Third-party
    'import_export',
    'crispy_forms',
    'crispy_bootstrap5',
    'django_plotly_dash',

    # Local
    'accounts.apps.AccountsConfig',
    'pages.apps.PagesConfig',
    'products.apps.ProductsConfig',
    'ai_models',
    'dashboard',
    'celery',
    'django_celery_beat',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'django_plotly_dash.middleware.BaseMiddleware',
    'accounts.middleware.RequireLoginMiddleware',
]

ROOT_URLCONF = 'django_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'products.context_processors.products_context',
            ],
        },
    },
]

WSGI_APPLICATION = 'django_project.wsgi.application'

# Database
DATABASES = {
    'default': dj_database_url.config(
        default='postgres://postgres:postgres@db:5432/postgres',
        conn_max_age=600,
        ssl_require=os.environ.get('DJANGO_ENV') == 'production'
    )
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

if not DEBUG:
    # Production storage configuration
    DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
    AWS_STORAGE_BUCKET_NAME = os.environ.get('AWS_STORAGE_BUCKET_NAME')
    AWS_S3_REGION_NAME = 'us-east-1'
    AWS_S3_CUSTOM_DOMAIN = f'{AWS_STORAGE_BUCKET_NAME}.s3.amazonaws.com'
    AWS_DEFAULT_ACL = 'private'
    AWS_QUERYSTRING_AUTH = False
else:
    # Local media storage
    DEFAULT_FILE_STORAGE = 'django.core.files.storage.FileSystemStorage'

# Security headers
if not DEBUG:
    SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_HSTS_SECONDS = 3600
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True

CSRF_TRUSTED_ORIGINS = [
    'https://boiling-bastion-62703-1fb7e4016adf.herokuapp.com',
    'http://localhost:8000',
    'https://app.insightsds.com',
]

# Custom user model
AUTH_USER_MODEL = 'accounts.CustomUser'

# Login redirects
LOGIN_URL = 'login'
LOGIN_REDIRECT_URL = 'home'
LOGOUT_REDIRECT_URL = 'home'

# Crispy Forms
CRISPY_ALLOWED_TEMPLATE_PACKS = 'bootstrap5'
CRISPY_TEMPLATE_PACK = 'bootstrap5'

# Plotly Dash
X_FRAME_OPTIONS = 'SAMEORIGIN'
PLOTLY_COMPONENTS = [
    'dash_core_components',
    'dash_html_components',
    'dash_renderer',
    'dpd_components',
]

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

#CELERY_BROKER_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
#CELERY_RESULT_BACKEND = os.environ['REDIS_URL']
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_TASK_TIME_LIMIT = 1800  # 30 minutes timeout for tasks
CELERY_WORKER_STATE_DB = '/tmp/celery_worker_state.db'
CELERY_BROKER_URL = 'redis://redis:6379/0'  # Using Redis
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
CELERY_WORKER_MAX_MEMORY_PER_CHILD = 300_000  # KB
CELERY_WORKER_MAX_TASKS_PER_CHILD = 100
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

# Parse the Redis URL
url = urlparse(REDIS_URL)

CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": REDIS_URL + "/1",  # Use Docker service name if applicable
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
            "IGNORE_EXCEPTIONS": True,  # Prevents crashes if Redis is down
        }
    }
}

# For Django Q and other Redis connections
REDIS_CONNECTION = {
    'host': url.hostname,
    'port': url.port,
    'password': url.password,
    'ssl': REDIS_URL.startswith('redis'),
    'ssl_cert_reqs': None
}
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


