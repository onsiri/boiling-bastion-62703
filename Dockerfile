# Pull base image
FROM python:3.10.4-slim-bullseye

# Set environment variables
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
COPY cert.pem /app/cert.pem
COPY key.pem /app/key.pem

# Set work directory
WORKDIR /code

# Install dependencies
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# Copy project
COPY . .

# Expose the port
EXPOSE 8000

# Collect static files
#RUN python manage.py collectstatic --noinput

# Run the command to start the Gunicorn server
CMD ["gunicorn", "django_project.wsgi:application", "--workers", "3", "--bind", "0.0.0.0:8000"]