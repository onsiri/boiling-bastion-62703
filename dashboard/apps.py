from django.apps import AppConfig


class DashboardConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'dashboard'

    def ready(self):
        # Force import of dash apps
        try:
            import dashboard.dash_apps.sales_dash
            print("Successfully imported sales_dash")  # Debug line
        except Exception as e:
            print(f"Error importing dash apps: {e}")  # Catch errors