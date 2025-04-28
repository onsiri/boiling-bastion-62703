from django.shortcuts import redirect
from django.urls import reverse

class RequireLoginMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        return response

    def process_view(self, request, view_func, view_args, view_kwargs):
        # Bypass authentication for specific views
        if getattr(view_func, 'allow_unauthenticated', False) or \
                request.resolver_match.url_name in ['login', 'signup']:
            return None

        # Redirect to login if not authenticated
        if not request.user.is_authenticated:
            return redirect(reverse('login'))  # Use your login URL name here

        return None


