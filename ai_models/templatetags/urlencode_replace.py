from django import template
from urllib.parse import urlencode

register = template.Library()

@register.simple_tag(name='urlencode_replace')
def urlencode_replace(request, **kwargs):
    updated = request.GET.copy()
    for key, value in kwargs.items():
        # Convert underscores in parameter names to hyphens for URL
        url_key = key.replace('_', '-')
        if value is None:
            if url_key in updated:
                del updated[url_key]
        else:
            updated[url_key] = str(value)
    return updated.urlencode()