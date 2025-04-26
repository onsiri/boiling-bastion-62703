from django import template
import ast

register = template.Library()

@register.filter(name="mul")
def mul(value, arg):
    """Multiplies the value by the argument."""
    return float(value) * float(arg)

@register.filter(name='split_item_code')
def split_item_code(value, part):
    try:
        cleaned = value.strip('()')
        parts = ast.literal_eval(cleaned)
        return {
            'code': parts[0],
            'description': parts[1],
            'cost': parts[2]
        }.get(part, 'N/A')
    except (ValueError, SyntaxError, IndexError):
        return 'N/A'

@register.filter(name='toggle')
def toggle(value):
    """Toggle between asc/desc for sorting"""
    return 'desc' if value == 'asc' else 'asc'

@register.filter
def format_currency(value):
    return f"${value:,.2f}" if value else "N/A"

@register.filter
def format_datetime(value):
    return value.strftime("%Y-%m-%d %H:%M") if value else ""


@register.simple_tag
def param_replace(request, **kwargs):
    params = request.GET.copy()
    for key, value in kwargs.items():
        if value:
            params[key] = value
        else:
            params.pop(key, None)
    return params.urlencode()