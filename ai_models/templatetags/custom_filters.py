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