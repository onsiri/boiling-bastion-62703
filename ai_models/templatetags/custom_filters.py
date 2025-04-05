from django import template

register = template.Library()

@register.filter(name="mul")
def mul(value, arg):
    """Multiplies the value by the argument."""
    return float(value) * float(arg)