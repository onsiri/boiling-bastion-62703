def allow_unauthenticated(view_func):
    view_func.allow_unauthenticated = True
    return view_func