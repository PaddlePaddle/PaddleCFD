def get_log_or_pass_function(log_fn):
    def wrapper(msg):
        if log_fn is not None:
            log_fn(msg)
    return wrapper


def log(log_fn, msg):
    if log_fn is not None:
        log_fn(msg)
