import time
from contextlib import ContextDecorator
from functools import wraps

__version__ = "1.0.11"

_async_start_event = None
_async_end_event = None


class ProfileNode:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = {}
        self.total_time = 0.0
        self.last_time = 0.0
        self.count = 0

    def get_or_create_child(self, name):
        if name not in self.children:
            self.children[name] = ProfileNode(name=name, parent=self)
        return self.children[name]

    def add_time(self, elapsed_seconds):
        self.last_time = elapsed_seconds
        self.total_time += elapsed_seconds
        self.count += 1

    def to_string(self, indent=0):
        if self.name == "root":
            lines = []
        else:
            lines = [
                f"{'  ' * indent}{self.name}: "
                f"total={self.total_time:.6f}s "
                f"last={self.last_time:.6f}s "
                f"count={self.count}"
            ]
            indent += 1
        for child in self.children.values():
            lines.append(child.to_string(indent=indent))
        return "\n".join(line for line in lines if line)


class Profiler:
    def __init__(self):
        self.root = ProfileNode("root")
        self.stack = [self.root]
        self.last_node = self.root

    @property
    def current_node(self):
        return self.stack[-1]

    def push(self, name):
        node = self.current_node
        for part in _split_name(name):
            node = node.get_or_create_child(part)
        self.stack.append(node)
        return node

    def pop(self, node, elapsed_seconds):
        assert self.stack[-1] is node
        node.add_time(elapsed_seconds)
        self.last_node = node
        self.stack.pop()

    def get_node(self, query):
        node = self.root
        for part in _split_name(query):
            assert part in node.children
            node = node.children[part]
        return node

    def reset(self):
        self.root = ProfileNode("root")
        self.stack = [self.root]
        self.last_node = self.root

    def to_string(self):
        text = self.root.to_string()
        return text or "<empty profiler>"


class _ProfileContext(ContextDecorator):
    def __init__(self, name, use_async=False):
        self.name = name
        self.use_async = use_async
        self.node = None
        self.start_time = None
        self.start_event = None
        self._using_async_event = False

    def __enter__(self):
        self.node = profiler.push(self.name)
        if self.use_async and _async_start_event is not None:
            try:
                self.start_event = _async_start_event()
                self._using_async_event = True
            except Exception:
                self.start_time = time.perf_counter()
        else:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._using_async_event and _async_end_event is not None:
            try:
                elapsed_seconds = _async_end_event(self.start_event)
            except Exception:
                elapsed_seconds = 0.0
        else:
            elapsed_seconds = time.perf_counter() - self.start_time
        profiler.pop(self.node, elapsed_seconds)
        return False


class Stopwatch:
    def __init__(self):
        self.start_time = None
        self.elapsed_seconds = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.elapsed_seconds = 0.0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_seconds = time.perf_counter() - self.start_time
        return False


def _split_name(name):
    return [part for part in str(name).split(".") if part]


def _profile_name(func):
    name = func.__name__
    if name.endswith("_model"):
        name = name[: -len("_model")]
    return name.lstrip("_") or func.__name__


def profile(func=None, name=None):
    if isinstance(func, str):
        return lambda wrapped: profile(wrapped, name=func)
    if func is None:
        return lambda wrapped: profile(wrapped, name=name)

    profile_name = name or _profile_name(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        with named_profile(profile_name):
            return func(*args, **kwargs)

    return wrapper


def named_profile(name):
    return _ProfileContext(name=name, use_async=False)


def named_profile_async(name):
    return _ProfileContext(name=name, use_async=True)


def setup_async(start_event, end_event):
    global _async_start_event, _async_end_event
    _async_start_event = start_event
    _async_end_event = end_event


def setup_async_as_sync():
    global _async_start_event, _async_end_event
    _async_start_event = None
    _async_end_event = None


def reset():
    profiler.reset()


profiler = Profiler()
