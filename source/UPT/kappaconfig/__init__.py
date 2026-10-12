import copy
from pathlib import Path

import yaml

from .entities.wrappers import KCScalar

__version__ = "1.0.29"


class Processor:
    def preorder_process(self, node, trace):
        pass


class Resolver:
    def __init__(self, pre_processors=None, post_processors=None, template_path=None):
        self.pre_processors = list(pre_processors or [])
        self.post_processors = list(post_processors or [])
        self.template_path = Path(template_path or ".").expanduser()
        self._template_cache = {}

    def resolve(self, node):
        wrapped = copy.deepcopy(node)
        for processor in self.pre_processors:
            _walk_preorder(wrapped, processor)
        primitive = _unwrap(wrapped)
        resolved = self._resolve_node(primitive, root=primitive, local_vars=None)
        for processor in self.post_processors:
            _walk_preorder(resolved, processor)
        return resolved

    def _resolve_node(self, node, root, local_vars):
        if isinstance(node, dict):
            if "template" in node:
                node = self._resolve_template_node(node, root=root, local_vars=local_vars)
            else:
                node = {
                    key: self._resolve_node(value, root=root, local_vars=local_vars)
                    for key, value in node.items()
                }
            return node
        if isinstance(node, list):
            return [
                self._resolve_node(value, root=root, local_vars=local_vars)
                for value in node
            ]
        if isinstance(node, str):
            return self._resolve_string(node, root=root, local_vars=local_vars)
        return node

    def _resolve_template_node(self, node, root, local_vars):
        template = self._resolve_node(node["template"], root=root, local_vars=local_vars)
        assert isinstance(template, dict), "template must resolve to a dict"
        template = copy.deepcopy(template)
        for key, value in node.items():
            if not key.startswith("template."):
                continue
            path = key[len("template.") :].split(".")
            _set_path(template, path, self._resolve_node(value, root=root, local_vars=local_vars))
        template_vars = template.get("vars")
        resolved = self._resolve_node(template, root=template, local_vars=template_vars)
        if isinstance(resolved, dict):
            resolved.pop("vars", None)
        return resolved

    def _resolve_string(self, value, root, local_vars):
        if _is_full_expression(value):
            return self._eval_expr(value[2:-1], root=root, local_vars=local_vars)
        if "${" not in value:
            return value
        result = value
        for expression in _find_expressions(value):
            resolved = self._eval_expr(expression, root=root, local_vars=local_vars)
            result = result.replace("${" + expression + "}", str(resolved))
        return result

    def _eval_expr(self, expression, root, local_vars):
        if expression.startswith("vars."):
            path = expression[len("vars.") :].split(".")
            source = local_vars if local_vars is not None else root.get("vars", {})
            return self._resolve_node(_get_path(source, path), root=root, local_vars=local_vars)
        if expression.startswith("yaml:"):
            return self._load_yaml(expression[len("yaml:") :])
        if expression.startswith("select:"):
            key, source_expr = _split_select(expression[len("select:") :])
            source = self._resolve_string(source_expr, root=root, local_vars=local_vars)
            assert isinstance(source, dict), f"select source for '{key}' must be a dict"
            assert key in source, f"select key '{key}' not found"
            return copy.deepcopy(source[key])
        raise ValueError(f"unsupported kappaconfig expression '${{{expression}}}'")

    def _load_yaml(self, uri):
        path = self.template_path / uri
        if path.suffix == "":
            path = path.with_suffix(".yaml")
        path = path.resolve()
        if path not in self._template_cache:
            self._template_cache[path] = _unwrap(from_file_uri(path))
        return copy.deepcopy(self._template_cache[path])


class DefaultResolver(Resolver):
    pass


def from_file_uri(uri):
    path = Path(uri).expanduser()
    with open(path) as f:
        return from_primitive(yaml.safe_load(f))


def from_primitive(value):
    if isinstance(value, dict):
        return {key: from_primitive(val) for key, val in value.items()}
    if isinstance(value, list):
        return [from_primitive(val) for val in value]
    return KCScalar(value)


def _unwrap(value):
    if isinstance(value, KCScalar):
        return value.value
    if isinstance(value, dict):
        return {key: _unwrap(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_unwrap(val) for val in value]
    return value


def _walk_preorder(node, processor, trace=None):
    trace = trace or []
    processor.preorder_process(node, trace)
    if trace:
        parent, accessor = trace[-1]
        node = parent[accessor]
    if isinstance(node, dict):
        for key in list(node.keys()):
            _walk_preorder(node[key], processor, trace + [(node, key)])
    elif isinstance(node, list):
        for index in range(len(node)):
            _walk_preorder(node[index], processor, trace + [(node, index)])


def _get_path(value, path):
    current = value
    for part in path:
        current = current[part]
    return current


def _set_path(value, path, new_value):
    current = value
    for part in path[:-1]:
        current = current.setdefault(part, {})
    current[path[-1]] = new_value


def _is_full_expression(value):
    if not (value.startswith("${") and value.endswith("}")):
        return False
    depth = 0
    index = 0
    while index < len(value):
        if value.startswith("${", index):
            depth += 1
            index += 2
            continue
        if value[index] == "}":
            depth -= 1
            if depth == 0 and index != len(value) - 1:
                return False
        index += 1
    return depth == 0


def _find_expressions(value):
    expressions = []
    index = 0
    while index < len(value):
        start = value.find("${", index)
        if start == -1:
            break
        depth = 1
        cursor = start + 2
        while cursor < len(value) and depth > 0:
            if value.startswith("${", cursor):
                depth += 1
                cursor += 2
                continue
            if value[cursor] == "}":
                depth -= 1
            cursor += 1
        assert depth == 0, f"unterminated kappaconfig expression in '{value}'"
        expressions.append(value[start + 2 : cursor - 1])
        index = cursor
    return expressions


def _split_select(value):
    depth = 0
    index = 0
    while index < len(value):
        if value.startswith("${", index):
            depth += 1
            index += 2
            continue
        char = value[index]
        if char == "}":
            depth -= 1
        elif char == ":" and depth == 0:
            return value[:index], value[index + 1 :]
        index += 1
    raise ValueError(f"invalid select expression '{value}'")


__all__ = [
    "DefaultResolver",
    "KCScalar",
    "Processor",
    "Resolver",
    "from_file_uri",
    "from_primitive",
]
