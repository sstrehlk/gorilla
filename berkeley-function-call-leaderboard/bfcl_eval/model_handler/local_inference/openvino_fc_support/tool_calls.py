import json
from typing import Any, Optional


def ensure_json_arguments(arguments: Any) -> Any:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return arguments


def _find_tool_parameters_schema(function_name: str, tools: list[dict]) -> Optional[dict]:
    for tool in tools:
        fn = tool.get("function", tool)
        if fn.get("name") == function_name:
            return fn.get("parameters") or {}
    return None


def _coerce_value_to_json_type(value: Any, json_type: str) -> Any:
    """Cast a string parameter value to the type declared in the tool's JSON
    schema, mirroring OVMS's Qwen3CoderToolParser::parseToolSchema type mapping
    (string/number|integer/boolean/array/object).
    """
    if not isinstance(value, str):
        return value

    if json_type == "integer":
        try:
            return int(value)
        except (ValueError, TypeError):
            return value
    if json_type == "number":
        try:
            return int(value)
        except (ValueError, TypeError):
            pass
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    if json_type == "boolean":
        lowered = value.strip().lower()
        if lowered in ("true", "1"):
            return True
        if lowered in ("false", "0"):
            return False
        return value
    if json_type in ("array", "object"):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    # "string" and unknown types: keep as-is
    return value


def coerce_arguments_to_schema(
    function_name: str, arguments: dict, tools: Optional[list[dict]]
) -> dict:
    """Cast each argument value to the type declared for it in the matching
    tool's JSON schema (looked up by function name in ``tools``).  No-op if
    ``tools`` is empty/None or the function/parameter is not found.
    """
    if not tools:
        return arguments

    parameters_schema = _find_tool_parameters_schema(function_name, tools)
    if not parameters_schema:
        return arguments

    properties: dict = parameters_schema.get("properties") or {}
    coerced = dict(arguments)
    for key, value in coerced.items():
        prop = properties.get(key)
        if not prop or "type" not in prop:
            continue
        coerced[key] = _coerce_value_to_json_type(value, prop["type"])
    return coerced


def make_tool_call(name: str, arguments: Any, index: int) -> dict:
    return {
        "id": f"call_{index}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": ensure_json_arguments(arguments),
        },
    }


def model_responses_from_tool_calls(tool_calls: list[dict]) -> list[dict]:
    model_responses = []
    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        name = function.get("name")
        if not name:
            continue
        model_responses.append({name: function.get("arguments", {})})
    return model_responses
