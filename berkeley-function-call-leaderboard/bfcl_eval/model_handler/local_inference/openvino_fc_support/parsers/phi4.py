import json
import re
from typing import Any

from ..tool_calls import make_tool_call


def parse(text: str) -> list[dict]:
    match = re.search(r"functools\s*(\[.*\])", text, flags=re.DOTALL)
    if not match:
        return []

    try:
        calls: Any = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    if not isinstance(calls, list):
        return []

    tool_calls = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        arguments = call.get("arguments", {})
        if name:
            tool_calls.append(make_tool_call(name, arguments, len(tool_calls)))
    return tool_calls
