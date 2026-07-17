import re
from typing import Optional

from ..tool_calls import coerce_arguments_to_schema, make_tool_call


def _parse_parameters(function_body: str) -> dict:
    arguments = {}
    pattern = re.compile(
        r"<parameter=([^>]+)>(.*?)</parameter>",
        flags=re.DOTALL,
    )
    for name, value in pattern.findall(function_body):
        arguments[name.strip()] = value.strip().strip("\n")
    return arguments


def parse(text: str, tools: Optional[list[dict]] = None) -> list[dict]:
    tool_calls = []
    pattern = re.compile(
        r"(?:<tool_call>\s*)?<function=([^>]+)>(.*?)(?:</function>\s*)?(?:</tool_call>|(?=<tool_call>|$))",
        flags=re.DOTALL,
    )
    for name, body in pattern.findall(text):
        function_name = name.strip()
        if not function_name:
            continue
        arguments = _parse_parameters(body)
        # OVMS's Qwen3CoderToolParser reads the tool's JSON schema and casts each
        # parameter value (always a raw string from the <parameter=...> XML) to
        # the declared type (integer/number/boolean/array/object). Mirror that
        # here so BFCL scoring sees the same typed values as OVMS.
        arguments = coerce_arguments_to_schema(function_name, arguments, tools)
        tool_calls.append(
            make_tool_call(function_name, arguments, len(tool_calls))
        )
    return tool_calls
