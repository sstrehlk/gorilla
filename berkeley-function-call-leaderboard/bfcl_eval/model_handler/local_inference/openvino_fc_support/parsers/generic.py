import json
import re
from typing import Any, Callable, Optional

from ..tool_calls import make_tool_call


def strip_thinking_tags(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)

    idx = text.rfind("</think>")
    if idx != -1:
        text = text[idx + len("</think>"):]

    idx = text.rfind("assistantfinal")
    if idx != -1:
        text = text[idx + len("assistantfinal"):]
    else:
        idx = text.find("assistantcommentary")
        if idx != -1:
            text = text[idx + len("assistantcommentary"):]
        else:
            for marker in ("finalanalysis", "finalcommentary"):
                idx = text.rfind(marker)
                if idx != -1:
                    text = text[idx + len(marker):]
                    break
    return text.strip()


def extract_python_style_function_calls(text: str) -> str:
    json_body = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    pattern = re.compile(
        r"to=(?:functions\.)?([\.\w]+?)(?:json|commentary|\s+(?:json|commentary)|\s*(?=\{))("
        + json_body + r")",
        re.DOTALL,
    )
    matches = pattern.findall(text)
    if not matches:
        return text

    calls = []
    for function_name, json_str in matches:
        try:
            arguments = json.loads(json_str)
        except json.JSONDecodeError:
            return text
        arguments_str = ", ".join(f"{key}={repr(value)}" for key, value in arguments.items())
        calls.append(f"{function_name}({arguments_str})")
    return ", ".join(calls)


def extract_json_tool_calls(text: str) -> list[dict]:
    decoder = json.JSONDecoder()
    parsed: Optional[Any] = None
    for json_start in re.finditer(r"[\[{]", text):
        try:
            parsed, _ = decoder.raw_decode(text[json_start.start():])
            break
        except json.JSONDecodeError:
            continue

    if parsed is None:
        return []
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []

    tool_calls = []
    for call in parsed:
        if not isinstance(call, dict):
            continue
        function = call.get("function", {})
        name = call.get("name") or function.get("name", "")
        arguments = call.get("arguments") or function.get("arguments", {})
        if name:
            tool_calls.append(make_tool_call(name, arguments, len(tool_calls)))
    return tool_calls


def parse(
    text: str,
    decode_python_calls: Callable[[str], Any],
) -> tuple[str, Any, list[dict]]:
    cleaned_text = strip_thinking_tags(text)

    extracted_text = extract_python_style_function_calls(cleaned_text)
    if extracted_text != cleaned_text:
        try:
            return cleaned_text, decode_python_calls(extracted_text), []
        except (SyntaxError, TypeError, ValueError):
            return cleaned_text, extracted_text, []

    tool_calls = extract_json_tool_calls(cleaned_text)
    if tool_calls:
        return cleaned_text, None, tool_calls

    return cleaned_text, cleaned_text, []
