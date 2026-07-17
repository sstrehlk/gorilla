from typing import Any, Optional

from bfcl_eval.constants.enums import ReturnFormat
from bfcl_eval.model_handler.utils import default_decode_ast_prompting

from .parsers import gemma4, generic, phi4, qwen3coder
from .tool_calls import make_tool_call, model_responses_from_tool_calls


def _decode_python_calls(text: str) -> Any:
    return default_decode_ast_prompting(
        text,
        ReturnFormat.PYTHON,
        has_tool_call_tag=False,
    )


def _select_structured_tool_parser(model_name: str, model_path: str, text: str):
    model_hint = f"{model_name} {model_path}".lower()
    if "functools" in text or "phi-4" in model_hint or "phi4" in model_hint:
        return phi4.parse
    if "<tool_call>" in text or "<function=" in text or "qwen3-coder" in model_hint:
        return qwen3coder.parse
    if text.strip().startswith("call:") or "gemma4" in model_hint or "gemma-4" in model_hint:
        return gemma4.parse
    return None


def parse_openvino_fc_response(
    text: str,
    model_name: str = "",
    model_path: str = "",
    tools: Optional[list[dict]] = None,
) -> dict:
    cleaned_text, model_responses, tool_calls = generic.parse(text, _decode_python_calls)

    if not tool_calls:
        parser = _select_structured_tool_parser(model_name, model_path, cleaned_text)
        if parser is qwen3coder.parse:
            tool_calls = parser(cleaned_text, tools=tools)
        elif parser is not None:
            tool_calls = parser(cleaned_text)

    if tool_calls:
        model_responses = model_responses_from_tool_calls(tool_calls)
    elif isinstance(model_responses, list):
        tool_calls = []
        for response in model_responses:
            if not isinstance(response, dict) or not response:
                continue
            name = next(iter(response))
            tool_calls.append(make_tool_call(name, response[name], len(tool_calls)))

    if model_responses is None:
        model_responses = cleaned_text

    tool_call_ids = [tool_call["id"] for tool_call in tool_calls]
    message_for_chat_history = {
        "role": "assistant",
        "content": cleaned_text,
    }
    if tool_calls:
        message_for_chat_history["tool_calls"] = tool_calls

    return {
        "text": cleaned_text,
        "model_responses": model_responses,
        "model_responses_message_for_chat_history": message_for_chat_history,
        "tool_call_ids": tool_call_ids,
    }
