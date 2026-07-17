"""Parser for Gemma-4 tool-call format.

Gemma-4 generates tool calls as:
    call:function_name{key:<|"|>string_value<|"|>,key2:numeric_value}

Multiple calls are separated by newlines or appear sequentially.
The <|"|> tokens are Gemma's custom string delimiters that replace
standard quotation marks in the model's chat template.
"""
import re

from ..tool_calls import make_tool_call

# Gemma string delimiter token
_GEMMA_QUOTE = '<|"|>'

# Matches a single call:fn{...} block.
# The body inside braces may contain nested braces (e.g. for dict args).
_CALL_PATTERN = re.compile(
    r"call:([\w.]+)\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
    re.DOTALL,
)

# Matches a single key:value pair inside a call body.
# Values may be:
#   - Gemma-quoted strings:  <|"|>...<|"|>
#   - Unquoted scalars:      anything up to the next comma or end-of-body
_PARAM_PATTERN = re.compile(
    r"([\w.]+):"
    r"(?:"
    r"<\|\"?\|\|?\"?\|>([^<]*(?:<(?!\|\"?\|\|?\"?\|>)[^<]*)*)<\|\"?\|\|?\"?\|>"  # Gemma-quoted string
    r"|([^,}]+)"                                                                    # unquoted scalar
    r")",
    re.DOTALL,
)

# Simpler pattern for the Gemma quote token (handles both <|"|> variants)
_GQ = re.compile(r'<\|"?\|"?\|>')


def _parse_body(body: str) -> dict:
    """Parse the key:value pairs inside a call:fn{...} body."""
    arguments: dict = {}
    # Normalize all Gemma quote variants to a single marker for easier parsing
    normalized = _GQ.sub(_GEMMA_QUOTE, body)
    pos = 0
    while pos < len(normalized):
        # Skip whitespace and commas between params
        m = re.match(r'[\s,]+', normalized[pos:])
        if m:
            pos += m.end()
            continue
        # Try key:<|"|>value<|"|>
        m = re.match(
            r'([\w.]+):' + re.escape(_GEMMA_QUOTE) + r'(.*?)' + re.escape(_GEMMA_QUOTE),
            normalized[pos:],
            re.DOTALL,
        )
        if m:
            arguments[m.group(1)] = m.group(2)
            pos += m.end()
            continue
        # Try key:unquoted_value (up to next comma or end)
        m = re.match(r'([\w.]+):([^,}]+)', normalized[pos:])
        if m:
            key = m.group(1)
            value: object = m.group(2).strip()
            # Attempt numeric coercion
            try:
                value = int(value)           # type: ignore[assignment]
            except (ValueError, TypeError):
                try:
                    value = float(value)     # type: ignore[assignment]
                except (ValueError, TypeError):
                    if value == 'true':
                        value = True
                    elif value == 'false':
                        value = False
                    elif value == 'null':
                        value = None
            arguments[key] = value
            pos += m.end()
            continue
        # Cannot parse — skip one character to avoid infinite loop
        pos += 1
    return arguments


def parse(text: str) -> list[dict]:
    """Parse Gemma-4 style tool calls from model output.

    Returns a list of tool-call dicts compatible with the BFCL evaluation
    framework (same format as ``make_tool_call`` produces).
    """
    tool_calls: list[dict] = []
    for match in _CALL_PATTERN.finditer(text):
        function_name = match.group(1)
        body = match.group(2)
        arguments = _parse_body(body)
        tool_calls.append(make_tool_call(function_name, arguments, len(tool_calls)))
    return tool_calls
