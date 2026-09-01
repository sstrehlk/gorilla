from pathlib import Path


def _template_name_for_model(model_path: str) -> str | None:
    model_path_lower = model_path.lower()
    if "gpt-oss" in model_path_lower or "gptoss" in model_path_lower:
        return "gptoss"
    if "phi-4" in model_path_lower or "phi4" in model_path_lower:
        return "phi4"
    if "qwen3-coder" in model_path_lower or "qwen3coder" in model_path_lower:
        return "qwen3coder"
    if "qwen3.6" in model_path_lower or "qwen36" in model_path_lower:
        return "qwen36"
    if "gemma-4" in model_path_lower or "gemma4" in model_path_lower:
        return "gemma4"
    return None


def find_bfcl_chat_template(model_path: str) -> str | None:
    """Return this repo's own `"bfcl"` per-model chat template text for
    `model_path`, or None if no `bfcl` template matches. Used by both
    `HFTokenizerAdapter` and `LlamaCppTokenizerAdapter`."""
    template_name = _template_name_for_model(model_path)
    if template_name is None:
        return None

    template_path = Path(__file__).parent / "templates" / template_name / "chat_template.jinja"
    if not template_path.is_file():
        return None

    return template_path.read_text(encoding="utf-8")
