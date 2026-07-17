from pathlib import Path


def _template_name_for_model(model_path: str) -> str | None:
    model_path_lower = model_path.lower()
    if "gpt-oss" in model_path_lower or "gptoss" in model_path_lower:
        return "gptoss"
    if "phi-4" in model_path_lower or "phi4" in model_path_lower:
        return "phi4"
    if "qwen3-coder" in model_path_lower or "qwen3coder" in model_path_lower:
        return "qwen3coder"
    return None


def apply_openvino_fc_chat_template(tokenizer, model_path: str) -> None:
    template_name = _template_name_for_model(model_path)
    if template_name is None:
        return

    template_path = Path(__file__).parent / "templates" / template_name / "chat_template.jinja"
    if not template_path.is_file():
        return

    tokenizer.chat_template = template_path.read_text(encoding="utf-8")
