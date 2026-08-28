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


def find_bundled_chat_template(model_path: str) -> str | None:
    """Return the bundled per-model chat template text for `model_path`, or None if
    no bundled template matches. This is the fallback branch of
    `apply_openvino_fc_chat_template`, exposed separately so tokenizer-less backends
    (e.g. LlamaCppHandler) can resolve the same template without an HF tokenizer
    object."""
    template_name = _template_name_for_model(model_path)
    if template_name is None:
        return None

    template_path = Path(__file__).parent / "templates" / template_name / "chat_template.jinja"
    if not template_path.is_file():
        return None

    return template_path.read_text(encoding="utf-8")


def apply_openvino_fc_chat_template(tokenizer, model_path: str) -> None:
    # Prefer the model's own chat_template.jinja. AutoTokenizer.from_pretrained
    # already auto-discovers and loads a chat_template.jinja file from the model
    # directory into tokenizer.chat_template, so if that happened we must NOT
    # override it with our own bundled copy - the model-provided template is the
    # authoritative one for this model. Only fall back to our own bundled
    # per-model template when the model directory doesn't ship a
    # chat_template.jinja at all.
    if tokenizer.chat_template:
        return

    bundled = find_bundled_chat_template(model_path)
    if bundled is not None:
        tokenizer.chat_template = bundled
