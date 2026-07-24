OpenVINO BFCL FC chat templates are stored under parser-specific folders:

- `gptoss/chat_template.jinja`
- `phi4/chat_template.jinja`
- `qwen3coder/chat_template.jinja`
- `qwen36/chat_template.jinja`
- `gemma4/chat_template.jinja`

When a matching template exists, `template_registry.py` assigns it to
`tokenizer.chat_template` before BFCL generation. If no matching template is
present for a different model family, the model-provided tokenizer template is
used unchanged.
