import glob
import json
import os
from datetime import datetime
from typing import Optional

import jinja2
from bfcl_eval.model_handler.local_inference.base_openvino_handler import (
    BaseOpenVINOHandler,
)
from bfcl_eval.model_handler.local_inference.openvino_fc_support.template_registry import (
    find_bfcl_chat_template,
)
from jinja2.sandbox import ImmutableSandboxedEnvironment
from overrides import override

# Passed via --ov-config as {"chat_template_source": "gguf"|"model_dir"|"bfcl"}.
_CHAT_TEMPLATE_SOURCES = ("gguf", "model_dir", "bfcl")


def _raise_exception(message: str):
    raise jinja2.exceptions.TemplateError(message)


def _tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(
        value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
    )


def _strftime_now(fmt: str) -> str:
    return datetime.now().strftime(fmt)


def _compile_chat_template(template_str: str) -> jinja2.Template:
    # Mirrors transformers' `PreTrainedTokenizerBase.apply_chat_template()` sandboxed
    # environment/globals (tojson filter, raise_exception/strftime_now globals), so
    # templates written against HF's chat-template semantics render identically here
    # without needing a real HF tokenizer object.
    jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    jinja_env.filters["tojson"] = _tojson
    jinja_env.globals["raise_exception"] = _raise_exception
    jinja_env.globals["strftime_now"] = _strftime_now
    return jinja_env.from_string(template_str)


def _find_gguf_file(model_path: str) -> str:
    if os.path.isfile(model_path) and model_path.endswith(".gguf"):
        return model_path
    matches = sorted(glob.glob(os.path.join(model_path, "*.gguf")))
    if not matches:
        raise FileNotFoundError(f"No .gguf file found in '{model_path}'")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple .gguf files found in '{model_path}': {matches}. "
            "Keep only the one to use in that directory."
        )
    return matches[0]


class LlamaCppTokenizerAdapter:
    """``TokenizerAdapter`` (see ``tokenizer_adapters.py``) for GGUF models:
    tokenization always comes from llama.cpp's own GGUF-embedded vocab
    (``llama_cpp.Llama.tokenize()``/``.detokenize()``), via a cheap
    ``vocab_only=True`` ``Llama`` instance kept separate from
    ``LlamaCppHandler``'s (weight-loading) generation instance.

    The chat template is independently selectable via ``chat_template_source``
    (read out of the ``device_properties`` dict passed to ``load()``, i.e. via
    ``--ov-config``) - by default it uses the template embedded in the ``.gguf``
    file itself (``tokenizer.chat_template`` GGUF metadata key), but can instead be
    pointed at the same ``chat_template.jinja`` file ``openvino-genai-FC``/
    ``openvino-genai-vlm-FC`` use (``"model_dir"``), for exact template parity with
    those handlers, or forced to this repo's own bfcl per-model fallback template
    (``"bfcl"``). Whichever source is picked, rendering itself goes through a
    minimal jinja2 sandboxed environment (mirroring HF's ``apply_chat_template()``
    globals/filters, see ``_compile_chat_template``), NOT an HF tokenizer.

    Only FC-style rendering (``render_chat_prompt``) is implemented -
    ``render_prompt`` (plain prompting mode, no tools) is intentionally
    unsupported since ``LlamaCppHandler`` only registers FC model names.
    """

    def __init__(self) -> None:
        self._llm = None  # vocab_only=True Llama instance, used only for tokenization
        self._chat_template: Optional[jinja2.Template] = None
        self._bos_token: str = ""
        self.max_context_length: int = 8192

    def _resolve_chat_template_text(self, source: str, model_path: str, gguf_path: str) -> str:
        if source == "gguf":
            template = self._llm.metadata.get("tokenizer.chat_template")
            if not template:
                raise ValueError(
                    f"chat_template_source='gguf' but '{gguf_path}' has no "
                    "'tokenizer.chat_template' metadata. Use chat_template_source="
                    "'model_dir' or 'bfcl' via --ov-config instead."
                )
            return template

        if source == "model_dir":
            model_dir = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
            candidate = os.path.join(model_dir, "chat_template.jinja")
            if os.path.isfile(candidate):
                with open(candidate, encoding="utf-8") as template_file:
                    return template_file.read()

            raise FileNotFoundError(
                f"chat_template_source='model_dir' but no chat_template.jinja found in "
                f"'{model_dir}'. Use chat_template_source='bfcl' via --ov-config "
                "if you want this repo's own bfcl per-model fallback template instead."
            )

        # source == "bfcl"
        bfcl_template = find_bfcl_chat_template(model_path)
        if bfcl_template is None:
            raise ValueError(
                f"chat_template_source='bfcl' but no bfcl template matches '{model_path}'."
            )
        return bfcl_template

    def load(
        self, model_path: str, local_model_path: Optional[str], device_properties: dict  # noqa: ARG002
    ) -> None:
        from llama_cpp import Llama

        chat_template_source = device_properties.pop("chat_template_source", "gguf")
        if chat_template_source not in _CHAT_TEMPLATE_SOURCES:
            raise ValueError(
                f"Invalid chat_template_source '{chat_template_source}'. "
                f"Must be one of {_CHAT_TEMPLATE_SOURCES}."
            )

        gguf_path = _find_gguf_file(model_path)
        # Cheap vocab-only load (no weights) to read GGUF metadata (bos token,
        # context length, embedded chat template) and get a llama.cpp tokenizer for
        # `count_tokens`, independent of the full weight-loading Llama instance
        # LlamaCppHandler builds later in its own `_load_model`.
        self._llm = Llama(model_path=gguf_path, vocab_only=True, verbose=False)
        self._bos_token = self._llm.detokenize(
            [self._llm.token_bos()], special=True
        ).decode("utf-8", errors="replace")

        template_text = self._resolve_chat_template_text(chat_template_source, model_path, gguf_path)
        self._chat_template = _compile_chat_template(template_text)
        print(f"[INFO] Chat template source: {chat_template_source}")

        context_length = None
        for key, value in self._llm.metadata.items():
            if key.endswith(".context_length"):
                try:
                    context_length = int(value)
                except (TypeError, ValueError):
                    context_length = None
                break
        self.max_context_length = context_length or 8192

    def count_tokens(self, text: str) -> int:
        return len(self._llm.tokenize(text.encode("utf-8"), add_bos=False))

    def render_prompt(self, messages: list[dict]) -> str:  # noqa: ARG002
        raise NotImplementedError(
            "LlamaCppTokenizerAdapter only supports FC-style rendering "
            "(render_chat_prompt) - LlamaCppHandler only registers FC model names."
        )

    def render_chat_prompt(
        self, message: list[dict], tools: Optional[list[dict]], enable_thinking: bool
    ) -> str:
        return self._chat_template.render(
            messages=message,
            tools=tools if tools else None,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            bos_token=self._bos_token,
        )

    def unload(self) -> None:
        self._llm = None
        self._chat_template = None


class LlamaCppHandler(BaseOpenVINOHandler):
    """
    Handler for GGUF models using **llama-cpp-python** (``llama_cpp.Llama``) as the
    generation backend.

    Unlike ``OpenVINOGenAIHandler``/``OpenVINOGenAIVLMHandler`` (registry names
    ``openvino-genai-FC`` / ``openvino-genai-vlm-FC``), this handler does NOT use an
    HF ``AutoTokenizer`` at all - tokenization and chat-template rendering are
    delegated (composition, via ``BaseOpenVINOHandler``'s ``_tokenizer_adapter_cls``)
    to ``LlamaCppTokenizerAdapter`` above, instead of the default
    ``HFTokenizerAdapter``. See that class's docstring for how the chat template is
    resolved. The tool-call parsing of the raw generated text still reuses the same
    ``parse_openvino_fc_response`` as the openvino-genai handlers (see
    ``openvino_fc_support/parser_registry.py``), inherited unmodified via
    ``_parse_query_response_FC`` in ``BaseOpenVINOHandler``.

    This class itself only implements the two backend-specific hooks
    ``BaseOpenVINOHandler`` actually requires: ``_load_model`` (loads the ``.gguf``
    file via ``llama_cpp.Llama``, for generation) and ``_generate`` (runs completion
    via that ``Llama`` instance). No OpenVINO/openvino_genai involvement at all.

    Requirements
    ------------
    Install llama-cpp-python::

        pip install llama-cpp-python

    Model directory
    ----------------
    ``--local-model-path`` must point to a directory containing exactly one
    ``*.gguf`` file (or point directly at the ``.gguf`` file itself). No HF
    tokenizer/config files are required. If ``chat_template_source="model_dir"``
    is used, that directory (or its bfcl-template fallback, see
    ``openvino_fc_support/template_registry.find_bfcl_chat_template``) must also
    contain a ``chat_template.jinja`` file.

    ``--ov-config`` knobs
    ----------------------
    The generic ``--ov-config`` JSON/flat-dict mechanism (parsed by
    ``BaseOpenVINOHandler._parse_ov_config``) is reused here for:
      - ``chat_template_source``: ``"gguf"`` (default), ``"model_dir"``, or
        ``"bfcl"`` (see ``LlamaCppTokenizerAdapter``).
      - Any remaining keys are passed straight through as ``llama_cpp.Llama``
        constructor kwargs, e.g. ``{"n_ctx": 16384, "n_gpu_layers": -1,
        "n_threads": 8}``. Defaults:
          - ``n_ctx``: ``min(self.max_context_length, 8192)`` if not overridden.
          - ``n_gpu_layers``: ``-1`` (offload all layers) when
            ``--openvino-device GPU``, else ``0`` (CPU-only), unless overridden.

    Usage example
    -------------
    ::

        bfcl generate \\
            --model llamacpp-FC \\
            --local-model-path /path/to/gguf_model_dir \\
            --openvino-device CPU \\
            --ov-config '{"chat_template_source": "model_dir"}' \\
            --test-category multi_turn_base
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        registry_name: str,
        is_fc_model: bool,
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self._llm = None

    @override
    def _create_tokenizer_adapter(self) -> LlamaCppTokenizerAdapter:
        return LlamaCppTokenizerAdapter()

    @override
    def _load_model(
        self, model_path: str, device: str = "CPU", device_properties: Optional[dict] = None
    ) -> None:
        from llama_cpp import Llama

        device_properties = dict(device_properties or {})
        gguf_path = _find_gguf_file(model_path)

        n_ctx = device_properties.pop("n_ctx", None)
        if n_ctx is None:
            n_ctx = min(self.max_context_length or 8192, 8192)

        n_gpu_layers = device_properties.pop("n_gpu_layers", None)
        if n_gpu_layers is None:
            n_gpu_layers = -1 if device.upper().startswith("GPU") else 0

        print(
            f"[INFO] Loading GGUF model via llama-cpp-python: {gguf_path} "
            f"(n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers}, extra={device_properties})"
        )
        self._llm = Llama(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            **device_properties,
        )

    @override
    def _unload_model(self) -> None:
        self._llm = None

    @override
    def _generate(self, formatted_prompt: str, max_new_tokens: int) -> str:
        # `formatted_prompt` is already a fully rendered chat-template string (see
        # BaseOpenVINOHandler._query_FC), so this calls llama.cpp's raw completion
        # API directly (NOT its chat-completion helper, which would re-apply a
        # template on top). Matches OpenVINOGenAIHandler's
        # `config.apply_chat_template = False` behavior for the same reason.
        do_sample = self.temperature > 0.01
        output = self._llm(
            formatted_prompt,
            max_tokens=max_new_tokens,
            temperature=self.temperature if do_sample else 0.0,
            echo=False,
        )
        return output["choices"][0]["text"]

