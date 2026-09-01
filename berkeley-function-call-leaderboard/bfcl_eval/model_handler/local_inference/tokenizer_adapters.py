from typing import Optional, Protocol, runtime_checkable

from bfcl_eval.model_handler.local_inference.openvino_fc_support.template_registry import (
    find_bfcl_chat_template,
)

# Passed via --ov-config as {"chat_template_source": "model_dir"|"bfcl"}.
_HF_CHAT_TEMPLATE_SOURCES = ("model_dir", "bfcl")


@runtime_checkable
class TokenizerAdapter(Protocol):
    """Pluggable tokenization + chat-template-rendering strategy for
    ``BaseOpenVINOHandler`` (composition, not inheritance).

    ``BaseOpenVINOHandler`` and its FC/prompting pipeline (``_query_FC``,
    ``_query_prompting``, ``_format_prompt``) depend only on this interface, never
    on a concrete tokenizer implementation. This lets backends with fundamentally
    different tokenizers (HF ``AutoTokenizer`` vs llama.cpp's GGUF-embedded vocab)
    plug into the exact same shared pipeline without the handler subclass needing
    to override any of that pipeline's internals - it only ever selects *which*
    adapter to use, via the ``_tokenizer_adapter_cls`` class attribute.
    """

    max_context_length: int

    def load(
        self, model_path: str, local_model_path: Optional[str], device_properties: dict
    ) -> None:
        """Load whatever backing tokenizer/config is needed and set
        ``self.max_context_length``. ``device_properties`` is the same mutable dict
        later passed to the handler's ``_load_model`` - adapters may ``pop()``
        adapter-specific keys out of it (e.g. ``chat_template_source``) before that
        happens."""
        ...

    def count_tokens(self, text: str) -> int: ...

    def render_prompt(self, messages: list[dict]) -> str:
        """Render a plain (non-FC, no tools) chat prompt. Used by prompting mode."""
        ...

    def render_chat_prompt(
        self, message: list[dict], tools: Optional[list[dict]], enable_thinking: bool
    ) -> str:
        """Render an FC chat prompt, embedding ``tools`` if the template supports it."""
        ...

    def unload(self) -> None: ...


class HFTokenizerAdapter:
    """Default adapter: ``transformers.AutoTokenizer``/``AutoConfig``. Used by every
    ``BaseOpenVINOHandler`` subclass that doesn't set a different
    ``_tokenizer_adapter_cls`` (``OpenVINOGenAIHandler``, ``OpenVINOGenAIVLMHandler``,
    ``OpenVINOOptimumHandler``).

    The chat template is selectable via ``chat_template_source`` (read out of the
    ``device_properties`` dict passed to ``load()``, i.e. via ``--ov-config``):
    ``"model_dir"`` (default) requires ``AutoTokenizer.from_pretrained`` to have
    auto-discovered a ``chat_template.jinja``/``tokenizer_config.json``-embedded
    template in the model directory; ``"bfcl"`` forces this repo's own bundled
    per-model template instead, overriding whatever ``model_dir`` shipped.

    The raw ``tokenizer`` attribute (a real HF ``PreTrainedTokenizerBase``) is kept
    public for handlers that need lower-level access beyond this adapter's interface
    (e.g. ``OpenVINOOptimumHandler._generate`` calls ``self.tokenizer(...)``/
    ``.decode()`` directly to drive ``OVModelForCausalLM.generate()``).
    """

    def __init__(self) -> None:
        self.tokenizer = None
        self.max_context_length: int = 4096

    def load(
        self, model_path: str, local_model_path: Optional[str], device_properties: dict
    ) -> None:
        from transformers import AutoConfig, AutoTokenizer

        chat_template_source = device_properties.pop("chat_template_source", "model_dir")
        if chat_template_source not in _HF_CHAT_TEMPLATE_SOURCES:
            raise ValueError(
                f"Invalid chat_template_source '{chat_template_source}'. "
                f"Must be one of {_HF_CHAT_TEMPLATE_SOURCES}."
            )

        load_kwargs: dict = {
            "pretrained_model_name_or_path": model_path,
            "trust_remote_code": True,
        }
        if local_model_path is not None:
            load_kwargs["local_files_only"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(**load_kwargs)
        config = AutoConfig.from_pretrained(**load_kwargs)
        self._apply_chat_template_source(chat_template_source, model_path)
        print(f"[INFO] Chat template source: {chat_template_source}")

        if hasattr(config, "max_position_embeddings"):
            self.max_context_length = config.max_position_embeddings
        elif hasattr(config, "text_config") and hasattr(
            config.text_config, "max_position_embeddings"
        ):
            # Multimodal configs (e.g. gemma-4's Gemma4Config) nest the text
            # decoder's own max_position_embeddings under text_config instead of
            # exposing it top-level - without this branch every such model fell
            # through to the 4096 fallback below (gemma-4-26b-a4b-it's real value
            # is 262144), silently starving max_new_tokens on long conversations.
            self.max_context_length = config.text_config.max_position_embeddings
        elif (
            self.tokenizer.model_max_length is not None
            and self.tokenizer.model_max_length < 1_000_000
        ):
            self.max_context_length = self.tokenizer.model_max_length
        else:
            self.max_context_length = 4096  # safe fallback

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.tokenize(text))

    def _apply_chat_template_source(self, chat_template_source: str, model_path: str) -> None:
        if chat_template_source == "model_dir":
            if not self.tokenizer.chat_template:
                raise FileNotFoundError(
                    f"chat_template_source='model_dir' but '{model_path}' has no "
                    "chat template (no chat_template.jinja / tokenizer_config.json "
                    "'chat_template' key found by AutoTokenizer). Use "
                    "chat_template_source='bfcl' via --ov-config instead."
                )
            return

        # chat_template_source == "bfcl"
        bfcl_template = find_bfcl_chat_template(model_path)
        if bfcl_template is None:
            raise ValueError(
                f"chat_template_source='bfcl' but no bfcl template matches '{model_path}'."
            )
        self.tokenizer.chat_template = bfcl_template

    def render_prompt(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def render_chat_prompt(
        self, message: list[dict], tools: Optional[list[dict]], enable_thinking: bool
    ) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                message,
                tools=tools if tools else None,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=enable_thinking,
            )
        except Exception:
            # Fallback: template does not support tools - render without
            return self.tokenizer.apply_chat_template(
                message, add_generation_prompt=True, tokenize=False
            )

    def unload(self) -> None:
        self.tokenizer = None
