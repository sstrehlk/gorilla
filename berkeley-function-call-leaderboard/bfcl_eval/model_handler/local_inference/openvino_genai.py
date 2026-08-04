from typing import Optional

from bfcl_eval.model_handler.local_inference.base_openvino_handler import (
    BaseOpenVINOHandler,
)
from overrides import override


class OpenVINOGenAIHandler(BaseOpenVINOHandler):
    """
    Handler for OpenVINO IR models using the **openvino-genai** library
    (``openvino_genai.LLMPipeline``).

    Requirements
    ------------
    Install the openvino-genai package::

        pip install openvino-genai

    Model directory
    ---------------
    The model directory must contain:
      - OpenVINO IR model files compatible with openvino-genai
        (``openvino_model.xml`` / ``openvino_model.bin`` produced e.g. by
        ``optimum-cli export openvino …`` or Optimum Intel's ``OVModelForCausalLM``)
      - Tokenizer files compatible with the ``openvino_tokenizers`` extension, i.e.
        ``openvino_tokenizer.xml`` / ``openvino_detokenizer.xml`` **or** standard
        HuggingFace tokenizer files (``tokenizer_config.json``, etc.)

    The ``transformers`` tokenizer that lives alongside the IR files is used for
    token-count bookkeeping (prompt / output token counts) while the actual text
    generation is driven by ``openvino_genai.LLMPipeline``.

    Usage example
    -------------
    ::

        bfcl generate \\
            --model Qwen/Qwen2.5-7B-Instruct-OV-genai \\
            --local-model-path /path/to/openvino_model_dir \\
            --openvino-device CPU \\
            --test-category simple

    Notes
    -----
    * ``openvino_genai.LLMPipeline`` accepts a filesystem path; it cannot fetch
      models from the HuggingFace Hub, so ``--local-model-path`` must always be
      provided.
    * Temperature-based sampling is enabled only when ``temperature > 0.01``; below
      that threshold greedy decoding is used (consistent with the BFCL default of
      ``temperature = 0.001``).
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
        self._pipeline = None

    @override
    def _load_model(
        self, model_path: str, device: str = "CPU", device_properties: Optional[dict] = None
    ) -> None:
        import openvino_genai
        import os

        device_properties = device_properties or {}

        # Detect VLM models by presence of vision embeddings model file
        is_vlm = os.path.exists(os.path.join(model_path, "openvino_vision_embeddings_model.xml"))
        if is_vlm:
            # Use ContinuousBatchingPipeline directly instead of VLMPipeline.
            # VLMPipeline's own prompt/embedding handling produces different
            # generation results than OVMS's serving engine even with an
            # identical (byte-for-byte, same token IDs) prompt and greedy
            # decoding - verified by a direct A/B test on multi_turn_base_0's
            # turn-0 request for gemma-4-26b-a4b-it: VLMPipeline (with or
            # without a scheduler_config) diverges from OVMS at the very first
            # generated tool-name token, while ContinuousBatchingPipeline used
            # directly (the same underlying pipeline class OVMS uses for its
            # "Visual Language Model Continuous Batching" servable) matches
            # OVMS's output exactly and reproducibly. Text-only (no image)
            # inputs are supported via the plain str overload of generate().
            print(f"[INFO] Vision model detected. Using ContinuousBatchingPipeline. device_properties={device_properties}")
            scheduler_config = openvino_genai.SchedulerConfig()
            scheduler_config.enable_prefix_caching = True
            self._pipeline = openvino_genai.ContinuousBatchingPipeline(
                model_path, scheduler_config, device, device_properties
            )
            self._is_vlm = True
            return

        self._is_vlm = False
        scheduler_config = openvino_genai.SchedulerConfig()
        scheduler_config.enable_prefix_caching = True
        pipeline_config = {"scheduler_config": scheduler_config, **device_properties}
        try:
            self._pipeline = openvino_genai.LLMPipeline(
                model_path, device, config=pipeline_config
            )
        except RuntimeError as e:
            if "unregistered_parameters" in str(e) or "beam_idx" in str(e) or "sampler_num_threads" in str(e):
                # Paged-attention backend doesn't support models with beam_idx
                # (e.g. stateless MoE models). Fall back to the simple pipeline.
                print(
                    f"[WARNING] PA backend failed ({e}). "
                    "Retrying without SchedulerConfig (no prefix caching)."
                )
                self._pipeline = openvino_genai.LLMPipeline(model_path, device, **device_properties)
            else:
                raise

    @override
    def _unload_model(self) -> None:
        self._pipeline = None

    @override
    def _generate(self, formatted_prompt: str, max_new_tokens: int) -> str:
        import openvino_genai

        # Start from the pipeline's own default GenerationConfig (loaded from the
        # model's generation_config.json at pipeline construction) instead of a
        # bare openvino_genai.GenerationConfig(). The bare default constructor has
        # eos_token_id = -1 and empty stop_token_ids, and since we pass this config
        # object directly to generate() (not via set_generation_config()), the
        # pipeline uses it as-is with no backfilling from the model's real EOS
        # token(s). That left generation with no stop condition other than
        # max_new_tokens, causing runaway/degenerate repetition until the token
        # budget was exhausted on every turn. Basing off get_generation_config()
        # preserves the model's eos_token_id/stop_token_ids (and sampling
        # defaults) while we still override what we need below.
        # ContinuousBatchingPipeline (used for VLM models, see _load_model) exposes
        # this via get_config() rather than get_generation_config().
        config = (
            self._pipeline.get_config()
            if getattr(self, "_is_vlm", False)
            else self._pipeline.get_generation_config()
        )
        config.max_new_tokens = max_new_tokens
        # `formatted_prompt` is already a fully rendered chat-template string
        # (built via self.tokenizer.apply_chat_template in _query_FC/_format_prompt).
        # openvino_genai.GenerationConfig.apply_chat_template defaults to True, which
        # causes LLMPipeline/ContinuousBatchingPipeline (pipeline_base.cpp) and
        # VLMPipeline (inputs_embedder.cpp) to wrap this already-rendered string as a
        # NEW {"role": "user", "content": formatted_prompt} message and apply the chat
        # template a SECOND time on top of it. OVMS explicitly sets this to false
        # (template is applied on the serving side). We must match that here to produce
        # identical model inputs and comparable accuracy results.
        config.apply_chat_template = False

        if self.temperature > 0.01:
            config.temperature = self.temperature
            config.do_sample = True
        else:
            config.do_sample = False

        if getattr(self, "_is_vlm", False):
            # ContinuousBatchingPipeline: text-only inference (no images).
            # The plain str overload of generate() returns a list[GenerationResult];
            # for this (text prompt in, not encoded input_ids) overload, m_generation_ids
            # already holds the decoded text candidate(s), not token ids.
            results = self._pipeline.generate(formatted_prompt, config)
            return results[0].m_generation_ids[0]

        # LLMPipeline.generate returns a str with only the newly generated text
        return self._pipeline.generate(formatted_prompt, config)


class OpenVINOGenAIVLMHandler(OpenVINOGenAIHandler):
    """
    Same as ``OpenVINOGenAIHandler``, but drives VLM models through
    ``openvino_genai.VLMPipeline`` instead of ``ContinuousBatchingPipeline``.

    Both pipeline classes are legitimate ways to run a VLM checkpoint through
    openvino-genai, and empirically they can disagree with each other (and with
    OVMS) on the same prompt/greedy-decoding config - see
    tmp/jira_repro_ticket.md for a minimal reproduction. This handler exists so
    both code paths can be run side by side (registry name
    ``openvino-genai-vlm-FC``) against identical testplans/datasets for
    accuracy comparison, without having to hand-edit the CB-based handler.

    Non-VLM models fall back to the parent class's LLMPipeline behavior
    unchanged.
    """

    @override
    def _load_model(
        self, model_path: str, device: str = "CPU", device_properties: Optional[dict] = None
    ) -> None:
        import openvino_genai
        import os

        device_properties = device_properties or {}

        is_vlm = os.path.exists(os.path.join(model_path, "openvino_vision_embeddings_model.xml"))
        if not is_vlm:
            super()._load_model(model_path, device, device_properties)
            return

        print(f"[INFO] Vision model detected. Using VLMPipeline. device_properties={device_properties}")
        self._pipeline = openvino_genai.VLMPipeline(model_path, device, **device_properties)
        self._is_vlm = True

    @override
    def _generate(self, formatted_prompt: str, max_new_tokens: int) -> str:
        if not getattr(self, "_is_vlm", False):
            return super()._generate(formatted_prompt, max_new_tokens)

        config = self._pipeline.get_generation_config()
        config.max_new_tokens = max_new_tokens
        # See OpenVINOGenAIHandler._generate: prompt is already fully rendered,
        # so the pipeline must not apply the chat template a second time.
        config.apply_chat_template = False

        if self.temperature > 0.01:
            config.temperature = self.temperature
            config.do_sample = True
        else:
            config.do_sample = False

        # Text-only inference (no images/videos) for BFCL FC test entries.
        result = self._pipeline.generate(formatted_prompt, images=[], generation_config=config)
        return result.texts[0]
