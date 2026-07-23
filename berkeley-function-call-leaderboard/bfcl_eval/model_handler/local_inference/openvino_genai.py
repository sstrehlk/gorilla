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
            print(f"[INFO] Vision model detected. Using VLMPipeline. device_properties={device_properties}")
            self._pipeline = openvino_genai.VLMPipeline(model_path, device, **device_properties)
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
        config = self._pipeline.get_generation_config()
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
            # VLMPipeline: text-only inference (no images)
            result = self._pipeline.generate(formatted_prompt, generation_config=config)
            return str(result)

        # LLMPipeline.generate returns a str with only the newly generated text
        return self._pipeline.generate(formatted_prompt, config)
