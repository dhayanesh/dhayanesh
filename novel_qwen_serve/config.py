from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _get_optional_bytes(byte_name: str, gb_name: str, default: int | None) -> int | None:
    byte_value = os.getenv(byte_name)
    if byte_value:
        if byte_value.lower() in {"0", "none", "null", "auto"}:
            return None
        return int(byte_value)
    gb_value = os.getenv(gb_name)
    if gb_value:
        if gb_value.lower() in {"0", "none", "null", "auto"}:
            return None
        return int(float(gb_value) * 1024**3)
    return default


@dataclass(frozen=True)
class AppConfig:
    model_id: str = "Qwen/Qwen3-0.6B"
    served_model_name: str = "qwen-small-suffix"

    num_replicas: int = 5
    cpus_per_replica: float = 2.0
    gpu_fraction_per_replica: float = 0.18
    max_ongoing_requests: int = 64

    host: str = "0.0.0.0"
    port: int = 8000

    dtype: str = "bfloat16"
    max_model_len: int = 8192
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 8192
    gpu_memory_utilization: float = 0.14
    kv_cache_memory_bytes: int | None = 2 * 1024**3
    max_cudagraph_capture_size: int = 64
    enforce_eager: bool = False
    tensor_parallel_size: int = 1
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    trust_remote_code: bool = True

    suffix_num_speculative_tokens: int = 8
    suffix_tree_depth: int = 64
    suffix_max_cached_requests: int = 50_000
    suffix_max_spec_factor: float = 1.0
    suffix_min_token_prob: float = 0.10

    shared_actor_name: str = "novel-shared-suffix-corpus"
    shared_max_sequences: int = 75_000
    shared_max_total_tokens: int = 12_000_000
    shared_max_sequence_tokens: int = 4096
    shared_prompt_suffix_tokens: int = 512
    shared_min_response_tokens: int = 8
    shared_sync_interval_s: float = 0.75
    shared_sync_batch: int = 512
    shared_sync_timeout_s: float = 0.10
    shared_use_object_store: bool = True
    shared_api_publish: bool = False

    default_max_tokens: int = 512
    default_temperature: float = 0.2
    default_top_p: float = 0.95
    default_top_k: int = 0

    @classmethod
    def from_env(cls) -> "AppConfig":
        replicas = _get_int("NUM_REPLICAS", cls.num_replicas)
        gpu_fraction = _get_float(
            "GPU_FRACTION_PER_REPLICA",
            min(cls.gpu_fraction_per_replica, 0.92 / max(replicas, 1)),
        )
        gpu_mem = _get_float(
            "GPU_MEMORY_UTILIZATION",
            min(cls.gpu_memory_utilization, 0.78 / max(replicas, 1)),
        )
        return cls(
            model_id=os.getenv("MODEL_ID", cls.model_id),
            served_model_name=os.getenv("SERVED_MODEL_NAME", cls.served_model_name),
            num_replicas=replicas,
            cpus_per_replica=_get_float("CPUS_PER_REPLICA", cls.cpus_per_replica),
            gpu_fraction_per_replica=gpu_fraction,
            max_ongoing_requests=_get_int("MAX_ONGOING_REQUESTS", cls.max_ongoing_requests),
            host=os.getenv("HOST", cls.host),
            port=_get_int("PORT", cls.port),
            dtype=os.getenv("DTYPE", cls.dtype),
            max_model_len=_get_int("MAX_MODEL_LEN", cls.max_model_len),
            max_num_seqs=_get_int("MAX_NUM_SEQS", cls.max_num_seqs),
            max_num_batched_tokens=_get_int(
                "MAX_NUM_BATCHED_TOKENS", cls.max_num_batched_tokens
            ),
            gpu_memory_utilization=gpu_mem,
            kv_cache_memory_bytes=_get_optional_bytes(
                "KV_CACHE_MEMORY_BYTES",
                "KV_CACHE_MEMORY_GB",
                cls.kv_cache_memory_bytes,
            ),
            max_cudagraph_capture_size=_get_int(
                "MAX_CUDAGRAPH_CAPTURE_SIZE", cls.max_cudagraph_capture_size
            ),
            enforce_eager=_get_bool("ENFORCE_EAGER", cls.enforce_eager),
            tensor_parallel_size=_get_int("TENSOR_PARALLEL_SIZE", cls.tensor_parallel_size),
            enable_prefix_caching=_get_bool(
                "ENABLE_PREFIX_CACHING", cls.enable_prefix_caching
            ),
            enable_chunked_prefill=_get_bool(
                "ENABLE_CHUNKED_PREFILL", cls.enable_chunked_prefill
            ),
            trust_remote_code=_get_bool("TRUST_REMOTE_CODE", cls.trust_remote_code),
            suffix_num_speculative_tokens=_get_int(
                "SUFFIX_NUM_SPECULATIVE_TOKENS", cls.suffix_num_speculative_tokens
            ),
            suffix_tree_depth=_get_int("SUFFIX_TREE_DEPTH", cls.suffix_tree_depth),
            suffix_max_cached_requests=_get_int(
                "SUFFIX_MAX_CACHED_REQUESTS", cls.suffix_max_cached_requests
            ),
            suffix_max_spec_factor=_get_float(
                "SUFFIX_MAX_SPEC_FACTOR", cls.suffix_max_spec_factor
            ),
            suffix_min_token_prob=_get_float(
                "SUFFIX_MIN_TOKEN_PROB", cls.suffix_min_token_prob
            ),
            shared_actor_name=os.getenv("SHARED_SUFFIX_ACTOR_NAME", cls.shared_actor_name),
            shared_max_sequences=_get_int("SHARED_MAX_SEQUENCES", cls.shared_max_sequences),
            shared_max_total_tokens=_get_int(
                "SHARED_MAX_TOTAL_TOKENS", cls.shared_max_total_tokens
            ),
            shared_max_sequence_tokens=_get_int(
                "SHARED_MAX_SEQUENCE_TOKENS", cls.shared_max_sequence_tokens
            ),
            shared_prompt_suffix_tokens=_get_int(
                "SHARED_PROMPT_SUFFIX_TOKENS", cls.shared_prompt_suffix_tokens
            ),
            shared_min_response_tokens=_get_int(
                "SHARED_MIN_RESPONSE_TOKENS", cls.shared_min_response_tokens
            ),
            shared_sync_interval_s=_get_float(
                "SHARED_SYNC_INTERVAL_S", cls.shared_sync_interval_s
            ),
            shared_sync_batch=_get_int("SHARED_SYNC_BATCH", cls.shared_sync_batch),
            shared_sync_timeout_s=_get_float(
                "SHARED_SYNC_TIMEOUT_S", cls.shared_sync_timeout_s
            ),
            shared_use_object_store=_get_bool(
                "SHARED_SUFFIX_USE_OBJECT_STORE", cls.shared_use_object_store
            ),
            shared_api_publish=_get_bool(
                "SHARED_SUFFIX_API_PUBLISH", cls.shared_api_publish
            ),
            default_max_tokens=_get_int("DEFAULT_MAX_TOKENS", cls.default_max_tokens),
            default_temperature=_get_float(
                "DEFAULT_TEMPERATURE", cls.default_temperature
            ),
            default_top_p=_get_float("DEFAULT_TOP_P", cls.default_top_p),
            default_top_k=_get_int("DEFAULT_TOP_K", cls.default_top_k),
        )

    def vllm_engine_kwargs(self) -> dict[str, Any]:
        cudagraph_sizes = sorted(
            {
                size
                for size in [1, 2, 4, 8, 16, 32, 64, self.max_cudagraph_capture_size]
                if size <= self.max_cudagraph_capture_size
            }
        )
        return {
            "model": self.model_id,
            "served_model_name": self.served_model_name,
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "kv_cache_memory_bytes": self.kv_cache_memory_bytes,
            "max_cudagraph_capture_size": self.max_cudagraph_capture_size,
            "cudagraph_capture_sizes": cudagraph_sizes,
            "enforce_eager": self.enforce_eager,
            "tensor_parallel_size": self.tensor_parallel_size,
            "enable_prefix_caching": self.enable_prefix_caching,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "trust_remote_code": self.trust_remote_code,
            "disable_log_stats": False,
            "speculative_config": {
                "method": "suffix",
                "num_speculative_tokens": self.suffix_num_speculative_tokens,
                "suffix_decoding_max_tree_depth": self.suffix_tree_depth,
                "suffix_decoding_max_cached_requests": self.suffix_max_cached_requests,
                "suffix_decoding_max_spec_factor": self.suffix_max_spec_factor,
                "suffix_decoding_min_token_prob": self.suffix_min_token_prob,
            },
        }
