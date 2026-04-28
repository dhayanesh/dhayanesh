from __future__ import annotations

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

if os.getenv("ENABLE_ARCTIC_GLOBAL_PLUGIN", "0") != "1":
    # vLLM 0.19 already has native suffix decoding that imports Arctic's
    # SuffixDecodingCache. The separate Arctic global vLLM plugin installed on
    # this machine is pinned to older vLLM releases, so keep it disabled unless
    # explicitly requested.
    os.environ["ARCTIC_INFERENCE_ENABLED"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from fastapi import FastAPI, HTTPException
from vllm import AsyncEngineArgs, SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine

from novel_qwen_serve.config import AppConfig
from novel_qwen_serve.schemas import NovelRequest, NovelResponse
from novel_qwen_serve.shared_suffix_actor import get_or_create_shared_suffix_actor
from novel_qwen_serve.shared_suffix_cache import (
    get_local_shared_cache_stats,
    install_shared_suffix_cache,
)

api = FastAPI(title="Novel Qwen Serve", version="0.1.0")


@api.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True}


try:
    from ray import serve
except ImportError as exc:  # pragma: no cover - import guard for static checks
    raise RuntimeError("Ray Serve is required. Install with `pip install -r requirements.txt`.") from exc


@serve.deployment
@serve.ingress(api)
class NovelQwenDeployment:
    def __init__(self, config: AppConfig, shared_suffix_actor: Any):
        self.config = config
        self.replica_id = f"replica-{uuid.uuid4().hex[:8]}"
        self.shared_suffix_actor = shared_suffix_actor
        self._export_suffix_patch_env(config)

        install_shared_suffix_cache(
            shared_actor=shared_suffix_actor,
            replica_id=self.replica_id,
            prompt_suffix_tokens=config.shared_prompt_suffix_tokens,
            min_response_tokens=config.shared_min_response_tokens,
            sync_interval_s=config.shared_sync_interval_s,
            sync_batch=config.shared_sync_batch,
            sync_timeout_s=config.shared_sync_timeout_s,
        )

        engine_args = AsyncEngineArgs(**config.vllm_engine_kwargs())
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    def _export_suffix_patch_env(self, config: AppConfig) -> None:
        workspace_root = str(Path(__file__).resolve().parents[1])
        python_path = os.environ.get("PYTHONPATH", "")
        if workspace_root not in python_path.split(os.pathsep):
            os.environ["PYTHONPATH"] = (
                workspace_root if not python_path else f"{workspace_root}{os.pathsep}{python_path}"
            )
        os.environ["NOVEL_SHARED_SUFFIX_PATCH"] = "1"
        os.environ["NOVEL_SHARED_SUFFIX_ACTOR_NAME"] = config.shared_actor_name
        os.environ["NOVEL_SHARED_SUFFIX_RAY_NAMESPACE"] = "serve"
        os.environ["NOVEL_SHARED_SUFFIX_REPLICA_ID"] = self.replica_id
        os.environ["NOVEL_SHARED_SUFFIX_PROMPT_TOKENS"] = str(
            config.shared_prompt_suffix_tokens
        )
        os.environ["NOVEL_SHARED_SUFFIX_MIN_RESPONSE_TOKENS"] = str(
            config.shared_min_response_tokens
        )
        os.environ["NOVEL_SHARED_SUFFIX_SYNC_INTERVAL_S"] = str(
            config.shared_sync_interval_s
        )
        os.environ["NOVEL_SHARED_SUFFIX_SYNC_BATCH"] = str(config.shared_sync_batch)
        os.environ["NOVEL_SHARED_SUFFIX_SYNC_TIMEOUT_S"] = str(
            config.shared_sync_timeout_s
        )
        os.environ["NOVEL_SHARED_SUFFIX_USE_OBJECT_STORE"] = str(
            config.shared_use_object_store
        )

    @api.post("/novel", response_model=NovelResponse)
    async def novel_endpoint(self, request: NovelRequest) -> NovelResponse:
        return await self.generate(request)

    @api.get("/suffix/stats")
    async def suffix_stats_endpoint(self) -> dict[str, Any]:
        return await self.suffix_stats()

    async def generate(self, request: NovelRequest) -> NovelResponse:
        request_id = request.request_id or f"novel-{uuid.uuid4().hex}"
        sampling_params = SamplingParams(
            temperature=(
                self.config.default_temperature
                if request.temperature is None
                else request.temperature
            ),
            top_p=self.config.default_top_p if request.top_p is None else request.top_p,
            top_k=self.config.default_top_k if request.top_k is None else request.top_k,
            max_tokens=(
                self.config.default_max_tokens if request.max_tokens is None else request.max_tokens
            ),
            stop=request.stop,
            seed=request.seed,
        )

        started = time.perf_counter()
        first_token_at: float | None = None
        final_output = None
        try:
            async for output in self.engine.generate(
                request.prompt,
                sampling_params,
                request_id=request_id,
            ):
                if first_token_at is None and output.outputs and output.outputs[0].token_ids:
                    first_token_at = time.perf_counter()
                final_output = output
        except asyncio.CancelledError:
            await self.engine.abort(request_id)
            raise
        except Exception as exc:
            await self.engine.abort(request_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if final_output is None or not final_output.outputs:
            raise HTTPException(status_code=500, detail="vLLM returned no output")

        ended = time.perf_counter()
        choice = final_output.outputs[0]
        prompt_tokens = len(final_output.prompt_token_ids or [])
        completion_tokens = len(choice.token_ids or [])
        if request.suffix_share and self.config.shared_api_publish:
            self._publish_completed_suffix(
                request_id,
                final_output.prompt_token_ids or [],
                choice.token_ids or [],
            )
        timings = {
            "total": round((ended - started) * 1000, 3),
            "time_to_first_token": round(
                ((first_token_at or ended) - started) * 1000,
                3,
            ),
        }

        suffix_stats = await self.suffix_stats(local_only=True)
        return NovelResponse(
            id=request_id,
            model=self.config.served_model_name,
            replica_id=self.replica_id,
            text=choice.text,
            finish_reason=getattr(choice, "finish_reason", None),
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            timings_ms=timings,
            suffix_cache=suffix_stats,
        )

    def _publish_completed_suffix(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        output_token_ids: list[int],
    ) -> None:
        if len(output_token_ids) < self.config.shared_min_response_tokens:
            return
        prompt_tail = list(prompt_token_ids)[-self.config.shared_prompt_suffix_tokens :]
        token_ids = np.asarray(prompt_tail + list(output_token_ids), dtype=np.int32)
        try:
            self.shared_suffix_actor.add_sequence.remote(
                self.replica_id,
                request_id,
                token_ids,
            )
        except Exception:
            pass

    async def suffix_stats(self, local_only: bool = False) -> dict[str, Any]:
        local = {
            "replica_id": self.replica_id,
            "model": self.config.model_id,
            "shared_sync_interval_s": self.config.shared_sync_interval_s,
            "cache_instances": get_local_shared_cache_stats(),
        }
        if local_only:
            return local
        try:
            import ray

            shared = await asyncio.to_thread(
                ray.get,
                self.shared_suffix_actor.stats.remote(),
            )
        except Exception as exc:
            shared = {"error": str(exc)}
        return {"local": local, "shared": shared}


def build_app(config: AppConfig | None = None):
    config = config or AppConfig.from_env()
    shared_suffix_actor = get_or_create_shared_suffix_actor(config)
    return NovelQwenDeployment.options(
        name="NovelQwenDeployment",
        num_replicas=config.num_replicas,
        max_ongoing_requests=config.max_ongoing_requests,
        ray_actor_options={
            "num_cpus": config.cpus_per_replica,
            "num_gpus": config.gpu_fraction_per_replica,
        },
    ).bind(config, shared_suffix_actor)


def main() -> None:
    import ray

    config = AppConfig.from_env()
    if not ray.is_initialized():
        ray.init(namespace="serve")
    serve.start(http_options={"host": config.host, "port": config.port}, detached=False)
    serve.run(build_app(config), route_prefix="/")
    print(
        f"Serving {config.served_model_name} from {config.model_id} at "
        f"http://{config.host}:{config.port}/novel with {config.num_replicas} replicas."
    )
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()
