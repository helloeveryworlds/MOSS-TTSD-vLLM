import argparse
import base64
import io
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from generation_utils import (
    MAX_CHANNELS,
    load_tokenizer_and_spt,
    process_batch,
)

MODEL_PATH = "fnlp/MOSS-TTSD-v0.5"
SYSTEM_PROMPT = "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text."
SPT_CONFIG_PATH = "XY_Tokenizer/config/xy_tokenizer_32k_config.yaml"
SPT_CHECKPOINT_PATH = "XY_Tokenizer/weights/xy_tokenizer.ckpt"


class GenerationRequest(BaseModel):
    text: str
    prompt_text: Optional[str] = None
    prompt_audio: Optional[str] = None
    prompt_audio_speaker1: Optional[str] = None
    prompt_text_speaker1: Optional[str] = None
    prompt_audio_speaker2: Optional[str] = None
    prompt_text_speaker2: Optional[str] = None
    use_normalize: Optional[bool] = None
    silence_duration: Optional[float] = None


@dataclass
class ServerState:
    engine: Any
    tokenizer: Any
    spt: Any
    device: str
    sampling_params: Any
    max_new_tokens: Optional[int]
    system_prompt: str
    default_use_normalize: bool
    default_silence_duration: float


def decode_audio_base64(audio_b64: str) -> Optional[Tuple[torch.Tensor, int]]:
    if not audio_b64:
        return None
    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception as exc:  # pragma: no cover - input validation
        raise ValueError("Failed to decode base64 audio data") from exc

    with io.BytesIO(audio_bytes) as buf:
        try:
            audio, sample_rate = sf.read(buf, dtype="float32")
        except Exception as exc:  # pragma: no cover - invalid audio
            raise ValueError("Invalid audio data provided") from exc

    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=1)

    audio_tensor = torch.from_numpy(audio.T)
    return audio_tensor, sample_rate


def build_item_from_request(payload: GenerationRequest) -> Dict[str, Any]:
    item: Dict[str, Any] = {"text": payload.text}

    if payload.prompt_text:
        item["prompt_text"] = payload.prompt_text

    if payload.prompt_audio:
        item["prompt_audio"] = decode_audio_base64(payload.prompt_audio)

    if payload.prompt_audio_speaker1:
        item["prompt_audio_speaker1"] = decode_audio_base64(payload.prompt_audio_speaker1)
    if payload.prompt_text_speaker1:
        item["prompt_text_speaker1"] = payload.prompt_text_speaker1

    if payload.prompt_audio_speaker2:
        item["prompt_audio_speaker2"] = decode_audio_base64(payload.prompt_audio_speaker2)
    if payload.prompt_text_speaker2:
        item["prompt_text_speaker2"] = payload.prompt_text_speaker2

    return item


def build_state(args: argparse.Namespace) -> ServerState:
    from vllm import LLM, SamplingParams

    dtype_mapping = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }

    tokenizer, spt = load_tokenizer_and_spt(
        model_path=args.model_path,
        spt_config_path=args.spt_config,
        spt_checkpoint_path=args.spt_checkpoint,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    spt = spt.to(device)
    spt.eval()

    engine = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        dtype=dtype_mapping[args.dtype],
    )

    max_tokens = args.max_new_tokens * MAX_CHANNELS if args.max_new_tokens else None
    top_k = args.top_k if args.top_k >= 0 else None

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=top_k,
        repetition_penalty=args.repetition_penalty,
        max_tokens=max_tokens,
    )

    return ServerState(
        engine=engine,
        tokenizer=tokenizer,
        spt=spt,
        device=device,
        sampling_params=sampling_params,
        max_new_tokens=args.max_new_tokens,
        system_prompt=args.system_prompt,
        default_use_normalize=args.default_use_normalize,
        default_silence_duration=args.silence_duration,
    )


def create_app(state: ServerState) -> FastAPI:
    app = FastAPI(title="MOSS-TTSD vLLM Server")

    @app.post("/generate_audio")
    async def generate_audio(payload: GenerationRequest):
        try:
            item = build_item_from_request(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        use_normalize = (
            payload.use_normalize
            if payload.use_normalize is not None
            else state.default_use_normalize
        )
        silence_duration = (
            payload.silence_duration
            if payload.silence_duration is not None
            else state.default_silence_duration
        )

        try:
            _, audio_results = process_batch(
                batch_items=[item],
                tokenizer=state.tokenizer,
                model=None,
                spt=state.spt,
                device=state.device,
                system_prompt=state.system_prompt,
                start_idx=0,
                use_normalize=use_normalize,
                silence_duration=silence_duration,
                vllm_engine=state.engine,
                vllm_sampling_params=state.sampling_params,
                max_new_tokens=state.max_new_tokens,
            )
        except Exception as exc:  # pragma: no cover - runtime errors
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if not audio_results or audio_results[0] is None:
            raise HTTPException(status_code=500, detail="Audio generation failed")

        result = audio_results[0]
        audio_tensor = result["audio_data"].cpu()
        sample_rate = int(result["sample_rate"])

        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor, sample_rate, format="wav")
        buffer.seek(0)

        headers = {
            "sample_rate": str(sample_rate),
        }

        usage = result.get("token_usage")
        if usage:
            headers.update(
                {
                    "prompt_tokens": str(usage.get("prompt_tokens", 0)),
                    "completion_tokens": str(usage.get("completion_tokens", 0)),
                    "total_tokens": str(usage.get("total_tokens", 0)),
                }
            )

        return Response(content=buffer.getvalue(), media_type="audio/wav", headers=headers)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MOSS-TTSD vLLM inference server")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--spt-config", default=SPT_CONFIG_PATH)
    parser.add_argument("--spt-checkpoint", default=SPT_CHECKPOINT_PATH)
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1, help="Set to a non-negative value to enable top-k sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=20000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--default-use-normalize", action="store_true")
    parser.add_argument("--silence-duration", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    state = build_state(args)
    app = create_app(state)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, workers=args.workers)


if __name__ == "__main__":
    main()
