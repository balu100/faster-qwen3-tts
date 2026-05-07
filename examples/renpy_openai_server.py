#!/usr/bin/env python3
"""
Ren'Py-friendly OpenAI-compatible TTS API server for faster-qwen3-tts.

This keeps the same POST /v1/audio/speech shape as examples/openai_server.py,
but is tuned for Ren'Py self-voicing playback:

- every request gets a numeric request id in logs and response headers
- a newer request cancels the previous active request immediately
- client disconnects cancel server-side generation at the next stream chunk
- logs include full input text, voice selection, queue/lock wait, chunks, and timing

Usage:
    python examples/renpy_openai_server.py \
        --voices voices.json \
        --port 8000

RenPyUniversalTTS should point its OpenAI URL at:
    http://127.0.0.1:8000/v1/audio/speech

For lowest latency with ffplay raw PCM mode, use response_format="pcm",
audio.raw_pcm=true, audio.sample_rate=24000, and audio.sample_format="s16le".
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import queue
import re
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger("renpy_openai_server")

app = FastAPI(title="faster-qwen3-tts Ren'Py OpenAI-compatible API")

tts_model = None
voices: dict[str, dict[str, Any]] = {}
default_voice: Optional[str] = None
SAMPLE_RATE = 24000

_model_lock = threading.Lock()
_state_lock = threading.RLock()
_request_counter = 0
_active_job: Optional["GenerationJob"] = None
_DONE = object()


@dataclass
class ServerSettings:
    chunk_size: int = 8
    max_new_tokens: int = 360
    xvec_only: bool = True
    latest_only: bool = True


settings = ServerSettings()


@dataclass
class GenerationJob:
    id: int
    text: str
    voice_name: str
    fmt: str
    client: str
    queue: queue.Queue = field(default_factory=queue.Queue)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    created_at: float = field(default_factory=time.perf_counter)
    model_started_at: Optional[float] = None
    cancel_reason: Optional[str] = None
    chunks: int = 0
    audio_seconds: float = 0.0
    sample_rate: int = SAMPLE_RATE
    _done_sent: bool = False
    _finish_logged: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def cancelled(self) -> bool:
        return self.cancel_event.is_set()


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "alloy"
    response_format: str = "wav"
    speed: float = 1.0
    stream: Optional[bool] = None

    # Optional faster-qwen3-tts overrides. RenPyUniversalTTS can send these
    # through openai.extra_body when a game needs per-request tuning.
    language: Optional[str] = None
    chunk_size: Optional[int] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    xvec_only: Optional[bool] = None
    non_streaming_mode: Optional[bool] = None
    instruct: Optional[str] = None


def _configure_logging(level_name: str, log_file: Optional[str]) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)


def _client_host(request: Request) -> str:
    if request.client is None:
        return "-"
    return request.client.host


def _preview(text: str, limit: int = 120) -> str:
    value = re.sub(r"\s+", " ", text).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _to_pcm16(pcm: np.ndarray) -> bytes:
    return np.clip(pcm * 32768, -32768, 32767).astype(np.int16).tobytes()


def _wav_header(sample_rate: int, data_len: int = 0xFFFFFFFF) -> bytes:
    channels = 1
    bits = 16
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    riff_size = 0xFFFFFFFF if data_len == 0xFFFFFFFF else 36 + data_len

    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", riff_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_len))
    return buf.getvalue()


def _to_wav_bytes(raw_pcm: bytes, sample_rate: int) -> bytes:
    return _wav_header(sample_rate, len(raw_pcm)) + raw_pcm


def _to_mp3_bytes(raw_pcm: bytes, sample_rate: int) -> bytes:
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise HTTPException(
            status_code=400,
            detail="response_format='mp3' requires pydub and ffmpeg: pip install pydub",
        ) from exc

    segment = AudioSegment(
        raw_pcm,
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    buf = io.BytesIO()
    segment.export(buf, format="mp3")
    return buf.getvalue()


def _coerce_int(value: Any, default: int, minimum: int, name: str) -> int:
    if value is None:
        return default

    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{name} must be an integer") from exc

    if coerced < minimum:
        raise HTTPException(status_code=400, detail=f"{name} must be >= {minimum}")

    return coerced


def _coerce_float(value: Any, default: float, minimum: Optional[float], name: str) -> float:
    if value is None:
        return default

    try:
        coerced = float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{name} must be a number") from exc

    if minimum is not None and coerced < minimum:
        raise HTTPException(status_code=400, detail=f"{name} must be >= {minimum}")

    return coerced


def resolve_voice(voice_name: str, request_id: Optional[int] = None) -> tuple[str, dict[str, Any]]:
    if voice_name in voices:
        return voice_name, voices[voice_name]

    if default_voice and default_voice in voices:
        logger.warning(
            "request=%s voice=%r not configured; falling back to default=%r",
            request_id if request_id is not None else "-",
            voice_name,
            default_voice,
        )
        return default_voice, voices[default_voice]

    raise HTTPException(
        status_code=400,
        detail=f"Voice {voice_name!r} is not configured. Available voices: {list(voices.keys())}",
    )


def _voice_value(
    req_value: Any,
    voice_cfg: dict[str, Any],
    key: str,
    default: Any,
) -> Any:
    if req_value is not None:
        return req_value
    if key in voice_cfg:
        return voice_cfg[key]
    return default


def _generation_kwargs(req: SpeechRequest, voice_cfg: dict[str, Any]) -> dict[str, Any]:
    chunk_size = _coerce_int(
        _voice_value(req.chunk_size, voice_cfg, "chunk_size", settings.chunk_size),
        settings.chunk_size,
        1,
        "chunk_size",
    )
    max_new_tokens = _coerce_int(
        _voice_value(req.max_new_tokens, voice_cfg, "max_new_tokens", settings.max_new_tokens),
        settings.max_new_tokens,
        2,
        "max_new_tokens",
    )

    return {
        "language": _voice_value(req.language, voice_cfg, "language", "Auto"),
        "ref_audio": voice_cfg["ref_audio"],
        "ref_text": voice_cfg.get("ref_text", ""),
        "chunk_size": chunk_size,
        "max_new_tokens": max_new_tokens,
        "temperature": _coerce_float(
            _voice_value(req.temperature, voice_cfg, "temperature", 0.9),
            0.9,
            0.0,
            "temperature",
        ),
        "top_k": _coerce_int(_voice_value(req.top_k, voice_cfg, "top_k", 50), 50, 0, "top_k"),
        "top_p": _coerce_float(_voice_value(req.top_p, voice_cfg, "top_p", 1.0), 1.0, 0.0, "top_p"),
        "repetition_penalty": _coerce_float(
            _voice_value(req.repetition_penalty, voice_cfg, "repetition_penalty", 1.05),
            1.05,
            0.0,
            "repetition_penalty",
        ),
        "xvec_only": _voice_value(req.xvec_only, voice_cfg, "xvec_only", settings.xvec_only),
        "non_streaming_mode": _voice_value(
            req.non_streaming_mode,
            voice_cfg,
            "non_streaming_mode",
            False,
        ),
        "instruct": _voice_value(req.instruct, voice_cfg, "instruct", None),
    }


def _queue_item(job: GenerationJob, item: Any) -> bool:
    with job._lock:
        if job._done_sent:
            return False
        job.queue.put(item)
        return True


def _queue_done(job: GenerationJob) -> None:
    with job._lock:
        if job._done_sent:
            return
        job._done_sent = True
        job.queue.put(_DONE)


def _cancel_job(job: GenerationJob, reason: str) -> None:
    first_cancel = not job.cancel_event.is_set()
    job.cancel_reason = reason
    job.cancel_event.set()

    if first_cancel:
        logger.info("request=%d canceled | reason=%s", job.id, reason)

    # Wake the HTTP streamer immediately. The model thread will release the GPU
    # lock after the current chunk boundary because the generator is pull-based.
    _queue_done(job)


def _start_job(text: str, voice_name: str, fmt: str, client: str) -> GenerationJob:
    global _active_job, _request_counter

    with _state_lock:
        _request_counter += 1
        job = GenerationJob(
            id=_request_counter,
            text=text,
            voice_name=voice_name,
            fmt=fmt,
            client=client,
            sample_rate=SAMPLE_RATE,
        )

        previous = _active_job
        if settings.latest_only:
            _active_job = job

    if settings.latest_only and previous is not None and previous is not job:
        _cancel_job(previous, f"superseded by request {job.id}")

    return job


def _finish_job(job: GenerationJob, status: str) -> None:
    global _active_job

    with _state_lock:
        if _active_job is job:
            _active_job = None

    with job._lock:
        if job._finish_logged:
            return
        job._finish_logged = True

    elapsed_ms = (time.perf_counter() - job.created_at) * 1000
    logger.info(
        "request=%d finished | status=%s chunks=%d audio_s=%.2f elapsed_ms=%.0f reason=%r",
        job.id,
        status,
        job.chunks,
        job.audio_seconds,
        elapsed_ms,
        job.cancel_reason,
    )


def _request_is_current(job: GenerationJob) -> bool:
    if not settings.latest_only:
        return True

    with _state_lock:
        return _active_job is job


def _producer(job: GenerationJob, req: SpeechRequest, voice_cfg: dict[str, Any]) -> None:
    status = "complete"
    lock_wait_start = time.perf_counter()

    try:
        kwargs = _generation_kwargs(req, voice_cfg)

        logger.info(
            (
                "request=%d waiting_for_model_lock | voice=%r fmt=%s mode=%s "
                "chunk_size=%s max_new_tokens=%s ref_audio=%r ref_text_chars=%d"
            ),
            job.id,
            job.voice_name,
            job.fmt,
            "xvec_only" if kwargs["xvec_only"] else "icl",
            kwargs["chunk_size"],
            kwargs["max_new_tokens"],
            kwargs["ref_audio"],
            len(kwargs["ref_text"] or ""),
        )

        with _model_lock:
            lock_wait_ms = (time.perf_counter() - lock_wait_start) * 1000
            if job.cancelled() or not _request_is_current(job):
                status = "canceled-before-model-start"
                logger.info("request=%d skipped_model_start | wait_ms=%.0f", job.id, lock_wait_ms)
                return

            job.model_started_at = time.perf_counter()
            logger.info("request=%d model_start | wait_ms=%.0f", job.id, lock_wait_ms)

            generator = tts_model.generate_voice_clone_streaming(
                text=req.input,
                language=kwargs["language"],
                ref_audio=kwargs["ref_audio"],
                ref_text=kwargs["ref_text"],
                chunk_size=kwargs["chunk_size"],
                max_new_tokens=kwargs["max_new_tokens"],
                temperature=kwargs["temperature"],
                top_k=kwargs["top_k"],
                top_p=kwargs["top_p"],
                repetition_penalty=kwargs["repetition_penalty"],
                xvec_only=kwargs["xvec_only"],
                non_streaming_mode=kwargs["non_streaming_mode"],
                instruct=kwargs["instruct"],
            )

            for audio_chunk, sr, timing in generator:
                if job.cancelled() or not _request_is_current(job):
                    status = "canceled-at-chunk-boundary"
                    break

                raw = _to_pcm16(audio_chunk)
                duration = len(audio_chunk) / float(sr) if sr else 0.0

                with job._lock:
                    job.chunks += 1
                    job.audio_seconds += duration
                    job.sample_rate = sr

                if not _queue_item(job, (raw, sr, timing)):
                    status = "stream-closed"
                    break

                logger.info(
                    (
                        "request=%d audio_chunk=%d | bytes=%d chunk_audio_s=%.2f "
                        "total_audio_s=%.2f prefill_ms=%.0f decode_ms=%.0f"
                    ),
                    job.id,
                    job.chunks,
                    len(raw),
                    duration,
                    job.audio_seconds,
                    timing.get("prefill_ms", 0),
                    timing.get("decode_ms", 0),
                )

            if job.cancelled() and status == "complete":
                status = "canceled"

    except Exception as exc:
        status = "error"
        logger.exception("request=%d generation_error", job.id)
        _queue_item(job, exc)
    finally:
        _queue_done(job)
        _finish_job(job, status)


def _start_producer_thread(job: GenerationJob, req: SpeechRequest, voice_cfg: dict[str, Any]) -> threading.Thread:
    thread = threading.Thread(
        target=_producer,
        args=(job, req, voice_cfg),
        daemon=True,
        name=f"renpy-tts-{job.id}",
    )
    thread.start()
    return thread


async def _queue_get_with_disconnect(job: GenerationJob, request: Request) -> Any:
    loop = asyncio.get_event_loop()

    while True:
        if await request.is_disconnected():
            _cancel_job(job, "client disconnected")
            return _DONE

        try:
            return await loop.run_in_executor(None, job.queue.get, True, 0.25)
        except queue.Empty:
            continue


async def _collect_raw_pcm(job: GenerationJob, request: Request) -> tuple[bytes, int]:
    chunks: list[bytes] = []
    sample_rate = SAMPLE_RATE

    while True:
        item = await _queue_get_with_disconnect(job, request)
        if item is _DONE:
            break
        if isinstance(item, Exception):
            raise item

        raw, sr, _timing = item
        chunks.append(raw)
        sample_rate = sr

    if job.cancelled():
        raise HTTPException(status_code=409, detail=f"Request {job.id} was canceled: {job.cancel_reason}")

    return b"".join(chunks), sample_rate


@app.get("/health")
async def health() -> dict[str, Any]:
    with _state_lock:
        active = _active_job

    return {
        "status": "ok",
        "model_loaded": tts_model is not None,
        "sample_rate": SAMPLE_RATE,
        "latest_only": settings.latest_only,
        "active_request_id": active.id if active else None,
    }


@app.get("/voices")
async def list_voices() -> dict[str, Any]:
    return {
        "default_voice": default_voice,
        "voices": sorted(voices.keys()),
    }


@app.post("/cancel")
@app.post("/v1/audio/cancel")
async def cancel_active() -> dict[str, Any]:
    with _state_lock:
        active = _active_job

    if active is None:
        return {"canceled": False, "active_request_id": None}

    _cancel_job(active, "manual cancel endpoint")
    return {"canceled": True, "active_request_id": active.id}


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest, request: Request):
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = req.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="'input' text is empty")

    fmt = req.response_format.lower().strip()
    content_types = {
        "wav": "audio/wav",
        "pcm": "audio/pcm",
        "mp3": "audio/mpeg",
    }
    if fmt not in content_types:
        raise HTTPException(status_code=400, detail="response_format must be one of: wav, pcm, mp3")

    resolved_voice, voice_cfg = resolve_voice(req.voice)
    job = _start_job(text=text, voice_name=resolved_voice, fmt=fmt, client=_client_host(request))

    logger.info(
        (
            "request=%d accepted | client=%s model=%r requested_voice=%r "
            "resolved_voice=%r fmt=%s chars=%d input=%r"
        ),
        job.id,
        job.client,
        req.model,
        req.voice,
        resolved_voice,
        fmt,
        len(text),
        _one_line(text),
    )

    if req.speed != 1.0:
        logger.warning("request=%d speed=%s is accepted for compatibility but not applied", job.id, req.speed)
    if req.stream is not None:
        logger.debug("request=%d stream=%r accepted for compatibility; response streams by format", job.id, req.stream)

    _start_producer_thread(job, req, voice_cfg)

    if fmt == "mp3":
        raw_pcm, sample_rate = await _collect_raw_pcm(job, request)
        return Response(
            content=_to_mp3_bytes(raw_pcm, sample_rate),
            media_type=content_types[fmt],
            headers={"X-TTS-Request-Id": str(job.id), "Cache-Control": "no-store"},
        )

    async def audio_stream():
        status = "stream-complete"
        try:
            if fmt == "wav":
                yield _wav_header(SAMPLE_RATE)

            while True:
                item = await _queue_get_with_disconnect(job, request)
                if item is _DONE:
                    break
                if isinstance(item, Exception):
                    raise item

                raw, _sr, _timing = item
                yield raw

        except asyncio.CancelledError:
            status = "stream-cancelled"
            _cancel_job(job, "stream task canceled")
            raise
        except Exception:
            status = "stream-error"
            _cancel_job(job, "stream error")
            logger.exception("request=%d stream_error", job.id)
            raise
        finally:
            if status == "stream-complete" and job.cancelled():
                status = "stream-canceled"
            _finish_job(job, status)

    return StreamingResponse(
        audio_stream(),
        media_type=content_types[fmt],
        headers={"X-TTS-Request-Id": str(job.id), "Cache-Control": "no-store"},
    )


def _load_voices(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], str]:
    if args.voices:
        with open(args.voices, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)

        if not isinstance(loaded, dict) or not loaded:
            raise SystemExit("--voices must be a non-empty JSON object")

        normalized: dict[str, dict[str, Any]] = {}
        for name, cfg in loaded.items():
            if not isinstance(cfg, dict):
                raise SystemExit(f"voice {name!r} must be a JSON object")
            if "ref_audio" not in cfg:
                raise SystemExit(f"voice {name!r} is missing ref_audio")
            normalized[str(name)] = cfg

        return normalized, next(iter(normalized))

    if not args.ref_audio:
        raise SystemExit("ERROR: provide --ref-audio <file> or --voices <config.json>")

    return {
        "default": {
            "ref_audio": args.ref_audio,
            "ref_text": args.ref_text,
            "language": args.language,
            "chunk_size": args.chunk_size,
            "max_new_tokens": args.max_new_tokens,
        }
    }, "default"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ren'Py-focused OpenAI-compatible TTS server for faster-qwen3-tts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--voices",
        default=os.environ.get("QWEN_TTS_VOICES"),
        metavar="FILE",
        help="JSON file mapping voice names to {ref_audio, ref_text, language}",
    )
    parser.add_argument(
        "--ref-audio",
        default=os.environ.get("QWEN_TTS_REF_AUDIO"),
        metavar="FILE",
        help="Reference audio file when --voices is not used",
    )
    parser.add_argument(
        "--ref-text",
        default=os.environ.get("QWEN_TTS_REF_TEXT", ""),
        help="Transcript of --ref-audio",
    )
    parser.add_argument(
        "--language",
        default=os.environ.get("QWEN_TTS_LANGUAGE", "Auto"),
        help="Target language when --voices is not used",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16", help="Torch dtype")
    parser.add_argument("--chunk-size", type=int, default=8, help="Default stream chunk size")
    parser.add_argument("--max-new-tokens", type=int, default=360, help="Default generation cap")
    parser.add_argument(
        "--icl-mode",
        action="store_true",
        help="Use full reference-audio ICL cloning by default instead of x-vector-only mode",
    )
    parser.add_argument("--no-latest-only", action="store_true", help="Do not cancel older requests automatically")
    parser.add_argument("--log-level", default=os.environ.get("QWEN_TTS_LOG_LEVEL", "INFO"))
    parser.add_argument("--log-file", default=os.environ.get("QWEN_TTS_LOG_FILE"))
    parser.add_argument("--access-log", action="store_true", help="Enable uvicorn access logs")
    return parser.parse_args()


def main() -> None:
    global SAMPLE_RATE, default_voice, settings, tts_model, voices

    args = _parse_args()
    _configure_logging(args.log_level, args.log_file)

    settings = ServerSettings(
        chunk_size=max(1, args.chunk_size),
        max_new_tokens=max(2, args.max_new_tokens),
        xvec_only=not args.icl_mode,
        latest_only=not args.no_latest_only,
    )

    voices, default_voice = _load_voices(args)
    logger.info("loaded_voices count=%d default=%r names=%s", len(voices), default_voice, sorted(voices.keys()))

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    from faster_qwen3_tts import FasterQwen3TTS

    logger.info("loading_model model=%s device=%s dtype=%s", args.model, args.device, args.dtype)
    tts_model = FasterQwen3TTS.from_pretrained(
        args.model,
        device=args.device,
        dtype=dtype_map[args.dtype],
    )
    SAMPLE_RATE = tts_model.sample_rate
    logger.info(
        "model_ready sample_rate=%d latest_only=%r default_mode=%s default_chunk_size=%d max_new_tokens=%d",
        SAMPLE_RATE,
        settings.latest_only,
        "xvec_only" if settings.xvec_only else "icl",
        settings.chunk_size,
        settings.max_new_tokens,
    )
    logger.info("listening url=http://%s:%d/v1/audio/speech", args.host, args.port)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        access_log=args.access_log,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
