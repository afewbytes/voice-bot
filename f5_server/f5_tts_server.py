#!/usr/bin/env python3
# F5-TTS gRPC server ― works with ≤ 0.6.x and ≥ 1.1.x
"""
$ python3 f5_tts_server.py \
      --checkpoint /app/checkpoints \
      --socket     /app/sockets/f5-tts.sock \
      --vocab      /app/checkpoints/vocab.json
"""
from __future__ import annotations

# ────────────────────────── std-lib ───────────────────────────────────────
import argparse
import asyncio
import contextlib
import inspect
import os
import time
import wave
from typing import Iterable

# ────────────────────────── third-party ───────────────────────────────────
import grpc
import numpy as np
import torch
import torchaudio
from importlib.resources import files

# ── prefer faster matmul on Ampere / Ada GPUs ─────────────────────────────
torch.set_float32_matmul_precision("high")

# ─── F5-TTS import chain (works across all released API layouts) ──────────
try:                             # ≤ 0.6.x
    from f5_tts import TTS as _TTS
except ImportError:
    try:                         # dev layout
        from f5_tts.infer.api import TTS as _TTS
    except ImportError:          # ≥ 1.0.0
        from f5_tts.api import F5TTS as _TTS

TTS = _TTS

# ─── gRPC stubs ───────────────────────────────────────────────────────────
import voice_pb2 as pb
import voice_pb2_grpc as pbg

# ─────────────── helpers ──────────────────────────────────────────────────
def resample(wav: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Resample NumPy waveform iff sample-rate differs."""
    return wav if from_sr == to_sr else torchaudio.functional.resample(
        torch.from_numpy(wav), from_sr, to_sr).numpy()


def _model_sr(engine) -> int:
    """Best-effort guess of the model’s native sampling-rate."""
    for root in (getattr(engine, n, None) for n in ("hps", "cfg")):
        if root and getattr(root, "data", None):
            return root.data.sampling_rate
    if hasattr(engine, "sample_rate"):
        return engine.sample_rate
    voc = getattr(engine, "vocoder", None)
    if voc and hasattr(voc, "sample_rate"):
        return voc.sample_rate
    return 24_000                                              # sensible fallback


_REF_WAV = str(
    files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav")
)
_REF_TXT = "some call me nature, others call me mother nature."


def _synth(engine: TTS, text: str, *, speed: float):
    """Hide the signature differences between the old and the new API."""
    if hasattr(engine, "tts"):                     # old API (≤ 0.6)
        return engine.tts(text, speed=speed, use_cache=False)

    sig = inspect.signature(engine.infer).parameters
    args = []
    if "ref_file" in sig:
        args.append(_REF_WAV)
    if "ref_text" in sig:
        args.append(_REF_TXT)
    if "gen_text" in sig:
        args.append(text)
    kwargs = {"speed": speed} if "speed" in sig else {}
    out = engine.infer(*args, **kwargs)            # new API (≥ 1.0)
    return out[0] if isinstance(out, tuple) else out


def startup_test(
    engine: TTS, out_sr: int, *, dump_path: str | None = None
) -> None:
    """Generate a quick ‘hello’ utterance to warm the model and (optionally) dump it."""
    print("Running startup test … “hello”")
    t0 = time.perf_counter()
    wav = _synth(engine, "hello", speed=1.0)
    dt = time.perf_counter() - t0
    wav = resample(wav, _model_sr(engine), out_sr)
    print(
        f"Generated {len(wav)} samples in {dt:.2f}s "
        f"(RTF {dt / (len(wav) / out_sr):.3f})"
    )

    # ── NEW: persist a mono 16-bit WAV next to the UNIX socket ────────────
    if dump_path and os.getenv("TTS_DUMP_WAV", "1") != "0":
        pcm = (
            np.clip(wav * 32_767, -32_768, 32_767)
            .astype(np.int16)
            .tobytes()
        )
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        with contextlib.closing(wave.open(dump_path, "wb")) as wf:
            wf.setnchannels(1)      # mono
            wf.setsampwidth(2)      # int16
            wf.setframerate(out_sr)
            wf.writeframes(pcm)
        print(f"✎  Wrote warm-up audio → {dump_path}")


# ──────────────── gRPC service implementation ────────────────────────────
class TextToSpeechService(pbg.TextToSpeechServicer):
    def __init__(self, engine: TTS, out_sr: int, page: int = 4096):
        self.engine = engine
        self.out_sr = out_sr
        self.page = page
        self.model_sr = _model_sr(engine)

    # ------------------ internal sync generator --------------------------
    def _synth_iter(
        self, text: str, rate: float
    ) -> Iterable[pb.AudioResponse]:
        if not text.strip():
            return
        wav = _synth(self.engine, text, speed=rate or 1.0)
        wav = resample(wav, self.model_sr, self.out_sr)

        # force mono so byte-count matches channel-count
        if wav.ndim == 2:
            wav = wav.mean(axis=0) if wav.shape[0] == 2 else wav.squeeze()

        # scale to int16 (32767 so +1.0 never wraps to −32768)
        pcm = (
            np.clip(wav * 32_767, -32_768, 32_767)
            .astype(np.int16)
            .tobytes()
        )

        for off in range(0, len(pcm), self.page):
            yield pb.AudioResponse(
                audio_chunk=pcm[off : off + self.page],
                sample_rate=self.out_sr,
                is_end=off + self.page >= len(pcm),
            )

    # ------------------- async façade ------------------------------------
    async def _asend(self, req: pb.TextRequest):
        loop = asyncio.get_running_loop()
        for r in await loop.run_in_executor(
            None,
            lambda: list(self._synth_iter(req.text, req.speaking_rate)),
        ):
            yield r

    async def SynthesizeText(self, request, context):
        async for r in self._asend(request):
            yield r

    async def SynthesizeStreamingText(self, req_iter, context):
        async for req in req_iter:
            async for r in self._asend(req):
                yield r


# ──────────────── CLI / bootstrap helpers ────────────────────────────────
def args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--socket", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument(
        "--device", default="cuda", choices=["cpu", "cuda", "mps"]
    )
    p.add_argument("--out_sr", type=int, default=22_050)
    p.add_argument("--threads", type=int, default=os.cpu_count())
    return p


def _load_engine(a) -> TTS:
    """Automatically pick the right F5-TTS constructor signature."""
    if "ckpt_file" in inspect.signature(TTS).parameters:
        ckpt = (
            next(
                (
                    os.path.join(a.checkpoint, f)
                    for f in os.listdir(a.checkpoint)
                    if f.endswith(".safetensors")
                ),
                a.checkpoint,
            )
            if os.path.isdir(a.checkpoint)
            else a.checkpoint
        )
        print(f"Using new F5-TTS API (ckpt_file={ckpt})")
        return TTS(ckpt_file=ckpt, vocab_file=a.vocab, device=a.device)

    print(f"Using legacy F5-TTS API (checkpoint dir={a.checkpoint})")
    return TTS(
        a.checkpoint,
        vocab_file=a.vocab,
        device=a.device,
        dtype="fp32" if a.device == "cpu" else "float16",
    )


async def serve(a):
    # be nice to OpenBLAS / MKL in multi-container setups
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(
        a.threads
    )

    print(f"Loading F5-TTS on {a.device} …")
    engine = _load_engine(a)

    # bring the UNIX socket up *first* so health-checks can succeed
    if os.path.exists(a.socket):
        os.unlink(a.socket)

    server = grpc.aio.server()
    pbg.add_TextToSpeechServicer_to_server(
        TextToSpeechService(engine, a.out_sr), server
    )
    server.add_insecure_port(f"unix://{a.socket}")
    await server.start()
    os.chmod(a.socket, 0o777)  # allow other services to connect
    open(os.path.join(os.path.dirname(a.socket), "ready"), "w").close()

    print("F5-TTS gRPC server ready – warm-up running in background.")
    test_wav = os.path.join(os.path.dirname(a.socket), "startup_test.wav")
    asyncio.create_task(
        asyncio.to_thread(startup_test, engine, a.out_sr, dump_path=test_wav)
    )

    await server.wait_for_termination()


def main() -> None:
    try:
        asyncio.run(serve(args_parser().parse_args()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()