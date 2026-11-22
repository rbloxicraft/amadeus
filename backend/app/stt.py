"""Reconnaissance vocale via faster-whisper (GPU si possible, sinon CPU)."""
import io
import logging
import os
import sys
import threading
from pathlib import Path

import numpy as np

from .config import CONFIG

log = logging.getLogger("amadeus.stt")

_model = None
# Indice de vocabulaire : sans lui, Whisper entend « Curissu » au lieu de « Kurisu »
_VOCAB = "Kurisu, Amadeus, Christina, Okabe, Mayuri, Daru, Suzuha, Steins;Gate."
_FILLER_PROMPTS = {
    "fr": " Euh… bah, je sais pas… hmm, attends, ah ouais ?",
    "en": " Um… well, I don't know… hmm, wait, oh yeah?",
}
_model_device = None
_last_language: str | None = None
_load_lock = threading.Lock()
# Une inférence à la fois : une passe provisoire ne doit pas ralentir la passe finale
_infer_lock = threading.Lock()


def _add_cuda_dll_dirs() -> None:
    """Rend les DLL cuBLAS/cuDNN utilisables par ctranslate2 (GPU).

    CTranslate2 ne trouve pas les DLL via add_dll_directory seul : il faut les
    PRÉ-CHARGER avec ctypes. Sources : wheels nvidia-* du venv backend, sinon
    torch/lib du venv GPT-SoVITS (torch CUDA déjà installé là-bas).
    """
    import ctypes

    candidates = list(Path(sys.prefix).glob("Lib/site-packages/nvidia/*/bin"))
    torch_lib = (CONFIG.DATA_DIR.parent / "tts-server" / "GPT-SoVITS" / "venv"
                 / "Lib" / "site-packages" / "torch" / "lib")
    if torch_lib.exists():
        candidates.append(torch_lib)
    for base in candidates:
        os.add_dll_directory(str(base))
    for base in candidates:
        for dll in ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll",
                    "cudnn64_9.dll"]:
            p = base / dll
            if p.exists():
                try:
                    ctypes.WinDLL(str(p))
                except OSError:
                    pass


def _load(device: str):
    from faster_whisper import WhisperModel

    if device == "cuda":
        return WhisperModel(CONFIG.STT_MODEL, device="cuda", compute_type="float16")
    return WhisperModel(CONFIG.STT_MODEL, device="cpu", compute_type="int8")


def get_model():
    global _model, _model_device
    with _load_lock:
        if _model is not None:
            return _model
        if CONFIG.STT_DEVICE in ("auto", "cuda"):
            try:
                _add_cuda_dll_dirs()
                _model, _model_device = _load("cuda"), "cuda"
                log.info("Whisper %s chargé sur GPU (cuda/float16)", CONFIG.STT_MODEL)
                return _model
            except Exception as e:
                log.warning("Whisper GPU impossible (%s) → fallback CPU int8", e)
        _model, _model_device = _load("cpu"), "cpu"
        log.info("Whisper %s chargé sur CPU (int8)", CONFIG.STT_MODEL)
        return _model


def _cpu_fallback():
    """Recharge en CPU après une erreur GPU au moment de la transcription."""
    global _model, _model_device
    with _load_lock:
        if _model_device != "cpu":
            log.warning("Erreur GPU en transcription → bascule Whisper sur CPU")
            _model, _model_device = _load("cpu"), "cpu"
        return _model


def _prompt(language: str | None) -> str:
    """Whisper imite le style de son amorce : avec des hésitations, il garde les « euh »."""
    return _VOCAB + _FILLER_PROMPTS.get(language or "", "")


def _transcribe(model, audio, *, vad: bool, partial: bool) -> dict:
    global _last_language
    forced = None if CONFIG.STT_LANGUAGE in ("", "auto") else CONFIG.STT_LANGUAGE
    known = forced or _last_language
    # provisoire : la langue de la dernière phrase évite une détection à chaque passage
    lang = forced or (known if partial else None)
    options = {"temperature": 0.0, "without_timestamps": True} if partial else {}
    with _infer_lock:
        segments, info = model.transcribe(
            audio,
            language=lang,
            vad_filter=vad,
            beam_size=1,
            condition_on_previous_text=False,
            initial_prompt=_prompt(known),
            **options,
        )
        # `segments` est un générateur : l'inférence se fait ici, sous le verrou
        text = "".join(seg.text for seg in segments).strip()
    if text and not partial:
        _last_language = info.language
    return {"text": text, "language": info.language}


def _run(audio, *, vad: bool, partial: bool) -> dict:
    model = get_model()
    try:
        return _transcribe(model, audio, vad=vad, partial=partial)
    except RuntimeError:
        # DLL CUDA manquante & co : le GPU a chargé mais échoue à l'usage
        return _transcribe(_cpu_fallback(), audio, vad=vad, partial=partial)


def warmup() -> None:
    get_model()


def partials_enabled() -> bool:
    """Transcription en direct : sur GPU en auto ; sur CPU elle prendrait du retard."""
    if CONFIG.STT_PARTIALS == "on":
        return True
    return CONFIG.STT_PARTIALS == "auto" and _model_device == "cuda"


def decode(audio_bytes: bytes) -> np.ndarray:
    """wav/webm/mp3 → float32 mono 16 kHz (PyAV, en mémoire)."""
    from faster_whisper.audio import decode_audio

    return decode_audio(io.BytesIO(audio_bytes), sampling_rate=16000)


def transcribe_pcm(audio: np.ndarray, *, partial: bool = False, vad: bool = False) -> dict:
    """→ {text, language}. `vad=False` : l'audio a déjà été découpé par le navigateur."""
    return _run(audio, vad=vad, partial=partial)


def transcribe(audio_bytes: bytes, *, vad: bool = True) -> dict:
    """→ {text, language}. Accepte wav/webm/mp3 (décodage via PyAV)."""
    return _run(decode(audio_bytes), vad=vad, partial=False)
