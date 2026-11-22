"""Émotion de la voix par emotion2vec (facultatif, sur CPU).

Désactivé par défaut : il faut installer backend/requirements-emotion.txt (plus de 1 Go)
et mettre VOICE_EMOTION=on. Il n'a pas été validé sur le français : seuls les résultats
au-dessus de VOICE_EMOTION_MIN_CONFIDENCE sont transmis, comme une simple hypothèse.
"""
import asyncio
import logging
import threading

import numpy as np

from .config import CONFIG

log = logging.getLogger("amadeus.emotion")

_MODEL = "emotion2vec/emotion2vec_plus_base"
_LABELS = {"angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"}
_EXTRA_WAIT_S = 0.15   # retard maximal accepté après Whisper

_model = None
_failed = False
_lock = threading.Lock()


def enabled() -> bool:
    return CONFIG.VOICE_EMOTION and not _failed


def _get_model():
    global _model, _failed
    if _model is None and not _failed:
        try:
            from funasr import AutoModel

            _model = AutoModel(model=_MODEL, hub="hf", device="cpu", disable_update=True)
            log.info("emotion2vec chargé sur CPU")
        except Exception:
            _failed = True
            log.exception("emotion2vec indisponible : la voix sera analysée sans lui")
    return _model


def warmup() -> None:
    if enabled():
        with _lock:
            _get_model()


def classify(audio: np.ndarray) -> dict | None:
    """→ {label, score} si le modèle est assez sûr de lui, sinon None."""
    if not enabled() or len(audio) < 8000:
        return None
    if not _lock.acquire(blocking=False):
        return None   # analyse précédente encore en cours : cette phrase part sans
    try:
        model = _get_model()
        if model is None:
            return None
        r = model.generate(audio, granularity="utterance", extract_embedding=False,
                           disable_pbar=True)[0]
        scores = [float(s) for s in r["scores"]]
        best = int(np.argmax(scores))
        label = str(r["labels"][best]).split("/")[-1]
        if label not in _LABELS or scores[best] < CONFIG.VOICE_EMOTION_MIN_CONFIDENCE:
            return None
        return {"label": label, "score": round(scores[best], 2)}
    except Exception:
        log.exception("Analyse d'émotion impossible")
        return None
    finally:
        _lock.release()


async def settle(task: asyncio.Future | None) -> dict | None:
    """Résultat s'il arrive vite ; l'émotion ne doit jamais retarder la réponse."""
    if task is None:
        return None
    done, _ = await asyncio.wait({task}, timeout=_EXTRA_WAIT_S)
    if not done or task.cancelled() or task.exception() is not None:
        return None
    return task.result()
