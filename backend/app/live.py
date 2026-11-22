"""Écoute en direct (WebSocket /api/listen).

Le navigateur envoie le micro pendant que l'utilisateur parle. Le travail qui attendait
autrefois la fin de la phrase se fait donc en parallèle :

    start → PCM… → (partial)… → prepare → prepared → commit → final

- `partial`  : transcription provisoire, affichée en direct, sert aussi à couper Kurisu ;
- `prepare`  : ~200 ms de silence : Whisper, l'analyse de la voix et la mémoire démarrent
               sans attendre d'être sûr que la phrase est finie ;
- `resume`   : l'utilisateur reprend : le résultat préparé est jeté ;
- `commit`   : fin de phrase confirmée : le résultat préparé sert directement.

Audio : PCM 16 bits little-endian, mono, 16 kHz.
"""
import asyncio
import difflib
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable

import numpy as np
from starlette.concurrency import run_in_threadpool

from . import emotion, stt, voice

log = logging.getLogger("amadeus.live")

SR = 16000
MAX_UTTERANCE_S = 60
MAX_FRAME_BYTES = 64_000
PARTIAL_MIN_S = 0.5
PARTIAL_GROWTH_S = 0.4
PARTIAL_WINDOW_S = 15

_CONTINUES = {
    # français
    "et", "mais", "ou", "donc", "car", "parce", "que", "qui", "si", "quand", "comme", "alors",
    "puis", "pour", "avec", "de", "du", "des", "le", "la", "les", "un", "une", "je", "tu",
    "il", "elle", "on", "nous", "vous", "ils", "elles", "mon", "ma", "mes", "ton", "ta", "tes",
    "euh", "heu", "bah", "ben", "genre",
    # anglais
    "and", "but", "or", "so", "because", "that", "which", "if", "when", "as", "to", "of",
    "the", "a", "an", "my", "your", "with", "for", "um", "uh", "like", "i",
}
_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?")


def looks_complete(text: str) -> bool:
    """La phrase semble finie ? Sert à décider combien de silence attendre."""
    t = text.strip()
    if not t or t.endswith((",", ";", ":", "-", "…", "...")):
        return False
    words = _WORD.findall(t.lower())
    if not words:
        return False
    return words[-1].split("'")[-1].split("’")[-1] not in _CONTINUES and t[-1] in ".!?。！？"


class MemoryPrefetch:
    """Souvenirs déjà cherchés à partir d'une transcription provisoire, par session."""

    MAX_AGE_S = 20.0
    MIN_SIMILARITY = 0.85
    MAX_SESSIONS = 32

    def __init__(self) -> None:
        self._items: dict[str, tuple[str, list[dict], float]] = {}

    def put(self, session: str, query: str, hits: list[dict]) -> None:
        self._items.pop(session, None)
        self._items[session] = (query, hits, time.monotonic())
        while len(self._items) > self.MAX_SESSIONS:
            self._items.pop(next(iter(self._items)))

    def take(self, session: str, query: str) -> list[dict] | None:
        item = self._items.pop(session, None)
        if item is None:
            return None
        previous, hits, at = item
        if time.monotonic() - at > self.MAX_AGE_S:
            return None
        ratio = difflib.SequenceMatcher(None, previous.lower(), query.lower()).ratio()
        return hits if ratio >= self.MIN_SIMILARITY else None

    def clear(self) -> None:
        self._items.clear()


class _Utterance:
    def __init__(self, uid: int, barge_in: bool) -> None:
        self.id = uid
        self.barge_in = barge_in
        self.pcm = bytearray()
        self.partial_at = 0
        self.prepared: asyncio.Task | None = None
        self.prepared_at = 0
        self.closed = False
        self.t0 = time.perf_counter()

    @property
    def seconds(self) -> float:
        return len(self.pcm) / (2 * SR)

    def samples(self, last_s: float | None = None) -> np.ndarray:
        data = self.pcm
        if last_s is not None:
            data = data[-int(last_s * SR) * 2:]
        return np.frombuffer(bytes(data), dtype=np.int16).astype(np.float32) / 32768


class LiveListener:
    """État d'une connexion WebSocket : au plus une phrase en cours."""

    def __init__(self, send: Callable[[dict], Awaitable[None]], session: str,
                 retrieve: Callable[[str], list[dict]], prefetch: MemoryPrefetch) -> None:
        self._send = send
        self._session = session
        self._retrieve = retrieve
        self._prefetch = prefetch
        self._utt: _Utterance | None = None
        self._partial_task: asyncio.Task | None = None
        self._background: set[asyncio.Task] = set()

    def _spawn(self, coro) -> asyncio.Task:
        """Tâche de fond gardée en référence (asyncio ne garde qu'une référence faible)."""
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    # ------------------------------------------------------------ messages reçus

    async def on_text(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
            kind, uid = msg["t"], int(msg["id"])
        except (ValueError, KeyError, TypeError):
            return
        if kind == "start":
            await self._cancel_current()
            self._utt = _Utterance(uid, bool(msg.get("bargeIn")))
            return
        u = self._utt
        if u is None or u.id != uid:
            return
        if kind == "prepare":
            self._prepare(u)
        elif kind == "resume":
            self._discard_prepared(u)
        elif kind == "commit":
            self._utt = None
            await self._stop_partials()
            self._spawn(self._finish(u))
        elif kind == "cancel":
            await self._cancel_current()

    def on_audio(self, data: bytes) -> None:
        u = self._utt
        if u is None or len(data) > MAX_FRAME_BYTES or len(data) % 2:
            return
        if len(u.pcm) + len(data) > MAX_UTTERANCE_S * SR * 2:
            return
        u.pcm += data
        if u.prepared is None and self._partial_task is None and self._wants_partial(u):
            self._partial_task = asyncio.create_task(self._partial(u))

    async def close(self) -> None:
        await self._cancel_current()

    # ------------------------------------------------------------ transcriptions

    def _wants_partial(self, u: _Utterance) -> bool:
        if not stt.partials_enabled() or u.seconds < PARTIAL_MIN_S:
            return False
        growth = PARTIAL_GROWTH_S * (2 if u.barge_in else 1)   # GPU partagé avec sa voix
        return len(u.pcm) - u.partial_at >= growth * SR * 2

    async def _partial(self, u: _Utterance) -> None:
        try:
            u.partial_at = len(u.pcm)
            audio = u.samples(PARTIAL_WINDOW_S)
            r = await run_in_threadpool(stt.transcribe_pcm, audio, partial=True)
            if self._utt is u and u.prepared is None and r["text"]:
                await self._send({"t": "partial", "id": u.id, "text": r["text"]})
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Transcription provisoire impossible")
        finally:
            if self._partial_task is asyncio.current_task():
                self._partial_task = None

    def _prepare(self, u: _Utterance) -> None:
        if u.prepared is None and u.pcm:
            u.prepared_at = len(u.pcm)
            u.prepared = asyncio.create_task(self._analyze(u.samples()))
            u.prepared.add_done_callback(lambda t: self._announce(u, t))

    def _announce(self, u: _Utterance, task: asyncio.Task) -> None:
        if task.cancelled() or task.exception() is not None or u.prepared is not task or u.closed:
            return
        r = task.result()
        self._spawn(self._safe_send({
            "t": "prepared", "id": u.id, "text": r["text"], "complete": looks_complete(r["text"]),
        }))

    def _discard_prepared(self, u: _Utterance) -> None:
        if u.prepared is not None:
            u.prepared.cancel()   # le thread Whisper finit seul ; son résultat est ignoré
            u.prepared = None

    async def _analyze(self, audio: np.ndarray) -> dict:
        """Whisper, voix et émotion en parallèle, puis souvenirs pour le prochain tour."""
        t0 = time.perf_counter()
        mood = (asyncio.ensure_future(run_in_threadpool(emotion.classify, audio))
                if emotion.enabled() else None)
        words, features = await asyncio.gather(
            run_in_threadpool(stt.transcribe_pcm, audio),
            run_in_threadpool(voice.safe_analyze, audio),
        )
        mood_result = await emotion.settle(mood)
        ms = round((time.perf_counter() - t0) * 1000)
        text = words["text"]
        if len(text) >= 3:
            self._spawn(self._fetch_memory(text))
        return {**words, "features": features, "emotion": mood_result, "stt_ms": ms}

    async def _fetch_memory(self, text: str) -> None:
        try:
            hits = await run_in_threadpool(self._retrieve, text)
            self._prefetch.put(self._session, text, hits)
        except Exception:
            log.exception("Recherche anticipée des souvenirs impossible")

    async def _finish(self, u: _Utterance) -> None:
        u.closed = True
        t0 = time.perf_counter()
        try:
            task = u.prepared
            reused = task is not None and u.prepared_at > 0
            if reused:
                try:
                    r = await task
                except asyncio.CancelledError:
                    if asyncio.current_task().cancelling():
                        raise
                    r, reused = await self._analyze(u.samples()), False
            else:
                r = await self._analyze(u.samples()) if u.pcm else None
            if r is None:
                await self._safe_send({"t": "final", "id": u.id, "text": "", "language": None, "voice": ""})
                return
            # une phrase d'interruption peut contenir l'écho de Kurisu : elle n'apprend rien
            hint = await run_in_threadpool(voice.summarize, r.get("features"), r["text"],
                                           r.get("emotion"), not u.barge_in)
            await self._safe_send({
                "t": "final", "id": u.id, "text": r["text"], "language": r.get("language"),
                "voice": hint,
                "timing": {"stt_ms": r.get("stt_ms"), "reused": reused,
                           "commit_wait_ms": round((time.perf_counter() - t0) * 1000)},
            })
        except Exception:
            log.exception("Fin de phrase impossible")
            await self._safe_send({"t": "error", "id": u.id})

    # ------------------------------------------------------------ nettoyage

    async def _stop_partials(self) -> None:
        task, self._partial_task = self._partial_task, None
        if task is not None:
            task.cancel()

    async def _cancel_current(self) -> None:
        u, self._utt = self._utt, None
        await self._stop_partials()
        if u is not None:
            u.closed = True
            self._discard_prepared(u)

    async def _safe_send(self, msg: dict) -> None:
        try:
            await self._send(msg)
        except Exception:
            pass   # navigateur parti : le prochain message fermera la boucle
