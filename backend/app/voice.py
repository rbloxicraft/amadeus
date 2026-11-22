"""Comment l'utilisateur a parlé : débit, hauteur, volume, pauses, hésitations.

Whisper ne garde que les mots. Ce module analyse le même audio, sur CPU et sans modèle,
puis compare chaque phrase à la voix HABITUELLE de l'utilisateur : « plus bas que
d'habitude » a un sens, « 180 Hz » n'en a aucun. Le résultat est un court indice en
anglais, ajouté au contexte du LLM. Il ne dépend pas de la langue parlée.
"""
import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CONFIG

log = logging.getLogger("amadeus.voice")

SR = 16000
_WIN = 640                 # 40 ms
_HOP = 160                 # 10 ms
_F0_MIN, _F0_MAX = 70.0, 420.0
_MIN_ANALYSIS_S = 0.5
_BASELINE_MIN_SAMPLES = 5  # avant ça, aucune comparaison : la voix n'est pas encore connue
_BASELINE_ALPHA = 0.08
_NOTABLE_Z = 1.0
_STRONG_Z = 1.8
_LONG_PAUSE_S = 0.6
_MAX_PITCH_OFFSET_ST = 9.0
_RISING_END_ST = 2.5

_FILLERS = re.compile(
    r"(?<![\w])(euh+|heu+|bah|ben|hum+|hmm+|um+|uh+|erm+|えっと|あの|うーん)(?![\w])", re.I)
_TRAILING = re.compile(r"(\.\.\.|…)\s*$")


@dataclass
class Features:
    duration_s: float
    speech_s: float
    loudness_db: float
    pitch_hz: float | None
    pitch_range_st: float | None
    pitch_end_st: float | None
    rate_syll_s: float
    longest_pause_s: float
    pause_ratio: float


def _frames(x: np.ndarray) -> np.ndarray:
    n = 1 + (len(x) - _WIN) // _HOP
    idx = np.arange(_WIN)[None, :] + _HOP * np.arange(n)[:, None]
    return x[idx]


def _speech_mask(db: np.ndarray) -> np.ndarray:
    """Trames parlées : au-dessus du bruit de fond de CET enregistrement."""
    floor, peak = np.percentile(db, 10), np.percentile(db, 95)
    threshold = max(-55.0, floor + 0.35 * max(peak - floor, 0.0))
    mask = db > threshold
    # referme les micro-coupures (< 60 ms) entre deux syllabes
    gaps = np.flatnonzero(np.diff(mask.astype(np.int8)))
    for a, b in zip(gaps[:-1], gaps[1:]):
        if not mask[a + 1] and b - a <= 6:
            mask[a + 1:b + 1] = True
    return mask


def _pitch(frames: np.ndarray) -> np.ndarray:
    """F0 par autocorrélation normalisée ; NaN sur les trames non voisées."""
    w = frames * np.hanning(_WIN)
    spec = np.fft.rfft(w, n=2 * _WIN, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :_WIN]
    energy = ac[:, 0]
    lo, hi = int(SR / _F0_MAX), min(int(SR / _F0_MIN), _WIN - 2)
    out = np.full(len(frames), np.nan)
    ok = energy > 1e-7
    if not ok.any():
        return out
    norm = ac[ok, lo:hi] / energy[ok, None]
    lag = np.argmax(norm, axis=1)
    peak = norm[np.arange(len(lag)), lag]
    values = np.full(len(lag), np.nan)
    voiced = peak >= 0.45
    values[voiced] = SR / (lag[voiced] + lo)
    out[ok] = values
    return out


def _longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def analyze(audio: np.ndarray) -> Features | None:
    """Mesures acoustiques d'une phrase (float32 mono 16 kHz), ou None si trop courte."""
    if len(audio) < SR * _MIN_ANALYSIS_S:
        return None
    x = np.asarray(audio, dtype=np.float32)
    x = x - float(x.mean())
    frames = _frames(x)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    mask = _speech_mask(db)
    if mask.sum() < 20:
        return None

    first, last = np.flatnonzero(mask)[[0, -1]]
    inner = mask[first:last + 1]
    speech_s = float(mask.sum()) * _HOP / SR

    f0 = _pitch(frames)
    f0[~mask] = np.nan
    voiced = f0[np.isfinite(f0)]
    pitch_hz = pitch_range = pitch_end = None
    if len(voiced) >= 15:
        pitch_hz = float(np.median(voiced))
        st = 12 * np.log2(voiced / pitch_hz)
        # l'autocorrélation se trompe parfois d'une octave : ces trames ne comptent pas
        st = st[np.abs(st) <= _MAX_PITCH_OFFSET_ST]
        pitch_range = float(np.percentile(st, 90) - np.percentile(st, 10))
        tail = st[-min(len(st), 35):]
        if len(tail) >= 12:
            # évolution de la hauteur sur la fin : monte (question) ou retombe
            third = max(len(tail) // 3, 3)
            pitch_end = float(np.median(tail[-third:]) - np.median(tail[:third]))

    # noyaux syllabiques : pics d'énergie voisée espacés d'au moins 100 ms
    env = np.convolve(rms, np.ones(5) / 5, mode="same")
    thr = float(np.percentile(env[mask], 35))
    peaks, last_peak = 0, -100
    for i in range(1, len(env) - 1):
        if mask[i] and env[i] > thr and env[i] >= env[i - 1] and env[i] > env[i + 1] and i - last_peak >= 10:
            peaks += 1
            last_peak = i

    return Features(
        duration_s=round(len(x) / SR, 3),
        speech_s=round(speech_s, 3),
        loudness_db=round(float(db[mask].mean()), 2),
        pitch_hz=None if pitch_hz is None else round(pitch_hz, 1),
        pitch_range_st=None if pitch_range is None else round(pitch_range, 2),
        pitch_end_st=None if pitch_end is None else round(pitch_end, 2),
        rate_syll_s=round(peaks / max(speech_s, 0.2), 2),
        longest_pause_s=round(_longest_run(~inner) * _HOP / SR, 3),
        pause_ratio=round(1 - float(inner.mean()), 3),
    )


class Baseline:
    """Moyenne et variance glissantes de la voix de l'utilisateur (data/voice_baseline.json)."""

    KEYS = ("loudness_db", "pitch_hz", "pitch_range_st", "rate_syll_s")
    MIN_STD = {"loudness_db": 2.0, "pitch_hz": 10.0, "pitch_range_st": 0.8, "rate_syll_s": 0.5}

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.stats: dict[str, dict[str, float]] = {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.stats = {k: data[k] for k in self.KEYS if k in data}
        except (OSError, ValueError, TypeError):
            pass

    def z(self, f: Features) -> dict[str, float]:
        out: dict[str, float] = {}
        with self._lock:
            for k in self.KEYS:
                value, s = getattr(f, k), self.stats.get(k)
                if value is None or not s or s["n"] < _BASELINE_MIN_SAMPLES:
                    continue
                out[k] = (value - s["mean"]) / max(float(np.sqrt(s["var"])), self.MIN_STD[k])
        return out

    def update(self, f: Features) -> None:
        if f.speech_s < 0.8:
            return
        with self._lock:
            for k in self.KEYS:
                value = getattr(f, k)
                if value is None:
                    continue
                s = self.stats.get(k)
                if not s:
                    self.stats[k] = {"mean": float(value), "var": self.MIN_STD[k] ** 2, "n": 1}
                    continue
                # moyenne simple au début, puis glissante : la voix du jour compte plus
                alpha = max(_BASELINE_ALPHA, 1 / (s["n"] + 1))
                delta = value - s["mean"]
                s["mean"] += alpha * delta
                s["var"] = (1 - alpha) * (s["var"] + alpha * delta * delta)
                s["n"] += 1
            payload = json.dumps(self.stats)
            try:
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(payload, encoding="utf-8")
                tmp.replace(self.path)
            except OSError:
                log.exception("Profil de voix non enregistré")

    def clear(self) -> None:
        with self._lock:
            self.stats = {}
            self.path.unlink(missing_ok=True)


def describe(f: Features | None, z: dict[str, float], text: str = "",
             emotion: dict | None = None) -> str:
    """Indice lisible par le LLM ; chaîne vide s'il n'y a rien de notable."""
    parts: list[str] = []
    if f is not None:
        def compare(key: str, high: str, low: str) -> None:
            v = z.get(key)
            if v is None or abs(v) < _NOTABLE_Z:
                return
            parts.append(("much " if abs(v) >= _STRONG_Z else "") + (high if v > 0 else low) + " than usual")

        compare("loudness_db", "louder", "quieter")
        compare("rate_syll_s", "faster", "slower")
        compare("pitch_hz", "higher-pitched", "lower-pitched")
        range_z = z.get("pitch_range_st")
        if range_z is not None and range_z >= _NOTABLE_Z:
            parts.append("more animated intonation than usual")
        elif range_z is not None and range_z <= -_NOTABLE_Z:
            parts.append("flatter intonation than usual")
        if f.longest_pause_s >= _LONG_PAUSE_S:
            parts.append(f"{f.longest_pause_s:.1f} s pause mid-sentence")
        if f.pitch_end_st is not None and f.pitch_end_st >= _RISING_END_ST:
            parts.append("voice rises at the end")

    fillers = len(_FILLERS.findall(text))
    if fillers:
        parts.append("hesitant" if fillers == 1 else f"hesitant ({fillers} fillers)")
    if text and _TRAILING.search(text):
        parts.append("trails off")
    if emotion and emotion.get("label") not in (None, "neutral"):
        parts.append(f"possibly {emotion['label']} (confidence {emotion['score']:.2f})")
    return "; ".join(parts)


_baseline: Baseline | None = None
_baseline_lock = threading.Lock()


def baseline() -> Baseline:
    global _baseline
    with _baseline_lock:
        if _baseline is None:
            _baseline = Baseline(CONFIG.DATA_DIR / "voice_baseline.json")
        return _baseline


def safe_analyze(audio: np.ndarray) -> Features | None:
    """Comme `analyze`, mais une erreur d'analyse ne doit jamais bloquer la phrase."""
    if not CONFIG.VOICE_ANALYSIS:
        return None
    try:
        return analyze(audio)
    except Exception:
        log.exception("Analyse de la voix impossible (la phrase part sans)")
        return None


def summarize(f: Features | None, text: str, emotion: dict | None, learn: bool) -> str:
    """Indice de la phrase ; `learn` l'ajoute à la voix habituelle de l'utilisateur."""
    if not CONFIG.VOICE_ANALYSIS:
        return ""
    b = baseline()
    hint = describe(f, b.z(f) if f else {}, text, emotion)
    if f is not None and learn and text:
        b.update(f)
    return hint


def clean_hint(value) -> str:
    """Indice reçu du navigateur : borné, et sans balise qui sortirait du bloc de contexte."""
    if not isinstance(value, str):
        return ""
    return re.sub(r"[<>\r\n]+", " ", value).strip()[:300]
