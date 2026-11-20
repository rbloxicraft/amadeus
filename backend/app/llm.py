"""Client LLM — deux « cerveaux » compatibles OpenAI, au choix dans l'interface.

  main   : cerveau principal (LLM_API_KEY / LLM_BASE_URL / LLM_MODEL dans .env)
  gptoss : GPT-OSS 120B chez Groq (GROQ_API_KEY), gratuit et rapide

Le choix est enregistré dans data/settings.json et s'applique dès le message suivant.
Les clients sont créés une seule fois par cerveau et réutilisés : la connexion HTTPS
reste ouverte entre deux messages. Le client async sert la boucle d'uvicorn ; le
client sync sert les threads d'arrière-plan (résumé mémoire).
"""
import json
import logging
import re
import threading
import time

import openai
from openai import AsyncOpenAI, OpenAI

from .config import CONFIG
from .persona import EMOTIONS

log = logging.getLogger("amadeus.llm")

BRAINS = {
    "main": {
        "label": "Principal",
        "detail": CONFIG.MODEL,
        "key": CONFIG.API_KEY, "base_url": CONFIG.BASE_URL, "model": CONFIG.MODEL,
        # la « réflexion » de certains modèles ajoute plusieurs secondes : coupée
        # (paramètre retiré automatiquement si l'API ne le connaît pas)
        "extra": {"extra_body": {"thinking": {"type": "disabled"}}} if CONFIG.DISABLE_THINKING else {},
    },
    "gptoss": {
        "label": "Rapide",
        "detail": "GPT-OSS 120B via Groq (gratuit)",
        "key": CONFIG.GROQ_API_KEY, "base_url": CONFIG.GROQ_BASE_URL, "model": CONFIG.GROQ_MODEL,
        # modèle à raisonnement : effort minimal, sinon il réfléchit avant de parler
        "extra": {"reasoning_effort": "low"},
    },
}
_ALIASES = {"glm": "main"}   # ancien nom du cerveau principal
DEFAULT_BRAIN = _ALIASES.get(CONFIG.BRAIN, CONFIG.BRAIN)
if DEFAULT_BRAIN not in BRAINS:
    DEFAULT_BRAIN = "main"

_SETTINGS = CONFIG.DATA_DIR / "settings.json"
_lock = threading.Lock()
_async_clients: dict[str, AsyncOpenAI] = {}
_sync_clients: dict[str, OpenAI] = {}
_extra_ok: dict[str, bool] = {k: True for k in BRAINS}   # False si l'API refuse le paramètre

_EMOTION_RE = re.compile(r"^\s*\[(" + "|".join(EMOTIONS) + r")\]\s*")


# ---------------------------------------------------------------- choix du cerveau

def _configured(brain: str) -> bool:
    key = BRAINS[brain]["key"]
    return bool(key) and "colle-ta-cle" not in key.lower()


def _load_brain() -> str:
    try:
        b = json.loads(_SETTINGS.read_text(encoding="utf-8")).get("brain")
        b = _ALIASES.get(b, b)
        if b in BRAINS:
            return b
    except (OSError, ValueError):
        pass
    return DEFAULT_BRAIN


_brain = _load_brain()


def current_brain() -> str:
    return _brain


def set_brain(brain: str) -> None:
    global _brain
    brain = _ALIASES.get(brain, brain)
    if brain not in BRAINS:
        raise ValueError(f"cerveau inconnu : {brain}")
    if not _configured(brain):
        raise ValueError(f"clé API manquante pour {BRAINS[brain]['label']}")
    with _lock:
        try:
            data = json.loads(_SETTINGS.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        data["brain"] = brain
        _SETTINGS.write_text(json.dumps(data, indent=2), encoding="utf-8")
        _brain = brain
    log.info("Cerveau : %s (%s)", BRAINS[brain]["label"], BRAINS[brain]["model"])


def brains_info() -> dict:
    return {
        "current": _brain,
        "options": [{"id": k, "label": b["label"], "detail": b["detail"], "model": b["model"],
                     "configured": _configured(k)} for k, b in BRAINS.items()],
    }


def is_configured() -> bool:
    return _configured(_brain)


def current_model() -> str:
    return BRAINS[_brain]["model"]


# ---------------------------------------------------------------- clients

def _require_key(brain: str) -> None:
    if not _configured(brain):
        name = "GROQ_API_KEY" if brain == "gptoss" else "LLM_API_KEY"
        raise RuntimeError(f"{name} manquant dans backend/.env")


def get_client(brain: str | None = None) -> AsyncOpenAI:
    brain = brain or _brain
    _require_key(brain)
    if brain not in _async_clients:
        b = BRAINS[brain]
        # Groq : aucun nouvel essai en cas de quota dépassé (429) → bascule immédiate sur
        # cerveau principal au lieu d'attendre la fin de la minute
        _async_clients[brain] = AsyncOpenAI(api_key=b["key"], base_url=b["base_url"], timeout=60.0,
                                            max_retries=0 if brain == "gptoss" else 1)
    return _async_clients[brain]


def get_sync_client(brain: str | None = None) -> OpenAI:
    brain = brain or _brain
    _require_key(brain)
    if brain not in _sync_clients:
        b = BRAINS[brain]
        _sync_clients[brain] = OpenAI(api_key=b["key"], base_url=b["base_url"],
                                      timeout=90.0, max_retries=1)
    return _sync_clients[brain]


def _extra(brain: str) -> dict:
    return BRAINS[brain]["extra"] if _extra_ok[brain] else {}


def _drop_extra(brain: str, err: Exception) -> bool:
    """Si l'API rejette le paramètre spécial (thinking / reasoning), on le retire pour
    toute la suite. → True si un nouvel essai est utile."""
    msg = str(err).lower()
    if _extra_ok[brain] and BRAINS[brain]["extra"] and ("thinking" in msg or "reasoning" in msg):
        log.warning("%s refuse le paramètre spécial → retiré (%s)", BRAINS[brain]["label"], err)
        _extra_ok[brain] = False
        return True
    return False


# ---------------------------------------------------------------- appels

# L'offre gratuite de Groq est limitée en jetons par minute (8 000 pour GPT-OSS 120B,
# soit 2-3 messages avec la mémoire). Quand elle est épuisée, le message part sur le
# cerveau principal
# et Groq est laissé au repos jusqu'à la fin de la fenêtre indiquée par l'API.
_cooldown_until: dict[str, float] = {}


def _route() -> list[str]:
    """Cerveaux à essayer, dans l'ordre."""
    order = [_brain]
    if _brain != "main" and _configured("main"):
        if time.monotonic() < _cooldown_until.get(_brain, 0):
            order = ["main"]
        else:
            order.append("main")
    return order


def _rate_limited(brain: str, err: openai.RateLimitError) -> None:
    wait = 20.0
    try:
        h = err.response.headers
        wait = float(h.get("retry-after") or h.get("x-ratelimit-reset-tokens", "20").rstrip("s") or 20)
    except Exception:
        pass
    wait = min(max(wait, 5.0), 120.0)
    _cooldown_until[brain] = time.monotonic() + wait
    log.warning("%s : quota par minute atteint → cerveau principal pendant %.0f s", BRAINS[brain]["label"], wait)


async def _create(brain: str, **kwargs):
    client = get_client(brain)
    kwargs["model"] = BRAINS[brain]["model"]
    try:
        return await client.chat.completions.create(**kwargs, **_extra(brain))
    except openai.BadRequestError as e:
        if not _drop_extra(brain, e):
            raise
        return await client.chat.completions.create(**kwargs)


async def _create_routed(**kwargs):
    route = _route()
    for i, brain in enumerate(route):
        try:
            return await _create(brain, **kwargs)
        except openai.RateLimitError as e:
            if i == len(route) - 1:
                raise
            _rate_limited(brain, e)


async def stream_chat(messages: list[dict], max_tokens: int = 600):
    """Stream les deltas de texte du cerveau choisi. Lève une exception en cas d'erreur."""
    stream = await _create_routed(messages=messages, stream=True, temperature=0.8, max_tokens=max_tokens)
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


async def complete(messages: list[dict], max_tokens: int = 800,
                   temperature: float = 0.5) -> str:
    """Appel non-streaming (salutations, traductions de secours)."""
    resp = await _create_routed(messages=messages, temperature=temperature, max_tokens=max_tokens)
    return (resp.choices[0].message.content or "").strip()


def complete_sync(messages: list[dict], max_tokens: int = 800,
                  temperature: float = 0.5) -> str:
    """Version bloquante (résumés mémoire, scripts). Rien ne presse ici : le cerveau
    principal, plus soigneux, s'il est configuré ; sinon le cerveau choisi."""
    brain = "main" if _configured("main") else _brain
    client = get_sync_client(brain)
    kwargs = dict(model=BRAINS[brain]["model"], messages=messages,
                  temperature=temperature, max_tokens=max_tokens)
    try:
        resp = client.chat.completions.create(**kwargs, **_extra(brain))
    except openai.BadRequestError as e:
        if not _drop_extra(brain, e):
            raise
        resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def parse_emotion_tag(text: str) -> tuple[str | None, str]:
    """Extrait la balise d'émotion en tête '[joy]...' → (emotion, reste).

    Retourne (None, text) si aucune balise valide n'ouvre le texte.
    """
    m = _EMOTION_RE.match(text)
    if m:
        return m.group(1), text[m.end():]
    return None, text
