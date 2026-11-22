"""Mémoire long-terme d'Amadeus.

Deux niveaux :
- ChromaDB (vectoriel) : chaque échange est embedé, retrieval par similarité.
- Profil glissant (markdown) : résumé LLM de qui est l'utilisateur et de la relation,
  mis à jour tous les MEMORY_SUMMARY_EVERY échanges, injecté dans le prompt système.

Toutes les fonctions sont bloquantes (embedding CPU, SQLite) : depuis du code
async, les appeler via run_in_executor.
"""
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import chromadb

from .config import CONFIG

log = logging.getLogger("amadeus.memory")

_lock = threading.Lock()
_embed_lock = threading.Lock()
_client = None
_collection = None
_embedder = None
_exchange_count = 0
_pending_exchanges: list[dict] = []


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        log.info("Chargement du modèle d'embedding %s…", CONFIG.EMBED_MODEL)
        _embedder = SentenceTransformer(CONFIG.EMBED_MODEL)
    return _embedder


def _get_collection():
    global _client, _collection, _exchange_count
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(CONFIG.DATA_DIR / "chroma"))
        _collection = _client.get_or_create_collection("amadeus_memories")
        _exchange_count = _collection.count()
    return _collection


def _embed(texts: list[str]) -> list[list[float]]:
    # verrou dédié : le tokenizer n'accepte pas deux appels simultanés
    with _embed_lock:
        return _get_embedder().encode(texts, normalize_embeddings=True).tolist()


def warmup() -> None:
    """Charge Chroma + le modèle d'embedding (appelé au démarrage, en arrière-plan)."""
    with _lock:
        _get_collection()
        _embed(["warmup"])
    log.info("Mémoire prête (%d échanges)", _exchange_count)


# ---------------------------------------------------------------- profil

def _profile_path() -> Path:
    return CONFIG.DATA_DIR / "memory_profile.md"


def get_profile() -> str:
    p = _profile_path()
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return ""


def save_profile(profile: str) -> None:
    _profile_path().write_text(profile.strip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------- stockage

def store_exchange(user_text: str, reply_text: str) -> None:
    """Archive un échange (vectoriel + file d'attente de résumé). Thread-safe."""
    global _exchange_count
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record = {
        "user": user_text.strip()[:1000],
        "kurisu": reply_text.strip()[:1000],
        "ts": ts,
    }
    try:
        # calcul hors du verrou de la base : archiver ne doit pas retarder la phrase suivante
        emb = _embed([f"U: {user_text}\nK: {reply_text}"])[0]
        with _lock:
            col = _get_collection()
            col.add(
                ids=[f"ex-{uuid.uuid4().hex}"],
                embeddings=[emb],
                documents=[json.dumps(record, ensure_ascii=False)],
                metadatas=[{"ts": ts}],
            )
            _exchange_count += 1
            _pending_exchanges.append(record)
            need_summary = len(_pending_exchanges) >= CONFIG.MEMORY_SUMMARY_EVERY
            pending = list(_pending_exchanges)
            if need_summary:
                _pending_exchanges.clear()
        if need_summary:
            _update_profile(pending)
    except Exception:
        log.exception("Échec du stockage mémoire (non bloquant)")


def _update_profile(exchanges: list[dict]) -> None:
    """Fusionne les nouveaux échanges dans le profil via LLM (appel bloquant)."""
    from .llm import complete_sync
    from .persona import SUMMARIZE_PROMPT

    profile = get_profile() or "(empty — first memories)"
    lines = "\n".join(
        f"- [{e['ts']}] User: {e['user']}\n  Kurisu: {e['kurisu']}" for e in exchanges
    )
    try:
        new_profile = complete_sync(
            [{"role": "user",
              "content": SUMMARIZE_PROMPT.format(profile=profile, exchanges=lines)}],
            max_tokens=900,
        )
        if new_profile:
            save_profile(new_profile)
            log.info("Profil mémoire mis à jour (%d échanges fusionnés)", len(exchanges))
    except Exception:
        log.exception("Échec du résumé mémoire (le profil reste inchangé)")
        # Remet les échanges en file pour le prochain essai
        with _lock:
            _pending_exchanges[:0] = exchanges


# ---------------------------------------------------------------- retrieval

def retrieve_relevant(query: str, k: int | None = None,
                      exclude_users: set[str] | None = None) -> list[dict]:
    """Récupère les k échanges les plus proches sémantiquement.

    `exclude_users` : messages déjà présents dans l'historique envoyé au LLM. Chaque échange
    est archivé tout de suite : sans ce filtre, on paierait deux fois le même texte."""
    k = k or CONFIG.MEMORY_RETRIEVE_K
    exclude = {u.strip()[:1000] for u in (exclude_users or ())}
    try:
        emb = _embed([query])[0]
        with _lock:
            col = _get_collection()
            n = col.count()
            if n == 0:
                return []
            res = col.query(query_embeddings=[emb], n_results=min(k + len(exclude), n))
        out = []
        for doc in res["documents"][0]:
            try:
                hit = json.loads(doc)
            except json.JSONDecodeError:
                continue
            if hit.get("user", "").strip() not in exclude:
                out.append(hit)
            if len(out) == k:
                break
        return out
    except Exception:
        log.exception("Échec du retrieval mémoire")
        return []


def profile_block(profile: str) -> str:
    """Profil : stable d'un tour à l'autre, il reste dans le préfixe mis en cache par l'API."""
    if not profile:
        return ""
    return ("# Amadeus memory subsystem (context from long-term memory — use naturally)\n\n"
            "## Memory profile (who the user is, your shared history):\n" + profile)


def retrieved_block(hits: list[dict]) -> str:
    """Souvenirs du tour : ils changent à chaque message, donc placés après l'historique."""
    if not hits:
        return ""
    lines = [f"- [{h.get('ts', '?')}] User: {h.get('user', '')} | You: {h.get('kurisu', '')}"
             for h in hits]
    return "## Most relevant past exchanges:\n" + "\n".join(lines)


def wipe() -> None:
    """Réinitialise toute la mémoire (profil + vectoriel)."""
    global _collection, _exchange_count
    with _lock:
        _get_collection()
        _client.delete_collection("amadeus_memories")
        _collection = None
        _exchange_count = 0
        _pending_exchanges.clear()
        p = _profile_path()
        if p.exists():
            p.unlink()


def stats() -> dict:
    """Profil + nombre d'échanges (force l'init lazy de Chroma)."""
    try:
        with _lock:
            _get_collection()
    except Exception:
        log.exception("Chroma indisponible")
    return {"profile": get_profile(), "exchanges": _exchange_count}
