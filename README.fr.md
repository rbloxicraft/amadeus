# AMADEUS — compagnon IA inspiré de Steins;Gate

[English](README.md) · **Français**

Un compagnon vocal façon **Amadeus** : une IA qui a la personnalité de Kurisu Makise
(sarcastique, tsundere, passionnée de science), avec une **mémoire à long terme**, la
**reconnaissance vocale**, un **avatar 3D** et une **voix japonaise** générée en temps réel
par GPT-SoVITS.

Tu lui parles en français (ou en anglais) : elle te répond à voix haute en japonais, avec
les sous-titres dans ta langue, comme un doublage.

```
Navigateur                              Backend (FastAPI)              Modèles
├─ Avatar VRM 3D / portrait 2D   ←SSE→  ├─ /api/chat : IA en flux     ├─ API d'IA (compatible OpenAI)
├─ Mains libres (Silero VAD)            ├─ /api/tts  : voix en flux   ├─ GPT-SoVITS (GPU, local)
├─ Lecture audio en streaming           ├─ /api/stt  : Whisper        └─ faster-whisper (local)
└─ Sous-titres, mémoire, réglages       └─ ChromaDB + profil mémoire
```

Kurisu commence à parler **environ 3 s** après la fin de ta phrase (2 s avec le cerveau
rapide) : tout est en flux continu, la voix démarre dès la première phrase écrite.

> Projet de fan, sans lien avec MAGES. / Chiyomaru Studio. **Aucun contenu du jeu ou de
> l'animé n'est inclus dans ce dépôt** : pas de voix, pas d'image, pas de modèle 3D.
> L'installateur télécharge le modèle de voix communautaire sur Hugging Face, et c'est
> toi qui ajoutes, si tu le souhaites, un avatar et un logo (voir plus bas).

---

## Ce qu'il te faut

| | Minimum | Conseillé |
|---|---|---|
| Système | Windows 10 ou 11 (64 bits) | |
| Carte graphique | — (voix de secours sans GPU) | **NVIDIA**, 6 Go de mémoire vidéo ou plus |
| Espace disque | 5 Go | 15 Go (avec la vraie voix) |
| Navigateur | Chrome, Edge ou Firefox (Zen marche aussi) | |
| Clé | Une **clé d'API d'IA** (voir plus bas) | |

L'installateur s'occupe de **Python 3.12** et de **Git** s'ils manquent (via `winget`, avec
ta permission). Sans carte NVIDIA, tout fonctionne sauf la vraie voix : Kurisu parle alors
avec une voix de synthèse de secours.

## Installation

1. **Télécharge le projet** : bouton vert **Code → Download ZIP** sur GitHub, puis
   extrais-le, par exemple dans `Documents\amadeus`. Ou, avec Git :
   ```
   git clone https://github.com/rbloxicraft/amadeus.git
   ```
   Évite les dossiers synchronisés (OneDrive) : l'installation pèse plusieurs Go.
2. **Prépare une clé d'API d'IA** (voir [La clé d'API](#la-clé-dapi) juste en dessous).
3. **Double-clique `install.bat`**. Il :
   - vérifie la machine (Python 3.12, Git, carte NVIDIA, espace disque) ;
   - installe le backend (cerveau, oreilles, mémoire) ;
   - installe le moteur de voix GPT-SoVITS, avec PyTorch pour carte graphique ;
   - télécharge la voix de Kurisu depuis Hugging Face et la prépare ;
   - te demande ta clé d'API et l'enregistre.

   Compte **20 à 40 minutes** la première fois, selon ta connexion (environ 5 Go à
   télécharger). Si une étape échoue, corrige le problème et relance `install.bat` : les
   étapes déjà faites sont sautées. Le détail de tout ce qui se passe est dans
   `data\install.log`.
4. **Double-clique `start_amadeus.bat`**. Il démarre la voix et le backend, attend qu'ils
   répondent, puis ouvre http://127.0.0.1:8642 dans ton navigateur. Au tout premier
   lancement, Whisper et la mémoire téléchargent encore leurs modèles (~2 Go).
5. Clique **Se connecter** (le navigateur exige un clic avant de jouer du son), autorise
   le micro, et parle-lui.

Pour tout arrêter : **`stop_amadeus.bat`**.

> Sans carte NVIDIA, ou pour une installation légère : `install.bat -NoVoice`.

## La clé d'API

Kurisu réfléchit grâce à une **API d'IA compatible avec le format OpenAI**. Ta clé reste
**sur ta machine**, dans `backend\.env`. Ce fichier est exclu de Git : il ne part jamais
sur GitHub. Ne le partage pas. Pour changer de clé, relance `install.bat`, ou modifie
`backend\.env` avec le Bloc-notes puis relance Amadeus.

**Recommandations** (l'installateur te propose les deux) :

| Fournisseur | Pourquoi | Prix |
|---|---|---|
| **[Z.ai](https://z.ai)** | Le plus fidèle au personnage dans nos essais (japonais naturel, ton tsundere) | Payant, peu cher |
| **[Groq](https://console.groq.com)** | Environ 1 s plus rapide ; japonais un peu moins naturel | Gratuit (quota par minute) |

Tu peux configurer **les deux** : dans Amadeus, **Réglages → Cerveau** bascule entre le
cerveau principal et le cerveau rapide (Groq). Quand le quota gratuit de Groq est épuisé
(environ 2 à 3 messages par minute avec la mémoire de Kurisu), le message part
automatiquement sur le cerveau principal : la conversation ne se bloque jamais.

**Un autre fournisseur ?** N'importe quelle API compatible OpenAI fonctionne : dans
l'installateur, choisis « autre fournisseur » (option 3), ou renseigne ces trois lignes
dans `backend\.env` :

```
LLM_API_KEY=ta-cle
LLM_BASE_URL=https://api.fournisseur.com/v1
LLM_MODEL=nom-du-modele
```

## Avatar et logo (facultatifs)

Aucun modèle de personnage n'est fourni : ils ont chacun leur propre licence.

- **Avatar 3D** : dépose un fichier **`.vrm`** (export [VRoid Studio](https://vroid.com/studio),
  par exemple) n'importe où sur la fenêtre d'Amadeus, ou copie-le en
  `frontend\vendor\models\kurisu.vrm`. Sans modèle, un noyau holographique s'affiche.
  Glisse pour tourner autour, molette pour zoomer.
- **Portrait 2D (expérimental)** : aucune image 2D n'est fournie, **il faut importer les
  tiennes** : menu **Avatar → Importer des images 2D** (ou glisser-déposer). Une image par
  expression, nommée `neutre`, `joie`, `colere`, `triste`, `gene`, `surprise` (noms anglais
  acceptés) ; ajoute `_parle` pour la version bouche ouverte (`joie_parle.png`), utilisée
  pendant qu'elle parle. PNG détouré conseillé ; `neutre` est la seule image indispensable.
  **Ce mode n'a été testé qu'avec des images factices, pas avec de vrais dessins** :
  attends-toi à quelques imperfections.
- **Logo** : une image `frontend\vendor\brand\amadeus-logo.png`. Sans image, le nom
  AMADEUS s'affiche en texte.

## Utilisation

- **Mains libres** (micro ambre, actif par défaut) : parle quand tu veux. Si tu parles
  pendant qu'elle répond, elle baisse la voix, puis s'arrête si ta phrase est confirmée
  (une toux ou l'écho de sa propre voix sont ignorés). Un clic sur le micro coupe les
  mains libres : tu maintiens alors le micro, ou **Espace**, pour parler.
- **Conversation** : l'historique écrit de vos échanges.
- **Mémoire** : ce qu'elle retient de toi. Tu peux le corriger ; « Tout effacer » repart
  de zéro. Chaque échange est mémorisé (ChromaDB) et, toutes les 6 réponses, un résumé de
  qui tu es est mis à jour et réinjecté dans chaque conversation.
- **Réglages** (choix mémorisés) :
  - **Cerveau** : principal ou rapide (Groq) ;
  - **Avatar en 2D** : tes images 2D, ou à défaut le modèle 3D vu de face, à plat ;
  - **Fond écran Amadeus** : écran sombre, grille verte mouvante, code flou ;
  - **Compteur de divergence** en tubes nixie.

## Réglages avancés (`backend\.env`)

| Variable | Rôle |
|---|---|
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | Cerveau principal : clé, adresse de l'API, nom du modèle |
| `LLM_DISABLE_THINKING` | `true` (défaut) : coupe la « réflexion » des modèles qui en ont une, ~2 s au lieu de 10+ avant le premier mot |
| `GROQ_API_KEY` | Cerveau rapide (Groq), facultatif |
| `BRAIN` | Cerveau au tout premier lancement : `main` ou `gptoss` (rapide) |
| `STT_MODEL` | `large-v3-turbo` (défaut), `medium`, `small` (machines modestes) |
| `STT_LANGUAGE` | `auto` (défaut) ou `fr` : plus fiable pour les phrases courtes |
| `TTS_STREAMING_MODE` | `3` (défaut) : voix régulière, sans blanc ; `0` : désactivé |
| `TTS_TEMPERATURE` / `TTS_TOP_K` / `TTS_TOP_P` | `0.7` / `5` / `0.9` : voix stable |
| `TTS_LOUDNESS_DB` | `-16` : même volume pour toutes les émotions |

## Dépannage

| Problème | Solution |
|---|---|
| `install.bat` échoue | Le message dit quelle étape ; le détail est dans `data\install.log`. Corrige puis relance : les étapes faites sont sautées |
| « Kurisu ne peut pas répondre » | Clé manquante ou refusée : relance `install.bat`, ou vérifie `backend\.env` |
| « Voix de secours active » | GPT-SoVITS éteint ou absent : relance `start_amadeus.bat` (ou installe la voix avec `install.bat`) |
| Un service ne démarre pas | `start_amadeus.bat` indique lequel ; sa fenêtre (Amadeus-Voix, réduite, ou Amadeus-Backend) montre l'erreur. `stop_amadeus.bat`, puis relance |
| Le micro ne marche pas | Autorise le micro pour `127.0.0.1` dans ton navigateur (icône à gauche de l'adresse) |
| Voix qui coupe | Chaque trou de son et chaque interruption sont notés dans `data\voice_diag.log` |
| Réponses lentes | Essaie le cerveau rapide (Réglages → Cerveau) |
| VRM refusé | Le fichier doit être un `.vrm` valide (export VRoid Studio, par exemple) |

## Structure

```
amadeus/
├── install.bat            Installation (double-clic)
├── start_amadeus.bat      Démarre la voix + le backend, ouvre l'interface
├── stop_amadeus.bat       Arrête tout
├── scripts/               Logique des trois .bat (PowerShell)
├── backend/               FastAPI : cerveau (llm, persona), mémoire, Whisper, voix
│   ├── app/
│   ├── tools/             transcribe_refs.py (préparation de la voix)
│   └── .env.example       Modèle de configuration (copié en .env à l'installation)
├── frontend/              Interface (three.js + three-vrm, sans étape de build)
├── tts-server/            Voix : téléchargement des modèles, correctifs GPT-SoVITS
│   ├── patches/           Correctifs d'Amadeus appliqués à GPT-SoVITS
│   └── voice_refs.json    Extraits de référence (un par émotion) à télécharger
└── data/                  Tes données : mémoire, réglages, journaux (jamais publiées)
```

## Crédits

- Voix : modèle GPT-SoVITS de Kurisu entraîné par la communauté,
  [bysq/TTS-KurisuMakise](https://huggingface.co/bysq/TTS-KurisuMakise), téléchargé à
  l'installation. Usage personnel uniquement : ne redistribue pas la voix.
- Moteur de voix : [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) (MIT).
- Reconnaissance vocale : [faster-whisper](https://github.com/SYSTRAN/faster-whisper) ;
  détection de parole : [Silero VAD](https://github.com/snakers4/silero-vad) via
  [vad-web](https://github.com/ricky0123/vad).
- 3D : [three.js](https://threejs.org) et [three-vrm](https://github.com/pixiv/three-vrm).
- Steins;Gate, Makise Kurisu et Amadeus appartiennent à leurs ayants droit
  (MAGES. / Chiyomaru Studio).

Le code de ce dépôt est sous licence [MIT](LICENSE).
