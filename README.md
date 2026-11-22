# AMADEUS — an AI companion like Makise Kurisu from Steins;Gate

**English** · [Français](README.fr.md)

A voice companion in the spirit of **Amadeus**: an AI with Kurisu Makise's personality
(sarcastic, tsundere, passionate about science), with **long-term memory**, **speech
recognition**, a **3D avatar** and a **Japanese voice** generated in real time by GPT-SoVITS.

You talk to her in English or French: she answers out loud in Japanese, with subtitles in
your language, like a dub.

```
Browser                                 Backend (FastAPI)              Models
├─ VRM 3D avatar / 2D portrait   ←SSE→  ├─ /api/chat : streamed LLM   ├─ AI API (OpenAI-compatible)
├─ Hands-free (Silero VAD)              ├─ /api/tts  : streamed voice ├─ GPT-SoVITS (GPU, local)
├─ Streaming audio playback             ├─ /api/stt  : Whisper        └─ faster-whisper (local)
└─ Subtitles, memory, settings          └─ ChromaDB + memory profile
```

Kurisu starts speaking **about 3 s** after you finish your sentence (2 s with the fast
brain): everything is streamed, and the voice starts with the very first written sentence.

> Fan project, not affiliated with MAGES. / Chiyomaru Studio. **No content from the game or
> the anime is included in this repository**: no voice, no image, no 3D model. The
> installer downloads the community voice model from Hugging Face, and you add an avatar
> and a logo yourself if you want (see below).

---

## Requirements

| | Minimum | Recommended |
|---|---|---|
| System | Windows 10 or 11 (64-bit) | |
| Graphics card | — (fallback voice without a GPU) | **NVIDIA**, 6 GB of video memory or more |
| Disk space | 5 GB | 15 GB (with the real voice) |
| Browser | Chrome, Edge or Firefox (Zen works too) | |
| API key | An **AI API key** (see below) | |

The installer takes care of **Python 3.12** and **Git** if they are missing (through
`winget`, with your permission). Without an NVIDIA card everything works except the real
voice: Kurisu then speaks with a fallback synthetic voice.

## Installation

1. **Download the project**: green **Code → Download ZIP** button on GitHub, then extract
   it, for example to `Documents\amadeus`. Or, with Git:
   ```
   git clone https://github.com/rbloxicraft/amadeus.git
   ```
   Avoid synced folders (OneDrive): the installation takes several GB.
2. **Get an AI API key** (see [The API key](#the-api-key) just below).
3. **Double-click `install.bat`**. It:
   - checks the machine (Python 3.12, Git, NVIDIA card, disk space);
   - installs the backend (brain, ears, memory);
   - installs the GPT-SoVITS voice engine, with PyTorch for the graphics card;
   - downloads Kurisu's voice from Hugging Face and prepares it;
   - asks for your API key and saves it.

   Allow **20 to 40 minutes** the first time, depending on your connection (about 5 GB to
   download). If a step fails, fix the problem and run `install.bat` again: steps already
   done are skipped. Everything is logged in `data\install.log`.

   The installer speaks French; its questions are short (choose 1, 2 or 3, paste your key).
4. **Double-click `start_amadeus.bat`**. It starts the voice and the backend, waits until
   they respond, then opens http://127.0.0.1:8642 in your browser. On the very first
   launch, Whisper and the memory still download their models (~2 GB).
5. Click **Se connecter** (the browser requires a click before playing sound), allow the
   microphone, and talk to her.

To stop everything: **`stop_amadeus.bat`**.

> Without an NVIDIA card, or for a light installation: `install.bat -NoVoice`.

## The API key

Kurisu thinks through an **AI API compatible with the OpenAI format**. Your key stays **on
your machine**, in `backend\.env`. That file is excluded from Git: it never goes to GitHub.
Don't share it. To change a key, run `install.bat` again, or edit `backend\.env` with
Notepad and restart Amadeus.

**Recommendations** (the installer offers both):

| Provider | Why | Price |
|---|---|---|
| **[Z.ai](https://z.ai)** | The most faithful to the character in our tests (natural Japanese, tsundere tone) | Paid, inexpensive |
| **[Groq](https://console.groq.com)** | About 1 s faster; Japanese a little less natural | Free (per-minute quota) |

You can set up **both**: in Amadeus, **Réglages → Cerveau** switches between the main brain
and the fast one (Groq). When Groq's free quota runs out (about 2 to 3 messages a minute
with Kurisu's memory), the message automatically goes to the main brain: the conversation
never freezes.

**Another provider?** Any OpenAI-compatible API works: in the installer choose
"autre fournisseur" (option 3), or set these three lines in `backend\.env`:

```
LLM_API_KEY=your-key
LLM_BASE_URL=https://api.provider.com/v1
LLM_MODEL=model-name
```

## Avatar and logo (optional)

No character model is provided: each one has its own license.

- **3D avatar**: drop a **`.vrm`** file (exported from [VRoid Studio](https://vroid.com/studio),
  for example) anywhere on the Amadeus window, or copy it to
  `frontend\vendor\models\kurisu.vrm`. Without a model, a holographic core is shown. Drag to
  orbit, scroll to zoom.
- **2D portrait (experimental)**: there are no 2D images in the project, **you have to import
  your own**: **Avatar → Importer des images 2D** menu (or drag and drop). One image per
  expression, named `neutral`, `joy`, `angry`, `sad`, `shy`, `surprised` (French names also
  accepted); add `_talk` for the open-mouth version (`joy_talk.png`), used while she speaks.
  A cut-out PNG works best; `neutral` is the only required image. **This mode has only been
  tested with placeholder images, not with real artwork**: expect rough edges.
- **Logo**: an image at `frontend\vendor\brand\amadeus-logo.png`. Without it, the name
  AMADEUS is shown as text.

## Using it

The interface is in French. Main elements:

- **Hands-free** (amber microphone, on by default): speak whenever you want. If you talk
  while she's answering, she lowers her voice, then stops if your sentence is confirmed (a
  cough or the echo of her own voice are ignored). Clicking the microphone turns hands-free
  off: then hold the microphone, or **Space**, to talk.
- **Conversation**: the written history of your exchanges.
- **Mémoire**: what she remembers about you. You can edit it; "Tout effacer" starts over.
  Every exchange is stored (ChromaDB) and, every 6 replies, a summary of who you are is
  updated and injected into every conversation.
- **Réglages** (choices are remembered):
  - **Cerveau**: main brain or fast brain (Groq);
  - **Avatar en 2D**: your 2D images, or else the 3D model seen flat from the front;
  - **Fond écran Amadeus**: dark screen, moving green grid, blurred code;
  - **Compteur de divergence** with nixie tubes.

## Advanced settings (`backend\.env`)

| Variable | Purpose |
|---|---|
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | Main brain: key, API address, model name |
| `LLM_DISABLE_THINKING` | `true` (default): turns off "thinking" for models that have it, ~2 s instead of 10+ before the first word |
| `GROQ_API_KEY` | Fast brain (Groq), optional. Also powers Kurisu's web search |
| `DEEPSEEK_API_KEY` | DeepSeek brain, optional (paid, very cheap): fast and steady, no per-minute quota; also the fallback when Groq's free quota runs out |
| `BRAIN` | Brain on the very first launch: `main`, `gptoss` (fast) or `deepseek` |
| `WEATHER_CITY` | Default city for the weather tool (Kurisu also knows the date and time, and can search the web) |
| `STT_MODEL` | `large-v3-turbo` (default), `medium`, `small` (modest machines) |
| `STT_LANGUAGE` | `auto` (default), `en` or `fr`: more reliable for short sentences |
| `STT_PARTIALS` | Live transcription while you speak. `auto` (default): on with an NVIDIA GPU; `on`; `off` |
| `VOICE_ANALYSIS` | `true` (default): tells Kurisu how you sounded (pace, pitch, pauses) compared with your usual voice |
| `VOICE_EMOTION` | `false` (default). `true` adds emotion2vec on CPU, after installing `backend\requirements-emotion.txt` (over 1 GB) |
| `TTS_EARLY_PHRASE` | `true` (default): starts the voice on the first clause of a reply |
| `TTS_REACTIONS` | `true` (default): plays a short pre-generated reaction if the reply is slow |
| `TTS_STREAMING_MODE` | `3` (default): steady voice, no gaps; `0`: off |
| `TTS_TEMPERATURE` / `TTS_TOP_K` / `TTS_TOP_P` | `0.7` / `5` / `0.9`: stable voice |
| `TTS_LOUDNESS_DB` | `-16`: same volume for every emotion |

## Troubleshooting

| Problem | Solution |
|---|---|
| `install.bat` fails | The message names the step; details are in `data\install.log`. Fix it and run it again: completed steps are skipped |
| "Kurisu ne peut pas répondre" | Missing or rejected key: run `install.bat` again, or check `backend\.env` |
| "Voix de secours active" | GPT-SoVITS is off or missing: run `start_amadeus.bat` again (or install the voice with `install.bat`) |
| A service won't start | `start_amadeus.bat` says which one; its window (Amadeus-Voix, minimized, or Amadeus-Backend) shows the error. Run `stop_amadeus.bat`, then start again |
| The microphone doesn't work | Allow the microphone for `127.0.0.1` in your browser (icon left of the address bar) |
| The voice cuts out | Every audio gap and interruption is logged in `data\voice_diag.log` |
| Slow answers | Try the fast brain (Réglages → Cerveau). Each voice turn's timings and API cache hit rate are logged as `latence` in `data\voice_diag.log` |
| VRM rejected | The file must be a valid `.vrm` (VRoid Studio export, for example) |

## Project layout

```
amadeus/
├── install.bat            Installation (double-click)
├── start_amadeus.bat      Starts the voice + backend, opens the interface
├── stop_amadeus.bat       Stops everything
├── scripts/               Logic of the three .bat files (PowerShell)
├── backend/               FastAPI: brain (llm, persona), memory, Whisper, voice
│   ├── app/
│   ├── tools/             transcribe_refs.py (voice preparation)
│   └── .env.example       Configuration template (copied to .env on install)
├── frontend/              Interface (three.js + three-vrm, no build step)
├── tts-server/            Voice: model download, GPT-SoVITS patches
│   ├── patches/           Amadeus patches applied to GPT-SoVITS
│   └── voice_refs.json    Reference clips (one per emotion) to download
└── data/                  Your data: memory, settings, logs (never published)
```

## Credits

- Voice: community-trained GPT-SoVITS model of Kurisu,
  [bysq/TTS-KurisuMakise](https://huggingface.co/bysq/TTS-KurisuMakise), downloaded at
  install time. Personal use only: don't redistribute the voice.
- Voice engine: [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) (MIT).
- Speech recognition: [faster-whisper](https://github.com/SYSTRAN/faster-whisper); voice
  activity detection: [Silero VAD](https://github.com/snakers4/silero-vad) via
  [vad-web](https://github.com/ricky0123/vad).
- 3D: [three.js](https://threejs.org) and [three-vrm](https://github.com/pixiv/three-vrm).
- Steins;Gate, Makise Kurisu and Amadeus belong to their respective owners
  (MAGES. / Chiyomaru Studio).

The code in this repository is under the [MIT](LICENSE) license.
