/* Audio — voix de Kurisu en streaming (Web Audio), lip-sync, micro.

   La voix arrive de /api/tts :
   - audio/wav : en-tête WAV puis PCM 16 bits au fil de la génération
     (GPT-SoVITS en streaming) → on joue chaque morceau dès qu'il arrive ;
   - autre (mp3 edge-tts) : fichier complet décodé d'un coup.
   Les morceaux sont programmés bout à bout sur l'horloge audio : pas de trou. */

const GAP_BETWEEN_SENTENCES = 0.12;   /* respiration naturelle entre deux phrases (s) */
const START_BUFFER = 0.8;             /* audio d'avance voulu avant de démarrer une phrase (s) */
const START_WAIT_MAX = 0.35;          /* attente maximale pour ce tampon (s) */
const START_WAIT_MIN = 0.08;          /* attente quand le GPU a largement de l'avance (s) */
const UNDERRUN_PAD = 0.12;            /* marge de reprise après un retard (s) */
const DUCK_LEVEL = 0.18;              /* volume de Kurisu pendant que tu lui parles */
const LIP_MAX = 0.65;                 /* ouverture max d'un visème */
const LIP_TOTAL_MAX = 0.85;           /* ouverture max cumulée des visèmes */

const AudioSys = {
  ctx: null,
  out: null,          /* gain → analyser → haut-parleurs */
  analyser: null,
  _freq: null,
  _wave: null,
  _sources: new Set(),
  _nextTime: 0,
  _token: 0,          /* incrémenté à chaque interruption */
  lipBands: null,

  /* À appeler depuis un geste utilisateur : les navigateurs bloquent l'audio avant. */
  unlock() {
    if (!this.ctx) {
      this.ctx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: 'interactive' });
      this.out = this.ctx.createGain();
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 2048;
      this.analyser.smoothingTimeConstant = 0.55;
      this.out.connect(this.analyser);
      this.analyser.connect(this.ctx.destination);
      this._freq = new Uint8Array(this.analyser.frequencyBinCount);
      this._wave = new Uint8Array(this.analyser.fftSize);
    }
    if (this.ctx.state === 'suspended') this.ctx.resume();
  },

  get playing() { return this._sources.size > 0; },

  /* Journal des incidents de lecture (trous, interruptions), consultable via
     __amadeus.AudioSys.log dans la console ; recopié dans data/voice_diag.log */
  log: [],
  diag(kind, info = {}) {
    const e = { t: new Date().toLocaleTimeString(), kind, ...info };
    this.log.push(e);
    if (this.log.length > 200) this.log.shift();
    console.debug('[voix]', kind, info);
    fetch('/api/diag', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(e), keepalive: true }).catch(() => {});
  },

  /* Joue un flux TTS (TtsStream). onStart() est appelé quand le son démarre
     réellement. Résout juste avant la fin, pour enchaîner la phrase suivante sans trou. */
  async play(stream, onStart) {
    this.unlock();
    const ctx = this.ctx;
    const token = this._token;
    let started = false;
    /* Tampon de démarrage : GPT-SoVITS envoie parfois un 1er morceau court (~0,6 s)
       suivi d'une pause ; démarrer trop tôt crée un blanc au milieu de la phrase.
       On attend d'avoir START_BUFFER d'audio d'avance, d'autant moins longtemps que
       la génération va vite. */
    if (this._nextTime <= ctx.currentTime) {
      await stream.buffered(START_BUFFER, this._startWait());
      if (this._cautious > 0) this._cautious--;
    }
    for await (const chunk of stream) {
      if (token !== this._token) return false;
      const buf = chunk instanceof AudioBuffer ? chunk : this._toBuffer(chunk, stream.sampleRate);
      if (!buf.length) continue;
      const gap = started ? 0 : GAP_BETWEEN_SENTENCES;
      /* morceau en retard (la lecture a rattrapé la génération) : on reprend avec
         un peu de marge plutôt que de hacher mot par mot */
      const late = started && this._nextTime < ctx.currentTime;
      const t = Math.max(this._nextTime + gap, ctx.currentTime + (late ? UNDERRUN_PAD : 0.03));
      if (late) {
        this._cautious = 3;   /* un trou : on reprend le grand tampon pour quelques phrases */
        this.diag('trou', { s: +(t - this._nextTime).toFixed(2), recu: +stream.seconds.toFixed(2), texte: stream.text });
      }
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(this.out);
      src.onended = () => this._sources.delete(src);
      this._sources.add(src);
      src.start(t);
      this._nextTime = t + buf.duration;
      if (!started) {
        started = true;
        if (onStart) setTimeout(onStart, Math.max(0, (t - ctx.currentTime) * 1000));
      }
    }
    /* attend la fin (à 150 ms près) sauf interruption */
    while (token === this._token && this._nextTime - ctx.currentTime > 0.15) {
      await new Promise((r) => setTimeout(r, Math.min(100, (this._nextTime - ctx.currentTime - 0.15) * 1000)));
    }
    return started && token === this._token;
  },

  /* Attente du tampon selon la vitesse mesurée (secondes d'audio par seconde de calcul) :
     inconnue ou proche du temps réel → prudence ; 2× ou plus → démarrage quasi immédiat. */
  _genRatio: null,
  _cautious: 0,
  _startWait() {
    if (this._cautious > 0 || this._genRatio === null) return START_WAIT_MAX;
    const k = Math.min(1, Math.max(0, (this._genRatio - 1.2) / 0.8));
    return START_WAIT_MAX - k * (START_WAIT_MAX - START_WAIT_MIN);
  },
  noteGeneration(ratio) {
    this._genRatio = this._genRatio === null ? ratio : this._genRatio * 0.7 + ratio * 0.3;
  },

  /* Baisse (on) ou rétablit (off) sa voix en douceur, sans rien couper. */
  duck(on) {
    if (!this.ctx) return;
    const g = this.out.gain, now = this.ctx.currentTime;
    g.cancelScheduledValues(now);
    g.setValueAtTime(g.value, now);
    g.setTargetAtTime(on ? DUCK_LEVEL : 1, now, on ? 0.05 : 0.15);
  },

  /* Coupe immédiatement tout ce qui est programmé. */
  stop() {
    if (this.ctx) {
      this.out.gain.cancelScheduledValues(this.ctx.currentTime);
      this.out.gain.setValueAtTime(1, this.ctx.currentTime);
    }
    this._token++;
    for (const s of this._sources) { try { s.stop(); } catch { /* déjà fini */ } }
    this._sources.clear();
    this._nextTime = 0;
  },

  _toBuffer(f32, sampleRate) {
    const buf = this.ctx.createBuffer(1, f32.length, sampleRate);
    buf.copyToChannel(f32, 0);
    return buf;
  },

  /* Appelée à chaque frame par la boucle de rendu de l'avatar. */
  updateLips() {
    if (!this.playing) { this.lipBands = null; return null; }
    const bins = this._freq;
    this.analyser.getByteFrequencyData(bins);
    const binHz = this.ctx.sampleRate / 2 / bins.length;
    const band = (lo, hi) => {
      const a = Math.max(0, Math.floor(lo / binHz));
      const b = Math.min(bins.length - 1, Math.ceil(hi / binHz));
      let s = 0;
      for (let i = a; i <= b; i++) s += bins[i];
      return s / ((b - a + 1) * 255);
    };
    /* F1 ≈ ouverture de bouche, F2 ≈ forme voyelle, aigus ≈ fricatives */
    const low = band(90, 700), mid = band(700, 1900), high = band(1900, 4600);
    const vol = Math.min(1, low * 1.35 + mid * 0.8 + high * 0.5);
    const raw = {
      aa: vol * (0.55 + low * 0.9),
      ih: vol * (0.2 + mid * 0.9),
      ee: vol * mid * 0.5,
      ou: vol * (0.1 + high * 0.8),
      oh: vol * low * 0.7,
    };
    /* plafonds : au-delà, la bouche du modèle traverse le menton */
    let sum = 0;
    for (const k in raw) { raw[k] = Math.min(raw[k], LIP_MAX); sum += raw[k]; }
    if (sum > LIP_TOTAL_MAX) for (const k in raw) raw[k] *= LIP_TOTAL_MAX / sum;
    this.lipBands = raw;
    return this.lipBands;
  },

  /* Forme d'onde courante de sa voix (Uint8Array, 128 = silence) ou null. */
  waveform() {
    if (!this.playing) return null;
    this.analyser.getByteTimeDomainData(this._wave);
    return this._wave;
  },
};

/* ------------------------------ Flux TTS ------------------------------
   Démarre la requête dès sa création et accumule l'audio décodé : la phrase
   suivante se télécharge pendant que la précédente est jouée. */

class TtsStream {
  constructor(text, emotion) {
    this.text = text;
    this.sampleRate = 32000;
    this.chunks = [];
    this.done = false;
    this.error = null;
    this._waiters = [];
    this._abort = new AbortController();
    this._run(text, emotion).catch((e) => { this.error = e; }).finally(() => { this.done = true; this._notify(); });
  }

  cancel() { this._abort.abort(); }

  /* durée d'audio déjà reçue (s) */
  get seconds() {
    let n = 0;
    for (const c of this.chunks) n += c instanceof AudioBuffer ? c.duration : c.length / this.sampleRate;
    return n;
  }

  /* attend d'avoir `sec` secondes d'audio (ou la fin du flux), au plus `maxWait` s
     après le 1er morceau */
  async buffered(sec, maxWait) {
    while (!this.chunks.length && !this.done) await new Promise((r) => { this._waiters.push(r); });
    const until = performance.now() + maxWait * 1000;
    while (!this.done && this.seconds < sec && performance.now() < until) {
      await new Promise((r) => {
        this._waiters.push(r);
        setTimeout(r, Math.max(0, until - performance.now()));
      });
    }
  }

  _notify() { const ws = this._waiters; this._waiters = []; for (const w of ws) w(); }

  _push(chunk) { this.chunks.push(chunk); this._notify(); }

  async _run(text, emotion) {
    const r = await fetch('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, emotion }),
      signal: this._abort.signal,
    });
    if (!r.ok) throw new Error(`TTS ${r.status}`);
    const type = r.headers.get('content-type') || '';

    if (!type.includes('wav') || !r.body) {
      /* mp3 complet (voix de secours) */
      AudioSys.unlock();
      const data = await r.arrayBuffer();
      this._push(await AudioSys.ctx.decodeAudioData(data));
      return;
    }

    /* WAV en streaming : en-tête puis PCM 16 bits little-endian */
    const reader = r.body.getReader();
    let head = new Uint8Array(0);
    let dataStarted = false;
    let carry = null;  /* octet orphelin entre deux morceaux */
    let firstAt = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) {
        /* vitesse réelle de génération ; un fichier déjà en cache arrive d'un bloc */
        const took = firstAt ? (performance.now() - firstAt) / 1000 : 0;
        if (took > 0.25 && this.seconds > 0.5) AudioSys.noteGeneration(this.seconds / took);
        break;
      }
      firstAt ||= performance.now();
      let bytes = value;
      if (!dataStarted) {
        head = concat(head, bytes);
        const off = findWavData(head, (sr) => { this.sampleRate = sr; });
        if (off < 0) continue;
        dataStarted = true;
        bytes = head.subarray(off);
      }
      if (carry) { bytes = concat(carry, bytes); carry = null; }
      if (bytes.length % 2) { carry = bytes.slice(-1); bytes = bytes.subarray(0, bytes.length - 1); }
      if (!bytes.length) continue;
      const pcm = new Int16Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length));
      const f32 = new Float32Array(pcm.length);
      for (let i = 0; i < pcm.length; i++) f32[i] = pcm[i] / 32768;
      this._push(f32);
    }
  }

  async *[Symbol.asyncIterator]() {
    let k = 0;
    for (;;) {
      while (k < this.chunks.length) yield this.chunks[k++];
      if (this.done) return;
      await new Promise((r) => { this._waiters.push(r); });
    }
  }
}

function concat(a, b) {
  const out = new Uint8Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

/* → index du début des données PCM, ou -1 si l'en-tête n'est pas encore complet */
function findWavData(h, onRate) {
  if (h.length < 12) return -1;
  const dv = new DataView(h.buffer, h.byteOffset, h.length);
  let p = 12;
  while (p + 8 <= h.length) {
    const id = String.fromCharCode(h[p], h[p + 1], h[p + 2], h[p + 3]);
    const size = dv.getUint32(p + 4, true);
    if (id === 'fmt ' && p + 16 <= h.length) onRate(dv.getUint32(p + 12, true));
    if (id === 'data') return p + 8 <= h.length ? p + 8 : -1;
    p += 8 + size + (size % 2);
  }
  return -1;
}

/* ------------------------- Micro : maintenir pour parler ------------------------- */

const Recorder = {
  media: null,
  chunks: [],
  active: false,
  _stream: null,
  _t0: 0,
  _analyser: null,
  _wave: null,
  _src: null,

  async start() {
    if (this.active) return;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : '';
    this.media = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    this.chunks = [];
    this.media.ondataavailable = (e) => e.data.size && this.chunks.push(e.data);
    this.media.start(200);
    this.active = true;
    this._t0 = performance.now();
    this._stream = stream;

    AudioSys.unlock();
    const ctx = AudioSys.ctx;
    if (!this._analyser) {
      this._analyser = ctx.createAnalyser();
      this._analyser.fftSize = 1024;
      this._wave = new Uint8Array(this._analyser.fftSize);
    }
    this._src = ctx.createMediaStreamSource(stream);
    this._src.connect(this._analyser);
  },

  waveform() {
    if (!this.active || !this._analyser) return null;
    this._analyser.getByteTimeDomainData(this._wave);
    return this._wave;
  },

  /* → Blob audio, ou null si trop court */
  stop() {
    return new Promise((resolve) => {
      if (!this.active || !this.media) return resolve(null);
      const dur = performance.now() - this._t0;
      this.media.onstop = () => {
        this._stream.getTracks().forEach((t) => t.stop());
        try { this._src?.disconnect(); } catch { /* déjà déconnecté */ }
        this._src = null;
        this.active = false;
        if (dur < 350 || this.chunks.length === 0) return resolve(null);
        resolve(new Blob(this.chunks, { type: this.chunks[0].type }));
      };
      this.media.stop();
    });
  },
};

/* ------------------------- Mains libres : détection de la parole -------------------------
   Silero VAD (réseau de neurones) dans le navigateur : détecte quand tu commences
   et finis de parler. Pendant que Kurisu parle, il faut une voix plus nette et
   plus longue pour lui couper la parole (évite qu'elle s'interrompe sur son propre écho). */

const VAD_BASE = 'https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.31/dist/';
const ORT_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.1/dist/';

const CALM = { positiveSpeechThreshold: 0.6, negativeSpeechThreshold: 0.4, minSpeechMs: 250 };
const WHILE_SHE_TALKS = { positiveSpeechThreshold: 0.85, negativeSpeechThreshold: 0.6, minSpeechMs: 450 };

/* Fin de phrase : le silence attendu dépend de ce qui vient d'être dit (ms). */
const FRAME_MS = 32;
const PREPARE_MS = 200;     /* Whisper et la mémoire démarrent, sans certitude */
const END_COMPLETE_MS = 320;   /* phrase visiblement finie */
const END_DEFAULT_MS = 550;    /* pas encore de transcription */
const END_UNFINISHED_MS = 850; /* « parce que… », « euh… » : elle attend la suite */
const VAD_END_LIVE_MS = END_UNFINISHED_MS + 150;
const VAD_END_BATCH_MS = 550;
const PREROLL_FRAMES = 10;  /* ~320 ms d'audio gardées avant le début détecté */
const FINAL_TIMEOUT_MS = 8000;

/* ------------------------- Liaison directe avec le backend -------------------------
   Le micro part en PCM 16 kHz pendant que tu parles (voir backend/app/live.py). Si elle
   tombe, les phrases repartent par POST /api/stt : rien n'est perdu. */

class LiveLink {
  constructor(session, onMessage, onState) {
    this.session = session;
    this.onMessage = onMessage;
    this.onState = onState;
    this.ws = null;
    this._retry = 500;
    this._closed = false;
  }

  get ready() { return this.ws?.readyState === WebSocket.OPEN; }

  connect() {
    if (this._closed || this.ws) return;
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${proto}//${location.host}/api/listen?session=${encodeURIComponent(this.session)}`);
    ws.binaryType = 'arraybuffer';
    this.ws = ws;
    ws.onopen = () => { this._retry = 500; this.onState(true); };
    ws.onmessage = (e) => {
      if (typeof e.data !== 'string') return;
      try { this.onMessage(JSON.parse(e.data)); } catch (err) { console.warn('[voix] message direct illisible', err); }
    };
    ws.onclose = () => {
      this.ws = null;
      this.onState(false);
      if (this._closed) return;
      setTimeout(() => this.connect(), this._retry);
      this._retry = Math.min(this._retry * 2, 10000);
    };
  }

  send(msg) { if (this.ready) this.ws.send(JSON.stringify(msg)); }

  audio(frames) {
    if (!this.ready) return;
    const n = frames.reduce((sum, f) => sum + f.length, 0);
    const pcm = new Int16Array(n);
    let p = 0;
    for (const f of frames) {
      for (let i = 0; i < f.length; i++) {
        const s = Math.max(-1, Math.min(1, f[i]));
        pcm[p++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
    }
    this.ws.send(pcm.buffer);
  }
}

const HandsFree = {
  vad: null,
  on: false,
  speaking: false,
  link: null,
  _wave: new Uint8Array(512),
  _hasWave: false,
  _strict: false,
  _segment: false,      /* le VAD considère qu'on est dans un tour de parole */
  _segmentLive: false,  /* ce tour est passé par la liaison directe */
  _preroll: [],
  _utt: null,           /* phrase en cours d'envoi */
  _uid: 0,
  _pending: new Map(),  /* phrases terminées, en attente de leur transcription */
  handlers: {
    wouldBargeIn: () => false, onStart() {}, onEnd() {}, onMisfire() {},
    onPartial() {}, onCommit() {}, onFinal() {}, onCancel() {},
  },

  /* Ouvre la liaison directe (facultative : sans elle, comportement par phrase entière). */
  connectLive(session) {
    if (this.link) return;
    this.link = new LiveLink(session, (m) => this._onLive(m), (up) => this._onLinkState(up));
    this.link.connect();
  },

  _onLinkState(up) {
    this.vad?.setOptions({ redemptionMs: up ? VAD_END_LIVE_MS : VAD_END_BATCH_MS });
    if (!up) for (const u of [...this._pending.values()]) this._fallback(u);
  },

  _onLive(m) {
    if (m.t === 'partial') {
      if (this._utt?.id === m.id) this.handlers.onPartial(m.text, this._utt);
    } else if (m.t === 'prepared') {
      if (this._utt?.id === m.id && this._utt.prepared) this._utt.complete = !!m.complete;
    } else if (m.t === 'final' || m.t === 'error') {
      const u = this._pending.get(m.id);
      if (!u) return;
      if (m.t === 'error') { this._fallback(u); return; }
      clearTimeout(u.timer);
      this._pending.delete(m.id);
      this.handlers.onFinal(m, u);
    }
  },

  _fallback(u) {
    clearTimeout(u.timer);
    this._pending.delete(u.id);
    const n = u.frames.reduce((sum, f) => sum + f.length, 0);
    const audio = new Float32Array(n);
    let p = 0;
    for (const f of u.frames) { audio.set(f, p); p += f.length; }
    this.handlers.onEnd(audio, u);
  },

  _begin(resumed = false) {
    this._segmentLive = true;
    const u = {
      id: ++this._uid, frames: [...this._preroll], speechMs: 0, silenceMs: 0,
      prepared: false, complete: null, bargeIn: !!this.handlers.wouldBargeIn(),
      confirmed: false, li: null, startAt: performance.now(), commitAt: 0, speechEndAt: 0,
    };
    this._utt = u;
    this.link.send({ t: 'start', id: u.id, bargeIn: u.bargeIn });
    this.link.audio(u.frames);
    if (resumed) this.handlers.onStart();
  },

  _commit(u) {
    this._utt = null;
    u.commitAt = performance.now();
    u.speechEndAt = u.commitAt - u.silenceMs;
    u.unfinished = u.complete === false;
    this.handlers.onCommit(u);
    if (!this.link?.ready) { this._fallback(u); return; }
    this._pending.set(u.id, u);
    u.timer = setTimeout(() => { if (this._pending.has(u.id)) this._fallback(u); }, FINAL_TIMEOUT_MS);
    this.link.send({ t: 'commit', id: u.id });
  },

  _cancel(u) {
    this._utt = null;
    this.link?.send({ t: 'cancel', id: u.id });
    this.handlers.onCancel(u);
  },

  _onFrame(p, frame) {
    this._preroll.push(frame);
    if (this._preroll.length > PREROLL_FRAMES) this._preroll.shift();
    const opts = this._strict ? WHILE_SHE_TALKS : CALM;
    const isSpeech = p.isSpeech >= opts.positiveSpeechThreshold;
    const u = this._utt;
    if (!u) {
      /* tu reprends après une fin de phrase anticipée, dans le même tour du VAD */
      if (this._segment && isSpeech && this.link?.ready) this._begin(true);
      if (this._utt) this._utt.speechMs += FRAME_MS;
      return;
    }
    u.frames.push(frame);
    this.link.audio([frame]);
    if (isSpeech) {
      u.speechMs += FRAME_MS;
      if (u.prepared) {
        u.prepared = false;
        u.complete = null;
        this.link.send({ t: 'resume', id: u.id });
      }
      u.silenceMs = 0;
      return;
    }
    if (p.isSpeech >= opts.negativeSpeechThreshold) return;
    u.silenceMs += FRAME_MS;
    if (u.speechMs < opts.minSpeechMs) {
      if (u.silenceMs >= END_UNFINISHED_MS) this._cancel(u);   /* bruit bref */
      return;
    }
    if (!u.prepared && u.silenceMs >= PREPARE_MS) {
      u.prepared = true;
      this.link.send({ t: 'prepare', id: u.id });
    }
    const target = u.complete === true ? END_COMPLETE_MS
      : u.complete === false ? END_UNFINISHED_MS : END_DEFAULT_MS;
    if (u.silenceMs >= target) this._commit(u);
  },

  _onSegmentEnd(audio) {
    this._segment = false;
    const u = this._utt;
    if (u) {
      const min = (this._strict ? WHILE_SHE_TALKS : CALM).minSpeechMs;
      if (u.speechMs >= min) this._commit(u); else this._cancel(u);
    } else if (!this._segmentLive) {
      this.handlers.onEnd(audio, null);
    }
  },

  async enable() {
    if (!window.vad?.MicVAD) throw new Error('détecteur de parole non chargé (connexion internet ?)');
    if (!this.vad) {
      this.vad = await window.vad.MicVAD.new({
        model: 'v5',
        baseAssetPath: VAD_BASE,
        onnxWASMBasePath: ORT_BASE,
        ...CALM,
        /* silence qui clôt le tour ; avec la liaison directe, _onFrame décide souvent avant */
        redemptionMs: this.link?.ready ? VAD_END_LIVE_MS : VAD_END_BATCH_MS,
        preSpeechPadMs: 300,
        getStream: () => navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        }),
        onSpeechStart: () => {
          this._segment = true;
          this._segmentLive = false;
          if (this.link?.ready) this._begin();
        },
        onSpeechRealStart: () => { this.speaking = true; this.handlers.onStart(); },
        onSpeechEnd: (audio) => { this.speaking = false; this._hasWave = false; this._onSegmentEnd(audio); },
        onVADMisfire: () => {
          this.speaking = false;
          this._hasWave = false;
          this._segment = false;
          if (this._utt) this._cancel(this._utt);
          this.handlers.onMisfire();
        },
        onFrameProcessed: (p, frame) => {
          if (frame) this._onFrame(p, frame);
          if (!this.speaking || !frame) return;
          const n = Math.min(frame.length, this._wave.length);
          for (let i = 0; i < n; i++) this._wave[i] = 128 + Math.max(-127, Math.min(127, frame[i] * 400));
          this._hasWave = true;
        },
      });
    }
    await this.vad.start();
    this.on = true;
  },

  async disable() {
    this.on = false;
    this.speaking = false;
    this._hasWave = false;
    this._segment = false;
    if (this._utt) this._cancel(this._utt);
    this._preroll = [];
    if (this.vad) await this.vad.pause();
  },

  /* plus exigeant pendant que Kurisu parle */
  setStrict(strict) {
    if (!this.vad || strict === this._strict) return;
    this._strict = strict;
    this.vad.setOptions(strict ? WHILE_SHE_TALKS : CALM);
  },

  waveform() { return this.on && this.speaking && this._hasWave ? this._wave : null; },

  /* Float32 16 kHz → WAV (pour /api/stt) */
  toWav(samples) {
    const sr = 16000;
    const buf = new ArrayBuffer(44 + samples.length * 2);
    const v = new DataView(buf);
    const w = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
    w(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); w(8, 'WAVE');
    w(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    w(36, 'data'); v.setUint32(40, samples.length * 2, true);
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([buf], { type: 'audio/wav' });
  },
};

export { AudioSys, TtsStream, Recorder, HandsFree };
