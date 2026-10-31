import { useEffect, useMemo, useRef, useState, type MouseEvent } from 'react';
import { ATTRIBUTION, cells, classColour, label, pageName, pictogramUrl, topicsOf, type Cell, type Gender, type LabelLang } from '../vocab';
import { speakAsync, useHasArabicVoice } from '../speech/tts';
import { DEFAULT_DETECTOR, startMic, TARGET_RATE, UtteranceDetector, type MicHandle } from '../audio/capture';
import { earcon, OutputGate, playAudio, unlockAudio } from '../audio/output';
import { loadRecordings, saveRecording } from '../storage/voiceStore';
import { addTemplate, deleteTemplate, loadTemplates, type TemplateSource } from '../storage/templateStore';
import { VoiceModel, wordFeatures, type SelfTest } from '../voice/voiceModel';
import { decide, fuse, ranked, type Channel, type Decision } from '../fusion/fuse';
import { PointerHistory, pointerEvidence, type PointerSample, type Rect } from '../fusion/pointer';
import { LevelMeter, useLevels } from '../components/LevelMeter';
import { useFitGrid } from '../components/fitGrid';
import { GazeTracker } from '../gaze/tracker';
import { Calibration, type CalibrationEnd } from '../gaze/Calibration';
import { GazeOverlay } from '../gaze/GazeOverlay';
import { loadModel, saveModel, type FitResult, type GazeModel, type PointerMode } from '../gaze/model';

const LANGS = ['lb', 'msa', 'en'] as const;
const GENDERS = ['m', 'f'] as const;
const POINTER_MODES = ['head', 'eyes'] as const;
const TOUCH_MODES = ['exact', 'rough'] as const;
const DWELL_CHOICES = ['0', '800', '1200', '2000', '3000'] as const;
const PANEL = ['open', 'closed'] as const;

const TEACH_COUNT = 3; // recordings asked for each time a word is taught
const LEARN_CONFIDENCE = 0.85; // a choice at least this sure also teaches the app how the word sounds...
const LEARN_UNTIL = 4; // ...while the word has fewer recordings than this
const MAX_PER_WORD = 8; // past this, a newly learned recording replaces the oldest learned one
const ASK_MS = 8000; // how long a "did you mean…?" stays open
const PRIOR_WEIGHT = 0.7; // weight of the earlier tries when the word is said again
const TOUCH_WAIT_MS = 2000; // rough touch: how long to wait for the word before the touch decides alone
const TOUCH_GIVE_UP_MS = 6000; // and the longest wait for a word that has already started
const TOUCH_SPREAD = 0.35; // rough touch: how far off a touch may be, as a fraction of the cell size
const GAZE_SPREAD = 1.5; // the calibration error underestimates the error in use (the head drifts)
const EYE_LEAD_MS = 700; // people look at a picture a little before they name it
const UNDO_MS = 15000; // deleting a voice choice this soon also forgets what was learned from it
const ECHO_MARGIN_MS = 250; // speech starting this soon after the app's own sound may be its echo
// The detector reports a word after a pause; its audio has a short lead-in before the speech and a tail after.
const SPEECH_OFFSET_MS = DEFAULT_DETECTOR.hangoverMs - DEFAULT_DETECTOR.preRollMs - DEFAULT_DETECTOR.tailMs;

type Mode = 'use' | 'teach' | 'record';

/** A "did you mean…?" in progress: what the evidence says so far, and every try at the word. */
interface Episode {
  time: number;
  posterior: Map<string, number>;
  tries: Float32Array[];
}

/** Why the app chose what it did, shown while listening. */
interface Explanation {
  seconds: number;
  speech: [string, number][] | null;
  pointer: [string, number][] | null;
  pointerKind: 'touch' | 'camera';
  again: boolean;
  learned: boolean;
  decision: Decision;
}

function load<T extends string>(key: string, allowed: readonly T[], fallback: T): T {
  try {
    const value = window.localStorage.getItem(key);
    return value !== null && (allowed as readonly string[]).includes(value) ? (value as T) : fallback;
  } catch {
    return fallback;
  }
}

function save(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Storage unavailable (private mode): the setting still applies for this visit.
  }
}

function describeCalibration(result: FitResult): string {
  const used = { both: 'both eyes', right: 'the right eye', left: 'the left eye', head: 'the head' }[result.best.choice];
  const across = Math.round(result.best.errorX * 100);
  const down = Math.round(result.best.errorY * 100);
  const moved =
    result.best.mode === 'eyes' && result.headMovement > 0.05
      ? ' Your head moved quite a bit; rest it (a hand under the chin helps) and recalibrate for better eye gaze.'
      : '';
  return `Calibrated with ${used}: typical error ±${across}% of the screen across, ±${down}% down.${moved}`;
}

const pct = (p: number) => `${Math.round(p * 100)}%`;

export function Grid() {
  const all = useMemo(() => cells(), []);
  const pages = useMemo(() => ['all', ...topicsOf(all)], [all]);
  const cellByKey = useMemo(() => new Map(all.map((c): [string, Cell] => [c.key, c])), [all]);
  const [lang, setLang] = useState<LabelLang>(() => load('las.lang', LANGS, 'lb'));
  const [gender, setGender] = useState<Gender>(() => load('las.gender', GENDERS, 'm'));
  const [page, setPage] = useState(() => load('las.page', pages, 'all'));
  const visible = useMemo(() => (page === 'all' ? all : all.filter((c) => c.topic === page)), [all, page]);
  const visibleKeys = useMemo(() => visible.map((c) => c.key), [visible]);
  const [message, setMessage] = useState<string[]>([]);
  const [showSettings, setShowSettings] = useState(() => load('las.settings', PANEL, 'open') === 'open');
  const hasArabic = useHasArabicVoice();
  const voiceLang = lang === 'en' ? 'en' : 'ar';
  const name = (key: string) => {
    const c = cellByKey.get(key);
    return c ? label(c, lang, gender) : key;
  };

  // One microphone for every mode: using the grid (when listening), teaching the app how the user says
  // words, or recording the app's own voice.
  const [mode, setMode] = useState<Mode>('use');
  const [listening, setListening] = useState(false);
  const micRef = useRef<MicHandle | null>(null);
  const micStarting = useRef<Promise<boolean> | null>(null);
  const detectorRef = useRef<UtteranceDetector | null>(null);
  const [micOn, setMicOn] = useState(false);
  const levels = useLevels(detectorRef, micOn);
  const [micError, setMicError] = useState<string | null>(null);
  const output = useRef(new OutputGate());

  // The app's voice: a recording for each label, played when the word is chosen.
  const [recordings, setRecordings] = useState<Map<string, Float32Array>>(() => new Map());
  const [armed, setArmed] = useState<string | null>(null);
  const armedRef = useRef<string | null>(null);

  // How the user says each word: their own recordings, and the recogniser built on them.
  const voice = useRef(new VoiceModel());
  const localIds = useRef(-1); // ids for recordings the browser could not store
  const [voiceVersion, setVoiceVersion] = useState(0);
  const teachRef = useRef<{ key: string; left: number } | null>(null);
  const [teaching, setTeaching] = useState<{ key: string; left: number } | null>(null);
  const [focusKey, setFocusKey] = useState<string | null>(null);
  const [selfTest, setSelfTest] = useState<SelfTest | null>(null);
  const [testProgress, setTestProgress] = useState<number | null>(null);

  // Choosing by voice: fusion with the pointer, "did you mean…?", and what was learned from each choice.
  const episodeRef = useRef<Episode | null>(null);
  const askTimer = useRef(0);
  const [candidates, setCandidates] = useState<string[]>([]);
  const [why, setWhy] = useState<Explanation | null>(null);
  const lastChoice = useRef<{ at: number; length: number; learned: number[] } | null>(null);

  // Pointing: the camera pointer (head or eyes) with dwell, or a rough touch that waits for the word.
  const trackerRef = useRef<GazeTracker | null>(null);
  const targets = useRef(new Map<string, HTMLElement>());
  const refCallbacks = useRef(new Map<string, (el: HTMLElement | null) => void>());
  const history = useRef(new PointerHistory());
  const [gazeOn, setGazeOn] = useState(false);
  const [calibrating, setCalibrating] = useState(false);
  const [pointerMode, setPointerMode] = useState<PointerMode>(() => load('las.pointerMode', POINTER_MODES, 'head'));
  const [gazeModel, setGazeModel] = useState<GazeModel | null>(() => loadModel(pointerMode));
  const [gazeStatus, setGazeStatus] = useState<string | null>(null);
  const [dwellMs, setDwellMs] = useState(() => Number(load('las.dwellMs', DWELL_CHOICES, '1200')));
  const [hoverKey, setHoverKey] = useState<string | null>(null);
  const [dwellKey, setDwellKey] = useState<string | null>(null);
  const dwellBlockedUntil = useRef(0);
  const [touchMode, setTouchMode] = useState<'exact' | 'rough'>(() => load('las.touch', TOUCH_MODES, 'exact'));
  const pendingTouch = useRef<PointerSample | null>(null);
  const touchTimer = useRef(0);
  const [touchMark, setTouchMark] = useState<{ x: number; y: number } | null>(null);

  const pageRef = useRef<HTMLDivElement>(null);
  const headRef = useRef<HTMLDivElement>(null);
  const gridRef = useRef<HTMLDivElement>(null);
  const layout = useFitGrid(gridRef, headRef, visible.length);
  const roughTouch = touchMode === 'rough' && listening && mode === 'use';

  // The detector and timers call the latest render's functions through this.
  const handlers = useRef({ utterance: (_audio: Float32Array) => {}, touchTimeout: () => {}, askTimeout: () => {} });

  useEffect(() => {
    void loadRecordings().then(setRecordings);
  }, []);

  const templatesLoaded = useRef(false);
  useEffect(() => {
    if (templatesLoaded.current) return;
    templatesLoaded.current = true;
    void loadTemplates().then((stored) => {
      for (const s of stored) {
        voice.current.add({ id: s.id, key: s.key, source: s.source ?? 'taught', feats: wordFeatures(s.audio), audio: s.audio });
      }
      setVoiceVersion((v) => v + 1);
    });
  }, []);

  // Self-test of the recordings whenever they change, in short slices so the pointer stays smooth.
  useEffect(() => {
    const model = voice.current;
    if (model.size === 0) {
      setSelfTest(null);
      setTestProgress(null);
      return;
    }
    let cancelled = false;
    let timer = 0;
    let shown = -1;
    const steps = model.selfTest();
    const run = () => {
      if (cancelled) return;
      const until = performance.now() + 12;
      let step = steps.next();
      while (!step.done && performance.now() < until) step = steps.next();
      if (step.done) {
        setSelfTest(step.value);
        setTestProgress(null);
        return;
      }
      const progress = Math.floor(step.value * 20) / 20;
      if (progress !== shown) {
        shown = progress;
        setTestProgress(progress);
      }
      timer = window.setTimeout(run, 0);
    };
    timer = window.setTimeout(run, 1500); // once the user pauses between recordings
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [voiceVersion]);

  useEffect(
    () => () => {
      micRef.current?.stop();
      trackerRef.current?.stop();
      window.clearTimeout(askTimer.current);
      window.clearTimeout(touchTimer.current);
    },
    [],
  );

  useEffect(() => {
    pageRef.current?.style.setProperty('--dwell', `${dwellMs || 1200}ms`);
  }, [dwellMs]);

  /** A stable ref callback per key, so the pointer knows where each target is. */
  function register(key: string) {
    let callback = refCallbacks.current.get(key);
    if (!callback) {
      callback = (el: HTMLElement | null) => {
        if (el) targets.current.set(key, el);
        else targets.current.delete(key);
      };
      refCallbacks.current.set(key, callback);
    }
    return callback;
  }

  function measure(keys: readonly string[]): Map<string, Rect> {
    const rects = new Map<string, Rect>();
    for (const key of keys) {
      const el = targets.current.get(key);
      if (el) rects.set(key, el.getBoundingClientRect());
    }
    return rects;
  }

  const arm = (text: string | null) => {
    armedRef.current = text;
    setArmed(text);
  };

  const setTeach = (t: { key: string; left: number } | null) => {
    teachRef.current = t;
    setTeaching(t);
  };

  // ---- Microphone and modes

  function ensureMic(): Promise<boolean> {
    unlockAudio();
    if (micRef.current) return Promise.resolve(true);
    let starting = micStarting.current;
    if (!starting) {
      setMicError(null);
      const detector = new UtteranceDetector((audio) => handlers.current.utterance(audio));
      starting = startMic((chunk) => detector.push(chunk))
        .then((mic) => {
          micRef.current = mic;
          detectorRef.current = detector;
          setMicOn(true);
          return true;
        })
        .catch((e: unknown) => {
          setMicError(`The microphone could not start: ${e instanceof Error ? e.message : String(e)}`);
          return false;
        })
        .finally(() => {
          micStarting.current = null;
        });
      micStarting.current = starting;
    }
    return starting;
  }

  function releaseMic() {
    micRef.current?.stop();
    micRef.current = null;
    detectorRef.current = null;
    setMicOn(false);
  }

  async function toggleListening() {
    if (listening) {
      setListening(false);
      endEpisode();
      cancelTouch();
      setWhy(null);
      if (mode === 'use') releaseMic();
      return;
    }
    if (await ensureMic()) setListening(true);
  }

  async function enterMode(next: 'teach' | 'record') {
    if (mode === next) {
      leaveMode();
      return;
    }
    if (!(await ensureMic())) return;
    endEpisode();
    cancelTouch();
    setTeach(null);
    arm(null);
    setMode(next);
  }

  function leaveMode() {
    setTeach(null);
    arm(null);
    setMode('use');
    if (!listening) releaseMic();
  }

  function handleUtterance(audio: Float32Array) {
    const now = performance.now();
    const speechStart = now - (audio.length / TARGET_RATE) * 1000 - SPEECH_OFFSET_MS;
    if (output.current.soundedSince(speechStart - ECHO_MARGIN_MS)) return; // the app's own voice, not the user
    if (mode === 'record') {
      const text = armedRef.current;
      if (!text) return;
      arm(null);
      setRecordings((r) => new Map(r).set(text, audio));
      void saveRecording(text, audio);
      void output.current.run(() => playAudio(audio));
    } else if (mode === 'teach') {
      const t = teachRef.current;
      if (!t) return;
      setTeach(t.left > 1 ? { key: t.key, left: t.left - 1 } : null);
      void addToVoice(t.key, audio, 'taught');
      void output.current.run(() => earcon('got'));
    } else if (listening) {
      chooseByVoice(audio, speechStart, now - DEFAULT_DETECTOR.hangoverMs);
    }
  }

  // ---- The user's recordings

  async function addToVoice(key: string, audio: Float32Array, source: TemplateSource): Promise<number> {
    const id = (await addTemplate(key, audio, source)) ?? localIds.current--;
    voice.current.add({ id, key, source, feats: wordFeatures(audio), audio });
    setVoiceVersion((v) => v + 1);
    return id;
  }

  /** Keeps how the user said a word they then confirmed; returns the recording's id, or null if not kept. */
  async function learn(key: string, audio: Float32Array): Promise<number | null> {
    const existing = voice.current.forKey(key);
    if (existing.length >= MAX_PER_WORD) {
      const oldest = existing.find((t) => t.source === 'learned');
      if (!oldest) return null; // only recordings taught on purpose: keep those
      forget(oldest.id);
    }
    return addToVoice(key, audio, 'learned');
  }

  function forget(id: number) {
    voice.current.remove(id);
    void deleteTemplate(id);
    setVoiceVersion((v) => v + 1);
  }

  function forgetWord(key: string) {
    for (const t of voice.current.removeKey(key)) void deleteTemplate(t.id);
    setVoiceVersion((v) => v + 1);
    if (teachRef.current?.key === key) setTeach(null);
  }

  function hearRecordings(key: string) {
    unlockAudio();
    const list = voice.current.forKey(key);
    void output.current.run(async () => {
      for (const t of list) await playAudio(t.audio);
    });
  }

  // ---- Choosing

  /** A spoken word: what it sounded like, fused with where the user pointed while saying it. */
  function chooseByVoice(audio: Float32Array, speechStart: number, speechEnd: number) {
    const now = performance.now();
    const keys = visibleKeys;
    const heard = voice.current.recognise(wordFeatures(audio), keys);
    // Where the user pointed: a rough touch waiting for this word, or the camera pointer while it was said.
    const touch = pendingTouch.current;
    if (touch) cancelTouch();
    const samples = touch ? [touch] : gazeOn ? history.current.between(speechStart - EYE_LEAD_MS, speechEnd) : [];
    const pointer = pointerEvidence(measure(keys), samples);
    const episode = liveEpisode(now);
    const channels: Channel[] = [];
    if (heard) channels.push({ dist: heard.dist, weight: 1 });
    if (pointer) channels.push({ dist: pointer, weight: 1 });
    if (episode) channels.push({ dist: episode.posterior, weight: PRIOR_WEIGHT });
    const fused = fuse(keys, channels);
    const decision: Decision = channels.length > 0 ? decide(fused) : { kind: 'none' };
    const tries = [...(episode?.tries ?? []), audio];

    // What to learn: after a question, every try was this word. A sure first choice is learned too while
    // the word has few recordings, unless the voice pointed at a different word it knows.
    let toLearn: Float32Array[] = [];
    if (decision.kind === 'select') {
      const voiceAgrees = heard === null || heard.best === decision.key || !heard.distances.has(decision.key);
      if (episode) toLearn = tries;
      else if (decision.probability >= LEARN_CONFIDENCE && voiceAgrees && voice.current.count(decision.key) < LEARN_UNTIL) {
        toLearn = [audio];
      }
    }
    setWhy({
      seconds: audio.length / TARGET_RATE,
      speech: heard ? ranked(heard.dist).slice(0, 3) : null,
      pointer: pointer ? ranked(pointer).slice(0, 3) : null,
      pointerKind: touch ? 'touch' : 'camera',
      again: episode !== null,
      learned: toLearn.length > 0,
      decision,
    });

    if (decision.kind === 'select') {
      endEpisode();
      void commit(decision.key, toLearn);
    } else if (decision.kind === 'ask') {
      startEpisode({ time: now, posterior: fused, tries }, decision.candidates.map(([key]) => key));
      void output.current.run(() => earcon('unsure'));
    } else {
      endEpisode();
      void output.current.run(() => earcon('missed'));
    }
  }

  /** Adds a word to the message and says it; `learnFrom` are tries at the word to learn from. */
  async function commit(key: string, learnFrom: Float32Array[]) {
    const c = cellByKey.get(key);
    if (!c) return;
    const text = label(c, lang, gender);
    setMessage((m) => [...m, text]);
    blockDwell();
    void speakWord(text);
    const record = { at: performance.now(), length: message.length + 1, learned: [] as number[] };
    lastChoice.current = record;
    for (const audio of learnFrom) {
      const id = await learn(key, audio);
      if (id !== null) record.learned.push(id);
    }
  }

  /** A tap, click or dwell on a cell. */
  function chooseCell(c: Cell) {
    if (mode === 'record') {
      arm(label(c, lang, gender));
      return;
    }
    if (mode === 'teach') {
      setTeach({ key: c.key, left: TEACH_COUNT });
      setFocusKey(c.key);
      return;
    }
    const episode = liveEpisode(performance.now());
    endEpisode();
    cancelTouch();
    void commit(c.key, episode ? episode.tries : []); // answering "did you mean…?" also teaches what was said
  }

  /** Rough touch: a touch says roughly where; the word said with it says which. */
  function onGridClick(e: MouseEvent<HTMLDivElement>) {
    if (!roughTouch || e.detail === 0) return; // keyboard presses stay exact
    unlockAudio();
    const sample: PointerSample = {
      t: performance.now(),
      x: e.clientX,
      y: e.clientY,
      sx: TOUCH_SPREAD * layout.cellWidth,
      sy: TOUCH_SPREAD * layout.rowHeight,
    };
    const keys = visibleKeys;
    const episode = liveEpisode(sample.t);
    if (episode) {
      // Answering "did you mean…?": the touch and what was heard decide together.
      const touch = pointerEvidence(measure(keys), [sample]);
      const fused = touch
        ? fuse(keys, [
            { dist: touch, weight: 1 },
            { dist: episode.posterior, weight: 1 },
          ])
        : episode.posterior;
      endEpisode();
      const best = ranked(fused)[0];
      if (best) void commit(best[0], episode.tries);
      return;
    }
    pendingTouch.current = sample;
    setTouchMark({ x: e.clientX, y: e.clientY });
    window.clearTimeout(touchTimer.current);
    touchTimer.current = window.setTimeout(() => handlers.current.touchTimeout(), TOUCH_WAIT_MS);
  }

  function touchTimedOut() {
    const touch = pendingTouch.current;
    if (!touch) return;
    if (detectorRef.current?.speaking && performance.now() - touch.t < TOUCH_GIVE_UP_MS) {
      touchTimer.current = window.setTimeout(() => handlers.current.touchTimeout(), 200); // a word is under way
      return;
    }
    cancelTouch();
    // No word came: the touch alone decides.
    const p = pointerEvidence(measure(visibleKeys), [touch]);
    const best = p ? ranked(p)[0] : undefined;
    if (best) void commit(best[0], []);
  }

  function cancelTouch() {
    window.clearTimeout(touchTimer.current);
    pendingTouch.current = null;
    setTouchMark(null);
  }

  function liveEpisode(now: number): Episode | null {
    const e = episodeRef.current;
    return e && now - e.time < ASK_MS ? e : null;
  }

  function startEpisode(e: Episode, keys: string[]) {
    episodeRef.current = e;
    setCandidates(keys);
    window.clearTimeout(askTimer.current);
    askTimer.current = window.setTimeout(() => handlers.current.askTimeout(), ASK_MS);
  }

  function endEpisode() {
    episodeRef.current = null;
    window.clearTimeout(askTimer.current);
    setCandidates([]);
  }

  function changePage(next: string) {
    if (!pages.includes(next) || next === page) return;
    endEpisode();
    cancelTouch();
    setPage(next);
    save('las.page', next);
  }

  // ---- The message and the app's voice

  function speakWord(text: string): Promise<void> {
    const recorded = recordings.get(text);
    return output.current.run(() => (recorded ? playAudio(recorded) : speakAsync(text, voiceLang)));
  }

  function sayMessage() {
    unlockAudio();
    const words = message;
    void output.current.run(async () => {
      for (const text of words) {
        const recorded = recordings.get(text);
        if (recorded) await playAudio(recorded);
        else await speakAsync(text, voiceLang);
      }
    });
  }

  function deleteLast() {
    const last = lastChoice.current;
    if (last && last.length === message.length && performance.now() - last.at < UNDO_MS) {
      for (const id of last.learned) forget(id); // a wrong choice must not teach the app
    }
    lastChoice.current = null;
    setMessage((m) => m.slice(0, -1));
  }

  function clearMessage() {
    lastChoice.current = null;
    setMessage([]);
  }

  // ---- Camera pointer

  async function startGaze() {
    setGazeStatus('Starting the camera and the face model…');
    const tracker = new GazeTracker();
    try {
      await tracker.start();
      trackerRef.current = tracker;
      setGazeOn(true);
      setGazeStatus(gazeModel ? 'Using your last calibration; recalibrate if the circle is off.' : null);
      if (!gazeModel) setCalibrating(true);
    } catch (e) {
      tracker.stop();
      setGazeStatus(`The camera pointer could not start: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  function stopGaze() {
    trackerRef.current?.stop();
    trackerRef.current = null;
    history.current.clear();
    setGazeOn(false);
    setCalibrating(false);
    setHoverKey(null);
    setDwellKey(null);
    setGazeStatus(null);
  }

  function changePointerMode(next: PointerMode) {
    setPointerMode(next);
    save('las.pointerMode', next);
    const model = loadModel(next);
    setGazeModel(model);
    setHoverKey(null);
    setDwellKey(null);
    setGazeStatus(model ? `Using your last ${next === 'eyes' ? 'eye-gaze' : 'head-pointer'} calibration.` : null);
    if (gazeOn && !model) setCalibrating(true);
  }

  function onCalibrated(result: FitResult | null, end: CalibrationEnd) {
    setCalibrating(false);
    if (result) {
      setGazeModel(result.best);
      saveModel(result.best);
      setGazeStatus(describeCalibration(result));
      return;
    }
    if (end === 'failed') {
      setGazeStatus('Calibration failed: the camera did not see your face clearly. Light your face and try again.');
    }
    if (!gazeModel) stopGaze();
  }

  function blockDwell() {
    dwellBlockedUntil.current = performance.now() + Math.max(dwellMs, 600);
  }

  function canDwell(): boolean {
    if (performance.now() < dwellBlockedUntil.current) return false;
    return !(mode === 'use' && listening && detectorRef.current?.speaking === true); // while speaking, the voice decides
  }

  function onDwell(key: string) {
    blockDwell();
    if (key.startsWith('tab:')) changePage(key.slice(4));
    else if (key === 'ctl:speak') sayMessage();
    else if (key === 'ctl:back') deleteLast();
    else if (key === 'ctl:clear') clearMessage();
    else {
      const c = cellByKey.get(key);
      if (c) chooseCell(c);
    }
  }

  handlers.current = { utterance: handleUtterance, touchTimeout: touchTimedOut, askTimeout: endEpisode };

  // ---- Display

  const counts = useMemo(
    () => new Map(all.map((c): [string, number] => [c.key, voice.current.count(c.key)])),
    [all, voiceVersion],
  );
  const taughtWords = [...counts.values()].filter((n) => n > 0).length;
  const missing = all.filter((c) => !recordings.has(label(c, lang, gender))).length;
  const labelSize = Math.round(Math.min(30, Math.max(14, layout.rowHeight * 0.13)));
  const pointed = (key: string, ...extra: (string | false)[]) =>
    [...extra, hoverKey === key && 'gaze-hover', dwellKey === key && 'dwelling'].filter(Boolean).join(' ');

  // Words the self-test mixes up are hard to tell apart by voice; if they are also neighbours on screen,
  // the pointer cannot separate them either.
  const position = new Map(visible.map((c, i): [string, [number, number]] => [c.key, [Math.floor(i / layout.cols), i % layout.cols]]));
  const neighbours = (a: string, b: string) => {
    const pa = position.get(a);
    const pb = position.get(b);
    return pa !== undefined && pb !== undefined && Math.abs(pa[0] - pb[0]) <= 1 && Math.abs(pa[1] - pb[1]) <= 1;
  };
  const selfTestLine =
    testProgress !== null
      ? `Checking the recordings… ${pct(testProgress)}`
      : selfTest && selfTest.tested > 0
        ? `Self-test: ${selfTest.correct} of ${selfTest.tested} recordings recognised from the others (${pct(selfTest.correct / selfTest.tested)}).`
        : taughtWords > 0
          ? 'Record each word at least twice for the self-test.'
          : '';
  const alike =
    testProgress === null && selfTest && selfTest.confusions.length > 0
      ? ` Sound alike to the app: ${selfTest.confusions
          .slice(0, 3)
          .map((c) => `${name(c.said)} → ${name(c.heard)}${neighbours(c.said, c.heard) ? ' (neighbours here: hard to tell apart)' : ''}`)
          .join(', ')}.`
      : '';

  function explain(e: Explanation) {
    const list = (items: [string, number][]) =>
      items
        .filter(([, p]) => p >= 0.01)
        .map(([k, p]) => (
          <span key={k} className="why-item">
            <bdi>{name(k)}</bdi> {pct(p)}
          </span>
        ));
    const d = e.decision;
    return (
      <>
        <span>
          Heard {e.seconds.toFixed(1)} s{e.again ? ' (again)' : ''}
        </span>
        {e.speech && <span className="why-part">voice: {list(e.speech)}</span>}
        {e.pointer && (
          <span className="why-part">
            {e.pointerKind}: {list(e.pointer)}
          </span>
        )}
        <span className="why-result">
          {d.kind === 'select' && (
            <>
              → <bdi>{name(d.key)}</bdi> {pct(d.probability)}
              {e.learned ? ' · learned' : ''}
            </>
          )}
          {d.kind === 'ask' && (
            <>
              → did you mean{' '}
              {d.candidates.map(([k], i) => (
                <span key={k}>
                  {i > 0 ? ' or ' : ''}
                  <bdi>{name(k)}</bdi>
                </span>
              ))}
              ? Point, tap or say it again.
            </>
          )}
          {d.kind === 'none' && <>→ not understood{e.speech || e.pointer ? '' : ': no taught words on this page and no pointer'}</>}
        </span>
      </>
    );
  }

  return (
    <div ref={pageRef} className="grid-page" dir={lang === 'en' ? 'ltr' : 'rtl'}>
      <div ref={headRef} className="grid-head">
        <div className="message-bar">
          <div className="message" aria-live="polite">
            {message.join(' ')}
          </div>
          <button ref={register('ctl:speak')} type="button" className={pointed('ctl:speak')} onClick={sayMessage} aria-label="Speak the message">
            ▶
          </button>
          <button ref={register('ctl:back')} type="button" className={pointed('ctl:back')} onClick={deleteLast} aria-label="Delete the last word">
            ⌫
          </button>
          <button ref={register('ctl:clear')} type="button" className={pointed('ctl:clear')} onClick={clearMessage} aria-label="Clear the message">
            ✕
          </button>
        </div>

        <div className="controls" dir="ltr">
          {gazeOn ? (
            <>
              <button type="button" className="on" onClick={stopGaze}>
                Stop pointer
              </button>
              <button type="button" onClick={() => setCalibrating(true)}>
                Recalibrate
              </button>
            </>
          ) : (
            <button type="button" onClick={() => void startGaze()}>
              Start pointer (camera)
            </button>
          )}
          <button type="button" className={listening ? 'on' : undefined} onClick={() => void toggleListening()}>
            {listening ? 'Stop listening' : 'Listen: choose by voice'}
          </button>
          <button type="button" className={mode === 'teach' ? 'on' : undefined} onClick={() => void enterMode('teach')}>
            {mode === 'teach' ? 'Done teaching' : 'Teach my words'}
          </button>
          <button
            type="button"
            aria-expanded={showSettings}
            onClick={() => {
              save('las.settings', showSettings ? 'closed' : 'open');
              setShowSettings(!showSettings);
            }}
          >
            Settings {showSettings ? '▴' : '▾'}
          </button>
        </div>

        {showSettings && (
          <div className="settings" dir="ltr">
            <label>
              Labels{' '}
              <select
                value={lang}
                onChange={(e) => {
                  const v = e.target.value as LabelLang;
                  setLang(v);
                  save('las.lang', v);
                }}
              >
                <option value="lb">لبناني (Lebanese)</option>
                <option value="msa">فصحى (MSA)</option>
                <option value="en">English</option>
              </select>
            </label>
            <label>
              User{' '}
              <select
                value={gender}
                onChange={(e) => {
                  const v = e.target.value as Gender;
                  setGender(v);
                  save('las.gender', v);
                }}
              >
                <option value="m">male</option>
                <option value="f">female</option>
              </select>
            </label>
            <label>
              Pointer{' '}
              <select value={pointerMode} onChange={(e) => changePointerMode(e.target.value as PointerMode)}>
                <option value="head">head</option>
                <option value="eyes">eyes</option>
              </select>
            </label>
            <label>
              Select by pointing{' '}
              <select
                value={String(dwellMs)}
                onChange={(e) => {
                  setDwellMs(Number(e.target.value));
                  save('las.dwellMs', e.target.value);
                }}
              >
                {DWELL_CHOICES.map((ms) => (
                  <option key={ms} value={ms}>
                    {ms === '0' ? 'off' : `after ${Number(ms) / 1000} s`}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Touch{' '}
              <select
                value={touchMode}
                onChange={(e) => {
                  const v = e.target.value as 'exact' | 'rough';
                  setTouchMode(v);
                  save('las.touch', v);
                }}
              >
                <option value="exact">exact</option>
                <option value="rough">rough: touch near it, then say it</option>
              </select>
            </label>
            <button type="button" className={mode === 'record' ? 'on' : undefined} onClick={() => void enterMode('record')}>
              {mode === 'record' ? 'Done recording' : "Record the app's voice"}
            </button>
          </div>
        )}

        <div className="status" dir="ltr">
          {micError && <div className="error">{micError}</div>}
          {gazeStatus && <div className="hint">{gazeStatus}</div>}
          {mode === 'teach' && (
            <div className="teach">
              <div className="hint">
                {teaching
                  ? `Say «${name(teaching.key)}» now (${TEACH_COUNT - teaching.left + 1} of ${TEACH_COUNT})…`
                  : `Tap a picture, then say it ${TEACH_COUNT} times with a short pause between. The user records their own way of saying it, however unclear.`}
              </div>
              {focusKey && (
                <div>
                  <bdi>«{name(focusKey)}»</bdi>: {counts.get(focusKey) ?? 0} recordings{' '}
                  <button type="button" onClick={() => hearRecordings(focusKey)}>
                    Hear them
                  </button>{' '}
                  <button type="button" onClick={() => forgetWord(focusKey)}>
                    Forget them
                  </button>
                </div>
              )}
              <div className="muted">
                {taughtWords} words taught. {selfTestLine}
                {alike}
              </div>
            </div>
          )}
          {mode === 'record' && (
            <div className="hint">
              {armed
                ? `Say «${armed}» now…`
                : `Tap a word, then say it clearly: the app will speak with this recording. ${missing} still to record.`}
            </div>
          )}
          {mode === 'use' && listening && (
            <div className="why">
              {why
                ? explain(why)
                : roughTouch
                  ? 'Listening: touch near a picture, then say it.'
                  : gazeOn
                    ? 'Listening: point at a picture and say it.'
                    : 'Listening: say a word.'}
            </div>
          )}
          {mode === 'use' && hasArabic === false && lang !== 'en' && missing > 0 && (
            <div className="warn">This device has no Arabic voice: use «Record the app's voice» in Settings so the words can be spoken.</div>
          )}
        </div>

        {micOn && (
          <div dir="ltr">
            <LevelMeter {...levels} />
          </div>
        )}

        <div className="pages">
          {pages.map((p) => (
            <button
              key={p}
              ref={register(`tab:${p}`)}
              type="button"
              className={pointed(`tab:${p}`, 'page-tab', p === page && 'active')}
              onClick={() => changePage(p)}
            >
              {pageName(p, lang)}
            </button>
          ))}
        </div>
      </div>

      <div
        ref={gridRef}
        className={roughTouch ? 'cells rough' : 'cells'}
        style={{ gridTemplateColumns: `repeat(${layout.cols}, minmax(0, 1fr))`, gridAutoRows: `${layout.rowHeight}px` }}
        onClick={onGridClick}
      >
        {visible.map((c) => {
          const text = label(c, lang, gender);
          const taught = counts.get(c.key) ?? 0;
          const classes = pointed(
            c.key,
            'cell',
            (armed === text || teaching?.key === c.key) && 'armed',
            candidates.includes(c.key) && 'candidate',
          );
          return (
            <button
              key={c.key}
              ref={register(c.key)}
              type="button"
              className={classes}
              style={{ borderColor: classColour(c.cls) }}
              onClick={(e) => {
                if (roughTouch && e.detail !== 0) return; // the grid handles rough touches
                chooseCell(c);
              }}
            >
              {recordings.has(text) && <span className="rec-dot" title="The app's voice for this word is recorded" />}
              {taught > 0 && (
                <span className="taught" title="Recordings of how the user says it">
                  🎤{taught}
                </span>
              )}
              <img src={pictogramUrl(c.arasaac)} alt="" loading="lazy" draggable={false} />
              <span className="cell-label" style={{ fontSize: labelSize }}>
                {text}
              </span>
            </button>
          );
        })}
      </div>

      <footer className="attribution" dir="ltr">
        {ATTRIBUTION}
      </footer>

      {touchMark && <div className="touch-mark" style={{ left: touchMark.x, top: touchMark.y }} />}
      {gazeOn && calibrating && trackerRef.current && (
        <Calibration tracker={trackerRef.current} mode={pointerMode} onDone={onCalibrated} />
      )}
      {gazeOn && !calibrating && gazeModel && trackerRef.current && (
        <GazeOverlay
          tracker={trackerRef.current}
          model={gazeModel}
          targets={targets}
          layoutKey={`${page}|${layout.cols}|${layout.rowHeight}|${layout.top}`}
          dwellMs={dwellMs > 0 ? dwellMs : null}
          canDwell={canDwell}
          onHover={(key, dwelling) => {
            setHoverKey(key);
            setDwellKey(dwelling ? key : null);
          }}
          onDwell={onDwell}
          onFrame={(x, y, sx, sy, t) => history.current.push({ t, x, y, sx: sx * GAZE_SPREAD, sy: sy * GAZE_SPREAD })}
        />
      )}
    </div>
  );
}
