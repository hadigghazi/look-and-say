import { useEffect, useMemo, useRef, useState } from 'react';
import { ATTRIBUTION, cells, classColour, label, pictogramUrl, type Cell, type Gender, type LabelLang } from '../vocab';
import { speak, speakAsync, useHasArabicVoice } from '../speech/tts';
import { playAudio, startMic, UtteranceDetector, type MicHandle } from '../audio/capture';
import { loadRecordings, saveRecording } from '../storage/voiceStore';
import { LevelMeter, useLevels } from '../components/LevelMeter';
import { GazeTracker } from '../gaze/tracker';
import { Calibration, type CalibrationEnd } from '../gaze/Calibration';
import { GazeOverlay } from '../gaze/GazeOverlay';
import { loadModel, saveModel, type FitResult, type GazeModel } from '../gaze/model';

const LANGS = ['lb', 'msa', 'en'] as const;
const GENDERS = ['m', 'f'] as const;
const DWELL_MS = 1200;

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
  const eye = { both: 'both eyes', right: 'the right eye', left: 'the left eye' }[result.best.choice];
  const across = Math.round(result.best.errorX * 100);
  const down = Math.round(result.best.errorY * 100);
  return `Calibrated with ${eye}: typical error ±${across}% of the screen across, ±${down}% down.`;
}

export function Grid() {
  const all = useMemo(() => cells(), []);
  const cellByKey = useMemo(() => new Map(all.map((c): [string, Cell] => [c.key, c])), [all]);
  const [lang, setLang] = useState<LabelLang>(() => load('las.lang', LANGS, 'lb'));
  const [gender, setGender] = useState<Gender>(() => load('las.gender', GENDERS, 'm'));
  const [message, setMessage] = useState<string[]>([]);
  const hasArabic = useHasArabicVoice();
  const voiceLang = lang === 'en' ? 'en' : 'ar';

  // Recorded voice: tap a word in recording mode, say it, and that recording is played from then on.
  const [recordings, setRecordings] = useState<Map<string, Float32Array>>(() => new Map());
  const [recordMode, setRecordMode] = useState(false);
  const [armed, setArmed] = useState<string | null>(null);
  const armedRef = useRef<string | null>(null);
  const micRef = useRef<MicHandle | null>(null);
  const detectorRef = useRef<UtteranceDetector | null>(null);
  const levels = useLevels(detectorRef, recordMode);
  const [micError, setMicError] = useState<string | null>(null);

  // Eye gaze: calibrate, then a cursor that knows which cell is looked at; looking long enough selects it.
  const trackerRef = useRef<GazeTracker | null>(null);
  const cellEls = useRef(new Map<string, HTMLElement>());
  const [gazeOn, setGazeOn] = useState(false);
  const [calibrating, setCalibrating] = useState(false);
  const [gazeModel, setGazeModel] = useState<GazeModel | null>(() => loadModel());
  const [gazeStatus, setGazeStatus] = useState<string | null>(null);
  const [dwell, setDwell] = useState(true);
  const [hoverKey, setHoverKey] = useState<string | null>(null);

  useEffect(() => {
    void loadRecordings().then(setRecordings);
  }, []);
  useEffect(
    () => () => {
      micRef.current?.stop();
      trackerRef.current?.stop();
    },
    [],
  );

  const arm = (text: string | null) => {
    armedRef.current = text;
    setArmed(text);
  };

  async function startRecording() {
    setMicError(null);
    const detector = new UtteranceDetector((audio) => {
      const text = armedRef.current;
      if (!text) return; // ignore speech (and our own playback) unless a word is waiting
      arm(null);
      setRecordings((r) => new Map(r).set(text, audio));
      void saveRecording(text, audio);
      void playAudio(audio);
    });
    detectorRef.current = detector;
    try {
      micRef.current = await startMic((chunk) => detector.push(chunk));
      setRecordMode(true);
    } catch (e) {
      setMicError(e instanceof Error ? e.message : String(e));
    }
  }

  function stopRecording() {
    micRef.current?.stop();
    micRef.current = null;
    arm(null);
    setRecordMode(false);
  }

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
      setGazeStatus(`Eye gaze could not start: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  function stopGaze() {
    trackerRef.current?.stop();
    trackerRef.current = null;
    setGazeOn(false);
    setCalibrating(false);
    setHoverKey(null);
    setGazeStatus(null);
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
      setGazeStatus('Calibration failed: the camera did not see your eyes clearly. Light your face and try again.');
    }
    if (!gazeModel) stopGaze();
  }

  const say = (text: string) => {
    const recorded = recordings.get(text);
    if (recorded) void playAudio(recorded);
    else speak(text, voiceLang);
  };

  async function sayMessage() {
    for (const text of message) {
      const recorded = recordings.get(text);
      if (recorded) await playAudio(recorded);
      else await speakAsync(text, voiceLang);
    }
  }

  const choose = (text: string) => {
    if (recordMode) {
      arm(text);
      return;
    }
    setMessage((m) => [...m, text]);
    say(text);
  };

  const missing = all.filter((c) => !recordings.has(label(c, lang, gender))).length;

  return (
    <div className="grid-page" dir={lang === 'en' ? 'ltr' : 'rtl'}>
      <div className="message-bar">
        <div className="message" aria-live="polite">
          {message.join(' ')}
        </div>
        <button type="button" onClick={() => void sayMessage()} aria-label="Speak the message">
          ▶
        </button>
        <button type="button" onClick={() => setMessage((m) => m.slice(0, -1))} aria-label="Delete the last word">
          ⌫
        </button>
        <button type="button" onClick={() => setMessage([])} aria-label="Clear the message">
          ✕
        </button>
      </div>

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

        {gazeOn ? (
          <>
            <button type="button" className="record-on" onClick={stopGaze}>
              Stop eye gaze
            </button>
            <button type="button" onClick={() => setCalibrating(true)}>
              Recalibrate
            </button>
            <label>
              <input type="checkbox" checked={dwell} onChange={(e) => setDwell(e.target.checked)} /> select by looking (
              {DWELL_MS / 1000} s)
            </label>
          </>
        ) : (
          <button type="button" onClick={() => void startGaze()}>
            Use eye gaze
          </button>
        )}

        {recordMode ? (
          <button type="button" className="record-on" onClick={stopRecording}>
            Done recording
          </button>
        ) : (
          <button type="button" onClick={() => void startRecording()}>
            Record the voice
          </button>
        )}
        {recordMode && (
          <span className="hint">{armed ? `Say «${armed}» now…` : `Tap a word, then say it. ${missing} still to record.`}</span>
        )}
        {micError && <span className="error">{micError}</span>}
        {gazeStatus && <span className="hint">{gazeStatus}</span>}
        {!recordMode && hasArabic === false && lang !== 'en' && missing > 0 && (
          <span className="warn">This device has no Arabic voice: use «Record the voice» so the words can be spoken.</span>
        )}
      </div>

      {recordMode && (
        <div dir="ltr">
          <LevelMeter {...levels} />
        </div>
      )}

      <div className="cells">
        {all.map((c) => {
          const text = label(c, lang, gender);
          const looked = hoverKey === c.key;
          const classes = ['cell', armed === text && 'armed', looked && 'gaze-hover', looked && dwell && 'dwelling']
            .filter(Boolean)
            .join(' ');
          return (
            <button
              key={c.key}
              ref={(el) => {
                if (el) cellEls.current.set(c.key, el);
                else cellEls.current.delete(c.key);
              }}
              type="button"
              className={classes}
              style={{ borderColor: classColour(c.cls) }}
              onClick={() => choose(text)}
            >
              {recordings.has(text) && <span className="rec-dot" title="Recorded" />}
              <img src={pictogramUrl(c.arasaac)} alt="" loading="lazy" draggable={false} />
              <span className="cell-label">{text}</span>
            </button>
          );
        })}
      </div>

      <footer className="attribution" dir="ltr">
        {ATTRIBUTION}
      </footer>

      {gazeOn && calibrating && trackerRef.current && <Calibration tracker={trackerRef.current} onDone={onCalibrated} />}
      {gazeOn && !calibrating && gazeModel && trackerRef.current && (
        <GazeOverlay
          tracker={trackerRef.current}
          model={gazeModel}
          cells={cellEls}
          dwellMs={dwell ? DWELL_MS : null}
          onHover={setHoverKey}
          onDwell={(key) => {
            const c = cellByKey.get(key);
            if (c) choose(label(c, lang, gender));
          }}
        />
      )}
    </div>
  );
}
