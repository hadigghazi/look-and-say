import { useEffect, useMemo, useRef, useState } from 'react';
import { ATTRIBUTION, cells, classColour, label, pictogramUrl, type Gender, type LabelLang } from '../vocab';
import { speak, speakAsync, useHasArabicVoice } from '../speech/tts';
import { playAudio, startMic, UtteranceDetector, type MicHandle } from '../audio/capture';
import { loadRecordings, saveRecording } from '../storage/voiceStore';

const LANGS = ['lb', 'msa', 'en'] as const;
const GENDERS = ['m', 'f'] as const;

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

export function Grid() {
  const all = useMemo(() => cells(), []);
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
  const [micError, setMicError] = useState<string | null>(null);

  useEffect(() => {
    void loadRecordings().then(setRecordings);
  }, []);
  useEffect(() => () => micRef.current?.stop(), []);

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
        {!recordMode && hasArabic === false && lang !== 'en' && missing > 0 && (
          <span className="warn">This device has no Arabic voice: use «Record the voice» so the words can be spoken.</span>
        )}
      </div>

      <div className="cells">
        {all.map((c) => {
          const text = label(c, lang, gender);
          const classes = ['cell', armed === text ? 'armed' : ''].join(' ').trim();
          return (
            <button key={c.key} type="button" className={classes} style={{ borderColor: classColour(c.cls) }} onClick={() => choose(text)}>
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
    </div>
  );
}
