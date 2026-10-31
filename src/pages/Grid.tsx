import { useMemo, useState } from 'react';
import { ATTRIBUTION, cells, classColour, label, pictogramUrl, type Gender, type LabelLang } from '../vocab';
import { speak, useHasArabicVoice } from '../speech/tts';

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

  const choose = (text: string) => {
    setMessage((m) => [...m, text]);
    speak(text, voiceLang);
  };

  return (
    <div className="grid-page" dir={lang === 'en' ? 'ltr' : 'rtl'}>
      <div className="message-bar">
        <div className="message" aria-live="polite">
          {message.join(' ')}
        </div>
        <button type="button" onClick={() => speak(message.join(' '), voiceLang)} aria-label="Speak the message">
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
        {hasArabic === false && lang !== 'en' && (
          <span className="warn">This device has no Arabic voice, so words may be read wrongly.</span>
        )}
      </div>

      <div className="cells">
        {all.map((c) => {
          const text = label(c, lang, gender);
          return (
            <button key={c.key} type="button" className="cell" style={{ borderColor: classColour(c.cls) }} onClick={() => choose(text)}>
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
