import { useEffect, useState } from 'react';

// Placeholder output: the browser's built-in voices. A voice the user chooses
// (synthetic Lebanese, consented recordings or an own-voice clone) replaces this later.

function normalisedLang(voice: SpeechSynthesisVoice): string {
  return voice.lang.toLowerCase().replace('_', '-');
}

function pickVoice(voices: SpeechSynthesisVoice[], lang: 'ar' | 'en'): SpeechSynthesisVoice | undefined {
  if (lang === 'ar') {
    return voices.find((v) => normalisedLang(v).startsWith('ar-lb')) ?? voices.find((v) => normalisedLang(v).startsWith('ar'));
  }
  return voices.find((v) => normalisedLang(v).startsWith('en'));
}

function utteranceFor(text: string, lang: 'ar' | 'en'): SpeechSynthesisUtterance {
  const utterance = new SpeechSynthesisUtterance(text);
  const voice = pickVoice(window.speechSynthesis.getVoices(), lang);
  if (voice) {
    utterance.voice = voice;
    utterance.lang = voice.lang;
  } else {
    utterance.lang = lang === 'ar' ? 'ar' : 'en-US';
  }
  return utterance;
}

export function speak(text: string, lang: 'ar' | 'en'): void {
  if (!text || !('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utteranceFor(text, lang));
}

/** Like speak, but resolves when the word has been said (or after 3 s, as some browsers never report it). */
export function speakAsync(text: string, lang: 'ar' | 'en'): Promise<void> {
  return new Promise((resolve) => {
    if (!text || !('speechSynthesis' in window)) {
      resolve();
      return;
    }
    const utterance = utteranceFor(text, lang);
    utterance.onend = () => resolve();
    utterance.onerror = () => resolve();
    window.speechSynthesis.speak(utterance);
    window.setTimeout(resolve, 3000);
  });
}

/** Whether the device offers an Arabic voice (null while the voice list is still loading). */
export function useHasArabicVoice(): boolean | null {
  const [has, setHas] = useState<boolean | null>(null);
  useEffect(() => {
    if (!('speechSynthesis' in window)) {
      setHas(false);
      return;
    }
    const check = () => {
      const voices = window.speechSynthesis.getVoices();
      if (voices.length > 0) setHas(pickVoice(voices, 'ar') !== undefined);
    };
    check();
    window.speechSynthesis.addEventListener('voiceschanged', check);
    return () => window.speechSynthesis.removeEventListener('voiceschanged', check);
  }, []);
  return has;
}
