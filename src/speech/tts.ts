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

export function speak(text: string, lang: 'ar' | 'en'): void {
  if (!text || !('speechSynthesis' in window)) return;
  const utterance = new SpeechSynthesisUtterance(text);
  const voice = pickVoice(window.speechSynthesis.getVoices(), lang);
  if (voice) {
    utterance.voice = voice;
    utterance.lang = voice.lang;
  } else {
    utterance.lang = lang === 'ar' ? 'ar' : 'en-US';
  }
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
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
