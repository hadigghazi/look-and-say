import { useEffect, useState, type MutableRefObject } from 'react';
import type { UtteranceDetector } from '../audio/capture';

export interface Levels {
  level: number;
  noise: number;
  threshold: number;
}

const toDb = (x: number) => 20 * Math.log10(Math.max(x, 1e-5));
// The bar runs from -70 dB (near silence) to 0 dB (the loudest the microphone can give).
const position = (x: number) => Math.min(100, Math.max(0, ((toDb(x) + 70) / 70) * 100));

/** Live loudness with two marks: the background noise, and the line speech has to cross to count. */
export function LevelMeter({ level, noise, threshold }: Levels) {
  return (
    <div className="level">
      <div className="meter" aria-label="Microphone level">
        <div className={level > threshold ? 'meter-fill speaking' : 'meter-fill'} style={{ width: `${position(level)}%` }} />
        <div className="meter-mark noise" style={{ left: `${position(noise)}%` }} title="Background noise" />
        <div className="meter-mark threshold" style={{ left: `${position(threshold)}%` }} title="Speech counts above this" />
      </div>
      <div className="meter-legend">
        Level {toDb(level).toFixed(0)} dB · background {toDb(noise).toFixed(0)} dB · speech counts above{' '}
        {toDb(threshold).toFixed(0)} dB
      </div>
    </div>
  );
}

/** Reads a detector's levels ten times a second while the microphone runs. */
export function useLevels(detector: MutableRefObject<UtteranceDetector | null>, running: boolean): Levels {
  const [levels, setLevels] = useState<Levels>({ level: 0, noise: 0, threshold: 0 });
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => {
      const d = detector.current;
      if (d) setLevels({ level: d.level, noise: d.noise, threshold: d.startThreshold });
    }, 100);
    return () => window.clearInterval(timer);
  }, [detector, running]);
  return levels;
}
