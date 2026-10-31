import { describe, expect, it } from 'vitest';
import { concat, resample, TARGET_RATE, trimSilence, UtteranceDetector } from './capture';
import { dtw } from './dtw';
import { fft, mfcc } from './mfcc';

const seq = (rows: number[][]) => rows.map((r) => Float64Array.from(r));
const samples = (seconds: number) => Math.round(TARGET_RATE * seconds);

/** Deterministic uniform noise with the given RMS, so tests always see the same signal. */
function noise(n: number, rms: number, seed = 1): Float32Array {
  const out = new Float32Array(n);
  let s = seed;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = ((s / 2 ** 32) * 2 - 1) * rms * Math.sqrt(3);
  }
  return out;
}

/** A 300 Hz tone, optionally shaped by an envelope (to imitate syllables). */
function tone(n: number, amplitude: number, envelope: (i: number) => number = () => 1): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = amplitude * envelope(i) * Math.sin((2 * Math.PI * 300 * i) / TARGET_RATE);
  return out;
}

describe('fft', () => {
  it('puts a pure tone in its frequency bin', () => {
    const n = 64;
    const bin = 8;
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    for (let i = 0; i < n; i++) re[i] = Math.cos((2 * Math.PI * bin * i) / n);
    fft(re, im);
    const magnitude = Array.from(re, (r, k) => Math.hypot(r, im[k]));
    const half = magnitude.slice(0, n / 2);
    expect(half.indexOf(Math.max(...half))).toBe(bin);
    expect(magnitude[bin]).toBeCloseTo(n / 2, 6);
  });
});

describe('mfcc', () => {
  it('gives 13 mean-normalised coefficients per 10 ms frame', () => {
    const signal = new Float32Array(TARGET_RATE); // 1 s
    for (let i = 0; i < signal.length; i++) {
      signal[i] = 0.3 * Math.sin((2 * Math.PI * 440 * i) / TARGET_RATE) + 0.01 * Math.sin(i * 12.9898);
    }
    const frames = mfcc(signal);
    expect(frames.length).toBe(Math.floor((TARGET_RATE - 400) / 160) + 1);
    expect(frames[0].length).toBe(13);
    for (let k = 0; k < 13; k++) {
      const mean = frames.reduce((s, f) => s + f[k], 0) / frames.length;
      expect(Math.abs(mean)).toBeLessThan(1e-9);
    }
  });
});

describe('dtw', () => {
  const a = seq([[0, 0], [1, 1], [2, 2], [3, 3]]);

  it('is zero for identical sequences', () => {
    expect(dtw(a, a)).toBe(0);
  });

  it('matches a slowed-down copy exactly, and better than a different sequence', () => {
    const slow = seq([[0, 0], [0, 0], [1, 1], [1, 1], [2, 2], [2, 2], [3, 3], [3, 3]]);
    const other = seq([[3, 0], [2, 1], [1, 2], [0, 3]]);
    expect(dtw(a, slow)).toBe(0);
    expect(dtw(a, slow)).toBeLessThan(dtw(a, other));
  });

  it('is symmetric', () => {
    const b = seq([[0, 1], [2, 2], [3, 1]]);
    expect(dtw(a, b)).toBeCloseTo(dtw(b, a), 12);
  });
});

describe('trimSilence', () => {
  it('keeps the word and a short margin, not the quiet around it', () => {
    const audio = concat([noise(samples(0.3), 0.001), tone(samples(0.4), 0.3), noise(samples(0.3), 0.001, 2)]);
    const seconds = trimSilence(audio).length / TARGET_RATE;
    expect(seconds).toBeGreaterThan(0.4);
    expect(seconds).toBeLessThan(0.5);
  });
});

describe('resample', () => {
  it('keeps the duration when going from 48 kHz to 16 kHz', () => {
    expect(resample(new Float32Array(48000), 48000, 16000).length).toBe(16000);
  });
});

describe('UtteranceDetector', () => {
  it('finds one word between two silences', () => {
    const found: Float32Array[] = [];
    const detector = new UtteranceDetector((audio) => found.push(audio));
    detector.push(new Float32Array(samples(0.5)));
    detector.push(tone(samples(0.4), 0.3));
    detector.push(new Float32Array(samples(0.6)));
    expect(found.length).toBe(1);
    const seconds = found[0].length / TARGET_RATE;
    expect(seconds).toBeGreaterThan(0.35);
    expect(seconds).toBeLessThan(0.8);
  });

  it('finds a quiet word with dips between syllables over background noise', () => {
    const found: Float32Array[] = [];
    const detector = new UtteranceDetector((audio) => found.push(audio));
    const syllables = (i: number) => 0.3 + 0.7 * Math.abs(Math.sin((4 * Math.PI * i) / TARGET_RATE));
    const word = tone(samples(0.5), 0.014, syllables);
    const backgroundUnderWord = noise(word.length, 0.002, 3);
    detector.push(noise(samples(0.8), 0.002, 1));
    detector.push(word.map((v, i) => v + backgroundUnderWord[i]));
    detector.push(noise(samples(0.8), 0.002, 5));
    expect(found.length).toBe(1);
  });

  it('ignores a short click', () => {
    const found: Float32Array[] = [];
    const detector = new UtteranceDetector((audio) => found.push(audio));
    detector.push(noise(samples(0.8), 0.002, 7));
    detector.push(tone(samples(0.02), 0.5));
    detector.push(noise(samples(0.8), 0.002, 9));
    expect(found.length).toBe(0);
  });
});
