import { describe, expect, it } from 'vitest';
import { resample, TARGET_RATE, UtteranceDetector } from './capture';
import { dtw } from './dtw';
import { fft, mfcc } from './mfcc';
import { classify } from './templates';

const seq = (rows: number[][]) => rows.map((r) => Float64Array.from(r));

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

describe('classify', () => {
  it('ranks the word whose template matches first, with probabilities summing to 1', () => {
    const water = seq([[0, 0], [1, 1], [2, 2]]);
    const juice = seq([[5, 5], [4, 6], [3, 7]]);
    const ranking = classify(seq([[0, 0], [1, 1], [1, 1], [2, 2]]), [
      { label: 'water', feats: water },
      { label: 'juice', feats: juice },
    ]);
    expect(ranking[0].word).toBe('water');
    expect(ranking.reduce((s, r) => s + r.probability, 0)).toBeCloseTo(1, 12);
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
    const silence = new Float32Array(TARGET_RATE / 2);
    const word = new Float32Array((TARGET_RATE * 2) / 5);
    for (let i = 0; i < word.length; i++) word[i] = 0.3 * Math.sin((2 * Math.PI * 300 * i) / TARGET_RATE);
    detector.push(silence);
    detector.push(word);
    detector.push(silence);
    expect(found.length).toBe(1);
    const seconds = found[0].length / TARGET_RATE;
    expect(seconds).toBeGreaterThan(0.35);
    expect(seconds).toBeLessThan(0.7);
  });
});
