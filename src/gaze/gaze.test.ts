import { describe, expect, it } from 'vitest';
import type { FaceFeatures } from './features';
import { fitGaze, predict, type Sample } from './model';
import { OneEuro } from './oneEuro';
import { ridge, solve } from './ridge';
import { mostLikely, pointerProbabilities, type Rect } from '../fusion/pointer';

const DOTS = [
  [0.1, 0.1], [0.5, 0.1], [0.9, 0.1],
  [0.7, 0.3], [0.3, 0.3],
  [0.1, 0.5], [0.5, 0.5], [0.9, 0.5],
  [0.7, 0.7], [0.3, 0.7],
  [0.1, 0.9], [0.5, 0.9], [0.9, 0.9],
];

function random(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32 - 0.5;
  };
}

/** Calibration samples from a made-up face, where `make` describes the features for a dot. */
function calibrate(make: (x: number, y: number, rand: () => number) => FaceFeatures, frames = 20): Sample[] {
  const rand = random(1);
  const samples: Sample[] = [];
  DOTS.forEach(([x, y], point) => {
    for (let k = 0; k < frames; k++) samples.push({ features: make(x, y, rand), target: { x, y }, point });
  });
  return samples;
}

const stillHead = (rand: () => number) => ({
  fx: 0.001 * rand(),
  fy: 0.001 * rand(),
  tx: 0.001 * rand(),
  ty: 0.001 * rand(),
  tz: 50 + 0.01 * rand(),
});

describe('solve', () => {
  it('solves a small linear system', () => {
    const x = solve([[2, 1, 0], [1, 3, 1], [0, 1, 4]], [4, 10, 14]);
    [1, 2, 3].forEach((v, i) => expect(x[i]).toBeCloseTo(v, 10));
  });
});

describe('ridge', () => {
  it('recovers a linear map when the penalty is tiny', () => {
    const Z: number[][] = [];
    const y: number[] = [];
    for (let a = 0; a < 5; a++) {
      for (let b = 0; b < 5; b++) {
        Z.push([1, a, b]);
        y.push(2 + 3 * a - b);
      }
    }
    const w = ridge(Z, y, 1e-9);
    [2, 3, -1].forEach((v, i) => expect(w[i]).toBeCloseTo(v, 6));
  });

  it('gives the weight to the lightly penalised input when two inputs are identical', () => {
    const Z: number[][] = [];
    const y: number[] = [];
    for (let a = 0; a < 10; a++) {
      Z.push([1, a, a]);
      y.push(a);
    }
    const w = ridge(Z, y, [1e-6, 1e3]);
    expect(w[1]).toBeCloseTo(1, 1);
    expect(Math.abs(w[2])).toBeLessThan(0.05);
  });
});

describe('fitGaze', () => {
  it('picks the eye that tracks when the other wanders, as with strabismus', () => {
    const samples = calibrate((x, y, rand) => ({
      right: { dx: 0.4 * (x - 0.5) + 0.005 * rand(), dy: 0.4 * (y - 0.5) + 0.005 * rand(), open: 0.3 },
      left: { dx: 0.3 * rand(), dy: 0.3 * rand(), open: 0.3 }, // unrelated to where the dot is
      head: stillHead(rand),
    }));
    const { best, all } = fitGaze(samples);
    expect(best.choice).toBe('right');
    expect(best.errorX).toBeLessThan(0.05);
    expect(best.errorY).toBeLessThan(0.05);
    const left = all.find((m) => m.choice === 'left')!;
    expect(left.errorX).toBeGreaterThan(best.errorX * 3);
    const p = predict(best, samples[0].features);
    expect(p.x).toBeCloseTo(0.1, 1);
    expect(p.y).toBeCloseTo(0.1, 1);
  });

  it('lets the eyes move the cursor even when the head drifted along during calibration', () => {
    // People turn their head slightly towards each dot; here the head follows the eyes exactly.
    const samples = calibrate((x, y, rand) => {
      const eye = { dx: 0.4 * (x - 0.5) + 0.003 * rand(), dy: 0.4 * (y - 0.5) + 0.003 * rand(), open: 0.3 };
      return {
        right: eye,
        left: { ...eye },
        head: { ...stillHead(rand), fx: 0.05 * (x - 0.5) + 0.001 * rand(), fy: 0.05 * (y - 0.5) + 0.001 * rand() },
      };
    });
    const { best } = fitGaze(samples);
    const centre = samples.find((s) => s.point === 6)!.features; // the middle dot
    const base = predict(best, centre);
    const look = 0.08; // eye widths of iris movement, which is 0.2 of the screen in this data
    const eyesOnly = predict(best, {
      ...centre,
      right: { ...centre.right, dx: centre.right.dx + look },
      left: { ...centre.left, dx: centre.left.dx + look },
    });
    expect(eyesOnly.x - base.x).toBeGreaterThan(0.15);
  });

  it('fits a head pointer from head direction alone', () => {
    const samples = calibrate((x, y, rand) => ({
      right: { dx: 0.01 * rand(), dy: 0.01 * rand(), open: 0.3 },
      left: { dx: 0.01 * rand(), dy: 0.01 * rand(), open: 0.3 },
      head: { ...stillHead(rand), fx: 0.3 * (x - 0.5) + 0.003 * rand(), fy: 0.3 * (y - 0.5) + 0.003 * rand() },
    }));
    const { best } = fitGaze(samples, 'head');
    expect(best.choice).toBe('head');
    expect(best.errorX).toBeLessThan(0.05);
  });
});

describe('OneEuro', () => {
  it('stays still on a still signal and follows a jump', () => {
    const f = new OneEuro();
    for (let t = 0; t < 10; t++) expect(f.filter(0.5, t * 33)).toBeCloseTo(0.5, 12);
    let y = 0;
    for (let t = 10; t < 40; t++) y = f.filter(1, t * 33);
    expect(y).toBeGreaterThan(0.9);
  });
});

describe('pointerProbabilities', () => {
  const rects = new Map<string, Rect>([
    ['water', { left: 0, top: 0, right: 100, bottom: 100 }],
    ['juice', { left: 110, top: 0, right: 210, bottom: 100 }],
    ['ball', { left: 500, top: 0, right: 600, bottom: 100 }],
  ]);

  it('favours the cell under the pointer', () => {
    const p = pointerProbabilities(rects, 50, 50, 40, 40);
    expect(mostLikely(p)).toBe('water');
    expect(p.get('ball')!).toBeLessThan(1e-6);
  });

  it('splits between two neighbours when the pointer falls in the gap', () => {
    const p = pointerProbabilities(rects, 105, 50, 40, 40);
    expect(p.get('water')!).toBeCloseTo(p.get('juice')!, 6);
    expect([...p.values()].reduce((s, v) => s + v, 0)).toBeCloseTo(1, 12);
  });
});
