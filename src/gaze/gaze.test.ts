import { describe, expect, it } from 'vitest';
import type { FaceFeatures } from './features';
import { fitGaze, predict, type Sample } from './model';
import { OneEuro } from './oneEuro';
import { ridge, solve } from './ridge';
import { mostLikely, pointerProbabilities, type Rect } from '../fusion/pointer';

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
});

describe('fitGaze', () => {
  it('picks the eye that tracks when the other wanders, as with strabismus', () => {
    let seed = 1;
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 2 ** 32 - 0.5;
    };
    const dots = [[0.1, 0.1], [0.5, 0.1], [0.9, 0.1], [0.9, 0.5], [0.5, 0.5], [0.1, 0.5], [0.1, 0.9], [0.5, 0.9], [0.9, 0.9]];
    const samples: Sample[] = [];
    dots.forEach(([x, y], point) => {
      for (let k = 0; k < 20; k++) {
        const features: FaceFeatures = {
          right: { h: 0.3 + 0.4 * x + 0.005 * rand(), v: 0.3 + 0.4 * y + 0.005 * rand(), open: 0.3 },
          left: { h: 0.5 + 0.3 * rand(), v: 0.5 + 0.3 * rand(), open: 0.3 }, // unrelated to where the dot is
          yaw: 0.001 * rand(),
          pitch: 0.001 * rand(),
          headX: 0.5 + 0.001 * rand(),
          headY: 0.5 + 0.001 * rand(),
        };
        samples.push({ features, target: { x, y }, point });
      }
    });
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
