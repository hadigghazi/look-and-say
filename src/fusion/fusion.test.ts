import { describe, expect, it } from 'vitest';
import { decide, fuse, ranked } from './fuse';
import { PointerHistory, pointerEvidence, type Rect } from './pointer';

const keys = ['water', 'juice', 'mum', 'ball'];
const dist = (entries: [string, number][]) => new Map(entries);

describe('fuse', () => {
  it('settles what neither channel can decide alone (mutual disambiguation)', () => {
    // The pointer landed between water and juice; the voice sounds like water or mum.
    const pointer = dist([['water', 0.45], ['juice', 0.45], ['mum', 0.05], ['ball', 0.05]]);
    const speech = dist([['water', 0.5], ['mum', 0.45], ['juice', 0.03], ['ball', 0.02]]);
    const fused = fuse(keys, [
      { dist: pointer, weight: 1 },
      { dist: speech, weight: 1 },
    ]);
    const [best, p] = ranked(fused)[0];
    expect(best).toBe('water');
    expect(p).toBeGreaterThan(0.8);
    expect([...fused.values()].reduce((s, q) => s + q, 0)).toBeCloseTo(1, 12);
  });

  it('uses the one channel that is there', () => {
    const fused = fuse(keys, [{ dist: dist([['water', 0.2], ['mum', 0.8]]), weight: 1 }]);
    expect(ranked(fused)[0][0]).toBe('mum');
  });

  it('keeps a floor so one channel cannot veto the other completely', () => {
    const fused = fuse(keys, [
      { dist: dist([['ball', 1]]), weight: 1 },
      { dist: dist([['water', 1]]), weight: 1 },
    ]);
    expect(fused.get('water')!).toBeGreaterThan(0.4);
    expect(fused.get('ball')!).toBeGreaterThan(0.4);
  });

  it('ignores a channel with weight zero', () => {
    const a = dist([['water', 0.7], ['juice', 0.3]]);
    const b = dist([['juice', 0.99], ['water', 0.01]]);
    const both = fuse(keys, [
      { dist: a, weight: 1 },
      { dist: b, weight: 0 },
    ]);
    expect(both.get('water')!).toBeCloseTo(fuse(keys, [{ dist: a, weight: 1 }]).get('water')!, 12);
  });
});

describe('decide', () => {
  it('chooses a word that is likely enough', () => {
    expect(decide(dist([['water', 0.7], ['juice', 0.2], ['mum', 0.1]]))).toEqual({ kind: 'select', key: 'water', probability: 0.7 });
  });

  it('asks between the likely words when unsure', () => {
    const d = decide(dist([['water', 0.45], ['juice', 0.4], ['mum', 0.1], ['ball', 0.05]]));
    expect(d.kind).toBe('ask');
    if (d.kind === 'ask') expect(d.candidates.map(([k]) => k)).toEqual(['water', 'juice']);
  });

  it('admits it understood nothing when the evidence is flat', () => {
    expect(decide(dist([['water', 0.25], ['juice', 0.25], ['mum', 0.25], ['ball', 0.25]])).kind).toBe('none');
  });
});

describe('pointer history', () => {
  it('keeps the last few seconds and returns a time window', () => {
    const h = new PointerHistory(1000);
    for (let t = 0; t <= 3000; t += 100) h.push({ t, x: t, y: 0, sx: 10, sy: 10 });
    expect(h.between(0, 1900)).toEqual([]);
    expect(h.between(2500, 2700).map((s) => s.t)).toEqual([2500, 2600, 2700]);
  });

  it('averages where the pointer was while the word was said', () => {
    const rects = new Map<string, Rect>([
      ['water', { left: 0, top: 0, right: 100, bottom: 100 }],
      ['juice', { left: 300, top: 0, right: 400, bottom: 100 }],
    ]);
    const at = (x: number) => ({ t: 0, x, y: 50, sx: 20, sy: 20 });
    const p = pointerEvidence(rects, [at(50), at(350)])!;
    expect(p.get('water')!).toBeCloseTo(0.5, 6);
    expect(p.get('juice')!).toBeCloseTo(0.5, 6);
    expect(pointerEvidence(rects, [])).toBeNull();
  });
});
