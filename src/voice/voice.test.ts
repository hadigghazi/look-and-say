import { describe, expect, it } from 'vitest';
import { ranked } from '../fusion/fuse';
import { VoiceModel, type SelfTest, type VoiceTemplate } from './voiceModel';

/** Deterministic pseudo-random numbers in [0, 1), so tests always see the same data. */
function random(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

// "Words" as paths through a 2-D feature space. Each recording follows its word's path at its own
// length and pace, with noise, the way one person's tries at a word differ.
const PATHS: Record<string, (t: number) => [number, number]> = {
  water: (t) => [t, 0],
  juice: (t) => [0, t],
  ball: (t) => [1 - t, 1],
  mum: (t) => [t, 0], // sounds exactly like water
};

function say(word: string, rand: () => number, noise = 0.05): Float64Array[] {
  const n = 16 + Math.floor(rand() * 9);
  const pace = 0.8 + 0.4 * rand();
  const out: Float64Array[] = [];
  for (let i = 0; i < n; i++) {
    const [x, y] = PATHS[word]((i / (n - 1)) ** pace);
    out.push(Float64Array.from([x + (rand() - 0.5) * 2 * noise, y + (rand() - 0.5) * 2 * noise]));
  }
  return out;
}

let nextId = 1;
function trained(words: string[], perWord: number, rand: () => number): VoiceModel {
  const model = new VoiceModel();
  for (const word of words) {
    for (let k = 0; k < perWord; k++) {
      const t: VoiceTemplate = { id: nextId++, key: word, source: 'taught', feats: say(word, rand), audio: new Float32Array(0) };
      model.add(t);
    }
  }
  return model;
}

function selfTest(model: VoiceModel): SelfTest {
  const steps = model.selfTest();
  let step = steps.next();
  while (!step.done) step = steps.next();
  return step.value;
}

const WORDS = ['water', 'juice', 'ball'];

describe('VoiceModel.recognise', () => {
  it("recognises a word from the user's own recordings", () => {
    const rand = random(7);
    const model = trained(WORDS, 3, rand);
    const r = model.recognise(say('juice', rand), WORDS)!;
    expect(r.best).toBe('juice');
    expect(ranked(r.dist)[0][0]).toBe('juice');
    expect(r.dist.get('juice')!).toBeGreaterThan(0.8);
    expect([...r.dist.values()].reduce((s, p) => s + p, 0)).toBeCloseTo(1, 12);
  });

  it('gives words without recordings a share of the doubt', () => {
    const rand = random(8);
    const model = trained(WORDS, 3, rand);
    const r = model.recognise(say('juice', rand), [...WORDS, 'dad'])!;
    expect(r.dist.get('dad')!).toBeGreaterThan(0);
    expect(r.dist.get('dad')!).toBeLessThan(r.dist.get('juice')!);
  });

  it('returns null when no word on screen has recordings', () => {
    const rand = random(9);
    expect(trained(WORDS, 3, rand).recognise(say('juice', rand), ['mum', 'dad'])).toBeNull();
  });
});

describe('VoiceModel.selfTest', () => {
  it('recognises each recording from the others and fits how sure to be', () => {
    const model = trained(WORDS, 3, random(11));
    const result = selfTest(model);
    expect(result.tested).toBe(9);
    expect(result.correct).toBe(9);
    expect(result.confusions).toEqual([]);
    expect(result.unknownDistance).not.toBeNull();
    expect(model.temperature).toBe(result.temperature);
  });

  it('reports words that sound alike', () => {
    const result = selfTest(trained(['water', 'mum', 'juice'], 3, random(12)));
    expect(result.correct).toBeLessThan(result.tested);
    expect(result.confusions.length).toBeGreaterThan(0);
    const alike = ['water', 'mum'];
    expect(result.confusions.every((c) => alike.includes(c.said) && alike.includes(c.heard))).toBe(true);
  });

  it('afterwards marks a sound unlike every word as unknown', () => {
    const model = trained(WORDS, 3, random(13));
    selfTest(model);
    const strange = Array.from({ length: 20 }, () => Float64Array.from([3, -3]));
    expect(model.recognise(strange, WORDS)!.unknown).toBeGreaterThan(0.9);
  });
});
