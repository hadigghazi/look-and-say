import { dtw } from './dtw';

export interface Template {
  label: string;
  feats: Float64Array[];
}

export interface Ranked {
  word: string;
  distance: number;
  probability: number;
}

/**
 * Ranks words by their closest recorded template. Probabilities come from a softmax over
 * negative distances; the temperature still needs calibrating on real recordings.
 */
export function classify(feats: Float64Array[], templates: Template[], temperature = 1): Ranked[] {
  const best = new Map<string, number>();
  for (const t of templates) {
    const d = dtw(feats, t.feats);
    const previous = best.get(t.label);
    if (previous === undefined || d < previous) best.set(t.label, d);
  }
  const entries = [...best.entries()].sort((x, y) => x[1] - y[1]);
  if (entries.length === 0 || !Number.isFinite(entries[0][1])) return [];
  const minDistance = entries[0][1];
  const weights = entries.map(([, d]) => Math.exp(-(d - minDistance) / temperature));
  const total = weights.reduce((s, w) => s + w, 0);
  return entries.map(([word, d], i) => ({ word, distance: d, probability: weights[i] / total }));
}
