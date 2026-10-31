// Late fusion. Each channel (the pointer, the voice, the earlier tries at the same word) gives a probability
// for every cell on screen. The decision multiplies them, each raised to its weight (a weighted sum of
// log-probabilities), and renormalises. A cell one channel rules out keeps a small floor, so a single
// confident mistake cannot veto the other channels completely.

export interface Channel {
  dist: Map<string, number>;
  weight: number;
}

const FLOOR = 1e-3;

export function fuse(keys: readonly string[], channels: readonly Channel[], floor = FLOOR): Map<string, number> {
  const logs = keys.map((key): [string, number] => [
    key,
    channels.reduce((s, c) => s + c.weight * Math.log(Math.max(c.dist.get(key) ?? 0, floor)), 0),
  ]);
  const max = Math.max(...logs.map(([, s]) => s));
  const weights = logs.map(([key, s]): [string, number] => [key, Math.exp(s - max)]);
  const total = weights.reduce((s, [, w]) => s + w, 0);
  return new Map(weights.map(([key, w]): [string, number] => [key, w / total]));
}

/** Mean of several distributions over the same cells (e.g. pointer frames during an utterance). */
export function averageDistributions(list: Map<string, number>[]): Map<string, number> {
  const out = new Map<string, number>();
  for (const d of list) for (const [key, p] of d) out.set(key, (out.get(key) ?? 0) + p / list.length);
  return out;
}

/** Entries from most to least likely. */
export function ranked(d: Map<string, number>): [string, number][] {
  return [...d.entries()].sort((a, b) => b[1] - a[1]);
}

export type Decision =
  | { kind: 'select'; key: string; probability: number }
  | { kind: 'ask'; candidates: [string, number][] }
  | { kind: 'none' };

export const ACCEPT = 0.6; // a word at least this likely is chosen outright
const ASK_FLOOR = 0.2; // a best word below this (or barely above an even spread) means the evidence says nothing
const MAX_CANDIDATES = 3;

/** Choose the word, ask "did you mean…?" among the few likely ones, or admit that nothing was understood. */
export function decide(fused: Map<string, number>, accept = ACCEPT): Decision {
  const order = ranked(fused);
  if (order.length === 0) return { kind: 'none' };
  const [key, p] = order[0];
  if (p >= accept) return { kind: 'select', key, probability: p };
  if (p < Math.min(0.5, Math.max(ASK_FLOOR, 1.5 / order.length))) return { kind: 'none' };
  return { kind: 'ask', candidates: order.filter(([, q]) => q >= p / 3).slice(0, MAX_CANDIDATES) };
}
