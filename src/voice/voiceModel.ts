import { trimSilence } from '../audio/capture';
import { dtw } from '../audio/dtw';
import { mfcc } from '../audio/mfcc';
import type { TemplateSource } from '../storage/templateStore';

// The user's own recordings of each word ("templates") and the recogniser built on them. A new word is
// compared with every recording by dynamic time warping; the closer it is to a word's recordings, the
// more likely that word. No speech model is involved, so it works for speech that no recogniser
// understands, in any language or dialect.

export interface VoiceTemplate {
  id: number;
  key: string; // the grid cell
  source: TemplateSource;
  feats: Float64Array[];
  audio: Float32Array;
}

export interface Recognition {
  /** Probability of each word asked about (sums to 1). Words without recordings share the doubt. */
  dist: Map<string, number>;
  /** Closest distance to each word that has recordings. */
  distances: Map<string, number>;
  /** The word with the closest recording. */
  best: string;
  /** How likely it was none of the words with recordings. */
  unknown: number;
}

export interface SelfTest {
  tested: number;
  correct: number;
  perWord: Map<string, { tested: number; correct: number }>;
  confusions: { said: string; heard: string; count: number }[];
  temperature: number;
  unknownDistance: number | null;
}

/** Features for recognition: MFCCs of the word itself, without the quiet around it. */
export function wordFeatures(audio: Float32Array): Float64Array[] {
  return mfcc(trimSilence(audio));
}

export const DEFAULT_TEMPERATURE = 0.1; // a word 10% farther than the best match gets e^-1 of its weight
const TEMPERATURES = [0.05, 0.07, 0.1, 0.14, 0.2, 0.3, 0.5];
const SELF_TEST_PER_WORD = 4; // the self-test uses each word's most recent recordings
const UNKNOWN_MARGIN = 1.15; // "none of these" sits just past the farthest usual same-word match
const MIN_TRIALS = 5;

const pairId = (a: VoiceTemplate, b: VoiceTemplate) => (a.id < b.id ? `${a.id}:${b.id}` : `${b.id}:${a.id}`);

function closest(byWord: Map<string, number>): string | null {
  let best: string | null = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const [key, d] of byWord) {
    if (d < bestDistance) {
      best = key;
      bestDistance = d;
    }
  }
  return best;
}

interface Trial {
  said: string;
  byWord: Map<string, number>; // closest distance to each word's other recordings
}

/** Mean log-probability of the right word over the trials, with softmax temperature `relative` × best distance. */
function meanLogLikelihood(trials: Trial[], relative: number): number {
  let sum = 0;
  for (const trial of trials) {
    const dmin = Math.min(...trial.byWord.values());
    const temperature = Math.max(1e-9, relative * dmin);
    let z = 0;
    for (const d of trial.byWord.values()) z += Math.exp(-(d - dmin) / temperature);
    sum += -((trial.byWord.get(trial.said) ?? dmin) - dmin) / temperature - Math.log(z);
  }
  return sum / trials.length;
}

export class VoiceModel {
  /** Softmax temperature, relative to the best match's distance; fitted by the self-test. */
  temperature = DEFAULT_TEMPERATURE;
  /** Distance past which a sound is probably none of the recorded words; null until the self-test has run. */
  unknownDistance: number | null = null;
  private templates: VoiceTemplate[] = [];
  private readonly pairs = new Map<string, number>();

  get size(): number {
    return this.templates.length;
  }

  forKey(key: string): VoiceTemplate[] {
    return this.templates.filter((t) => t.key === key);
  }

  count(key: string): number {
    return this.forKey(key).length;
  }

  add(template: VoiceTemplate): void {
    this.templates.push(template);
  }

  remove(id: number): void {
    this.templates = this.templates.filter((t) => t.id !== id);
  }

  removeKey(key: string): VoiceTemplate[] {
    const gone = this.forKey(key);
    this.templates = this.templates.filter((t) => t.key !== key);
    return gone;
  }

  /**
   * How likely each of `keys` is, given a spoken word. Words without recordings, and "none of these",
   * sit at the unknown distance; the doubt "none of these" carries is spread evenly over all the keys,
   * so an unclear sound still works as a "now" signal while the pointer says which word.
   * Null when none of the keys has recordings.
   */
  recognise(feats: Float64Array[], keys: readonly string[]): Recognition | null {
    const wanted = new Set(keys);
    const distances = new Map<string, number>();
    for (const t of this.templates) {
      if (!wanted.has(t.key)) continue;
      const d = dtw(feats, t.feats);
      if (d < (distances.get(t.key) ?? Number.POSITIVE_INFINITY)) distances.set(t.key, d);
    }
    const best = closest(distances);
    if (best === null) return null;
    const dmin = distances.get(best) ?? 0;
    const temperature = Math.max(1e-9, this.temperature * dmin);
    // Before the self-test there is no absolute scale, so "none of these" sits two temperatures past the best.
    const unknownAt = this.unknownDistance ?? dmin * (1 + 2 * this.temperature);
    const reference = Math.min(dmin, unknownAt);
    const weight = (d: number) => Math.exp(-(d - reference) / temperature);
    const none = weight(unknownAt);
    const raw = keys.map((key): [string, number] => [key, weight(distances.get(key) ?? unknownAt)]);
    const total = raw.reduce((s, [, w]) => s + w, none);
    const dist = new Map(raw.map(([key, w]): [string, number] => [key, (w + none / keys.length) / total]));
    const unknown = raw.reduce((s, [key, w]) => (distances.has(key) ? s : s + w), none) / total;
    return { dist, distances, best, unknown };
  }

  /**
   * Leave-one-out self-test: each recording is recognised from the others. It reports how often that
   * works and which words get mixed up, and fits the temperature (how sure to be) and the unknown
   * distance to this user's recordings. A generator that yields its progress (0 to 1) after each
   * comparison, so the page can run it in short slices; distances are cached between runs.
   */
  *selfTest(): Generator<number, SelfTest, void> {
    const byKey = new Map<string, VoiceTemplate[]>();
    for (const t of this.templates) byKey.set(t.key, [...(byKey.get(t.key) ?? []), t]);
    const set = [...byKey.values()].flatMap((list) => list.slice(-SELF_TEST_PER_WORD));

    const missing: [VoiceTemplate, VoiceTemplate][] = [];
    for (let i = 0; i < set.length; i++) {
      for (let j = i + 1; j < set.length; j++) if (!this.pairs.has(pairId(set[i], set[j]))) missing.push([set[i], set[j]]);
    }
    for (let k = 0; k < missing.length; k++) {
      const [a, b] = missing[k];
      this.pairs.set(pairId(a, b), dtw(a.feats, b.feats));
      yield (k + 1) / missing.length;
    }

    const counts = new Map<string, number>();
    for (const t of set) counts.set(t.key, (counts.get(t.key) ?? 0) + 1);
    const trials: Trial[] = [];
    for (const t of set) {
      if ((counts.get(t.key) ?? 0) < 2) continue; // needs another recording of the same word
      const byWord = new Map<string, number>();
      for (const other of set) {
        if (other === t) continue;
        const d = this.pairs.get(pairId(t, other)) ?? dtw(t.feats, other.feats);
        if (d < (byWord.get(other.key) ?? Number.POSITIVE_INFINITY)) byWord.set(other.key, d);
      }
      trials.push({ said: t.key, byWord });
    }

    let correct = 0;
    const perWord = new Map<string, { tested: number; correct: number }>();
    const confusions = new Map<string, { said: string; heard: string; count: number }>();
    for (const trial of trials) {
      const heard = closest(trial.byWord);
      const right = heard === trial.said;
      if (right) correct++;
      const word = perWord.get(trial.said) ?? { tested: 0, correct: 0 };
      word.tested++;
      if (right) word.correct++;
      perWord.set(trial.said, word);
      if (!right && heard !== null) {
        const id = `${trial.said}>${heard}`;
        const c = confusions.get(id) ?? { said: trial.said, heard, count: 0 };
        c.count++;
        confusions.set(id, c);
      }
    }

    // How sure to be: the temperature under which the right words were most likely.
    const informative = trials.filter((trial) => trial.byWord.size >= 2);
    let temperature = DEFAULT_TEMPERATURE;
    if (informative.length >= MIN_TRIALS) {
      let bestScore = Number.NEGATIVE_INFINITY;
      for (const candidate of TEMPERATURES) {
        const score = meanLogLikelihood(informative, candidate);
        if (score > bestScore) {
          bestScore = score;
          temperature = candidate;
        }
      }
    }
    // "None of these": past nearly all of the distances between two recordings of the same word.
    const same = trials.map((trial) => trial.byWord.get(trial.said) ?? 0).sort((a, b) => a - b);
    const unknownDistance =
      same.length >= MIN_TRIALS ? UNKNOWN_MARGIN * same[Math.min(same.length - 1, Math.floor(0.9 * same.length))] : null;

    this.temperature = temperature;
    this.unknownDistance = unknownDistance;
    return {
      tested: trials.length,
      correct,
      perWord,
      confusions: [...confusions.values()].sort((a, b) => b.count - a.count),
      temperature,
      unknownDistance,
    };
  }
}
