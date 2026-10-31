// Pointer probabilities: how likely each cell is the target, given where the gaze (or a touch) landed
// and how accurate that pointer is. Distance is measured to the cell's edge (zero inside the cell), then
// passed through a Gaussian whose width is the pointer's measured error.

import { averageDistributions } from './fuse';

export interface Rect {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

export function pointerProbabilities(rects: Map<string, Rect>, x: number, y: number, sigmaX: number, sigmaY: number): Map<string, number> {
  const weights = new Map<string, number>();
  let total = 0;
  for (const [key, r] of rects) {
    const dx = Math.max(r.left - x, 0, x - r.right) / sigmaX;
    const dy = Math.max(r.top - y, 0, y - r.bottom) / sigmaY;
    const w = Math.exp(-0.5 * (dx * dx + dy * dy));
    weights.set(key, w);
    total += w;
  }
  if (total > 0) for (const [key, w] of weights) weights.set(key, w / total);
  return weights;
}

/** The most likely key, or null when every probability is zero (the pointer is far from everything). */
export function mostLikely(probabilities: Map<string, number>): string | null {
  let best: string | null = null;
  let bestP = 0;
  for (const [key, p] of probabilities) {
    if (p > bestP) {
      best = key;
      bestP = p;
    }
  }
  return best;
}

export interface PointerSample {
  t: number; // performance.now() time
  x: number;
  y: number;
  sx: number; // spread (one standard deviation) across, in pixels
  sy: number; // and down
}

/** The last few seconds of pointer positions, so a spoken word can be matched with where the user pointed. */
export class PointerHistory {
  private samples: PointerSample[] = [];

  constructor(private readonly keepMs = 5000) {}

  push(sample: PointerSample): void {
    this.samples.push(sample);
    while (this.samples.length > 0 && this.samples[0].t < sample.t - this.keepMs) this.samples.shift();
  }

  between(from: number, to: number): PointerSample[] {
    return this.samples.filter((s) => s.t >= from && s.t <= to);
  }

  clear(): void {
    this.samples = [];
  }
}

/** How likely each cell is the target given a stretch of pointing: the mean over its samples, or null. */
export function pointerEvidence(rects: Map<string, Rect>, samples: readonly PointerSample[]): Map<string, number> | null {
  if (samples.length === 0 || rects.size === 0) return null;
  return averageDistributions(samples.map((s) => pointerProbabilities(rects, s.x, s.y, s.sx, s.sy)));
}
