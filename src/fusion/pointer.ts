// Pointer probabilities: how likely each cell is the target, given where the gaze (or a touch) landed
// and how accurate that pointer is. Distance is measured to the cell's edge (zero inside the cell), then
// passed through a Gaussian whose width is the pointer's measured error.

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

export function mostLikely(probabilities: Map<string, number>): string | null {
  let best: string | null = null;
  let bestP = -1;
  for (const [key, p] of probabilities) {
    if (p > bestP) {
      best = key;
      bestP = p;
    }
  }
  return best;
}
