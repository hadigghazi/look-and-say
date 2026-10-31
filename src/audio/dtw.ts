// Dynamic time warping: how different two recordings are once their timing is lined up,
// so a word said slowly still matches the same word said quickly.

function distance(a: Float64Array, b: Float64Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

/**
 * Length-normalised DTW distance between two feature sequences.
 * `band` limits how far the alignment may stray from the diagonal (a fraction of the longer length).
 */
export function dtw(a: Float64Array[], b: Float64Array[], band = 0.3): number {
  const n = a.length;
  const m = b.length;
  if (n === 0 || m === 0) return Number.POSITIVE_INFINITY;
  const width = Math.max(Math.ceil(band * Math.max(n, m)), Math.abs(n - m));
  let prev = new Float64Array(m + 1).fill(Number.POSITIVE_INFINITY);
  let cur = new Float64Array(m + 1).fill(Number.POSITIVE_INFINITY);
  prev[0] = 0;
  for (let i = 1; i <= n; i++) {
    cur.fill(Number.POSITIVE_INFINITY);
    const from = Math.max(1, i - width);
    const to = Math.min(m, i + width);
    for (let j = from; j <= to; j++) {
      cur[j] = distance(a[i - 1], b[j - 1]) + Math.min(prev[j], cur[j - 1], prev[j - 1]);
    }
    [prev, cur] = [cur, prev];
  }
  return prev[m] / (n + m);
}
