// Ridge regression: least squares with a small penalty on the weights, so a handful of calibration
// points cannot be over-fitted. Solved exactly with Gaussian elimination (the systems are 10 x 10).

export interface Standardiser {
  mean: number[];
  std: number[];
}

export function standardiser(rows: number[][]): Standardiser {
  const d = rows[0].length;
  const mean = new Array<number>(d).fill(0);
  const std = new Array<number>(d).fill(0);
  for (const r of rows) for (let i = 0; i < d; i++) mean[i] += r[i] / rows.length;
  for (const r of rows) for (let i = 0; i < d; i++) std[i] += (r[i] - mean[i]) ** 2 / rows.length;
  return { mean, std: std.map((v) => Math.sqrt(v) || 1) };
}

/** Standardised inputs with a leading 1 for the bias. */
export function standardise(x: number[], s: Standardiser): number[] {
  return [1, ...x.map((v, i) => (v - s.mean[i]) / s.std[i])];
}

/** Solves A x = b by Gaussian elimination with partial pivoting. */
export function solve(A: number[][], b: number[]): number[] {
  const n = b.length;
  const m = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let pivot = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r][c]) > Math.abs(m[pivot][c])) pivot = r;
    [m[c], m[pivot]] = [m[pivot], m[c]];
    const p = m[c][c] || 1e-12;
    for (let r = c + 1; r < n; r++) {
      const f = m[r][c] / p;
      for (let k = c; k <= n; k++) m[r][k] -= f * m[c][k];
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = m[r][n];
    for (let k = r + 1; k < n; k++) s -= m[r][k] * x[k];
    x[r] = s / (m[r][r] || 1e-12);
  }
  return x;
}

/** Weights minimising |Z w - y|^2 + lambda |w|^2, leaving the bias (column 0) unpenalised. */
export function ridge(Z: number[][], y: number[], lambda: number): number[] {
  const d = Z[0].length;
  const A = Array.from({ length: d }, () => new Array<number>(d).fill(0));
  const b = new Array<number>(d).fill(0);
  Z.forEach((z, r) => {
    for (let i = 0; i < d; i++) {
      b[i] += z[i] * y[r];
      for (let j = 0; j < d; j++) A[i][j] += z[i] * z[j];
    }
  });
  for (let i = 1; i < d; i++) A[i][i] += lambda;
  return solve(A, b);
}

export function dot(w: number[], z: number[]): number {
  return w.reduce((s, wi, i) => s + wi * z[i], 0);
}
