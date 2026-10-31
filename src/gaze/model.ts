import { featureVector, penalties, type FaceFeatures, type PointerChoice } from './features';
import { dot, ridge, standardise, standardiser } from './ridge';

// A per-user pointer model: two ridge regressions (screen x and y, as fractions of the window) fitted on
// calibration samples. In eye mode, models are fitted for the right eye, the left eye and both together,
// and the most accurate is used, so a user whose eyes do not move together (strabismus) still gets the
// better eye. In head mode, the head's direction is the pointer.

export type PointerMode = 'eyes' | 'head';

export interface Sample {
  features: FaceFeatures;
  target: { x: number; y: number }; // where the dot was, as fractions of the window
  point: number; // which calibration dot
}

export interface GazeModel {
  mode: PointerMode;
  choice: PointerChoice;
  mean: number[];
  std: number[];
  wx: number[];
  wy: number[];
  errorX: number; // typical error across, as a fraction of the window width (leave-one-dot-out)
  errorY: number; // typical error down, as a fraction of the window height
}

export interface FitResult {
  best: GazeModel;
  all: GazeModel[];
  headMovement: number; // spread of head direction during calibration; large means the head moved
}

const EYE_CHOICES: PointerChoice[] = ['both', 'right', 'left'];

function fitWeights(samples: Sample[], choice: PointerChoice) {
  const X = samples.map((s) => featureVector(s.features, choice));
  const s = standardiser(X);
  const Z = X.map((x) => standardise(x, s));
  const lambda = penalties(choice);
  return {
    mean: s.mean,
    std: s.std,
    wx: ridge(Z, samples.map((t) => t.target.x), lambda),
    wy: ridge(Z, samples.map((t) => t.target.y), lambda),
  };
}

export function predict(m: Pick<GazeModel, 'choice' | 'mean' | 'std' | 'wx' | 'wy'>, f: FaceFeatures): { x: number; y: number } {
  const z = standardise(featureVector(f, m.choice), { mean: m.mean, std: m.std });
  return { x: dot(m.wx, z), y: dot(m.wy, z) };
}

/** Error measured honestly: fit without each dot in turn and test on the dot left out. */
function crossValidatedError(samples: Sample[], choice: PointerChoice): { x: number; y: number } {
  const points = [...new Set(samples.map((s) => s.point))];
  let sx = 0;
  let sy = 0;
  let n = 0;
  for (const p of points) {
    const train = samples.filter((s) => s.point !== p);
    const test = samples.filter((s) => s.point === p);
    if (train.length < 10 || test.length === 0) continue;
    const m = { choice, ...fitWeights(train, choice) };
    for (const t of test) {
      const g = predict(m, t.features);
      sx += (g.x - t.target.x) ** 2;
      sy += (g.y - t.target.y) ** 2;
      n++;
    }
  }
  return n > 0 ? { x: Math.sqrt(sx / n), y: Math.sqrt(sy / n) } : { x: 1, y: 1 };
}

function spread(values: number[]): number {
  const mean = values.reduce((s, v) => s + v, 0) / values.length;
  return Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length);
}

export function fitGaze(samples: Sample[], mode: PointerMode = 'eyes'): FitResult {
  const choices: PointerChoice[] = mode === 'eyes' ? EYE_CHOICES : ['head'];
  const all = choices.map((choice): GazeModel => {
    const error = crossValidatedError(samples, choice);
    return { mode, choice, ...fitWeights(samples, choice), errorX: error.x, errorY: error.y };
  });
  const size = (m: GazeModel) => Math.hypot(m.errorX, m.errorY);
  const best = all.reduce((a, b) => (size(b) < size(a) ? b : a));
  const headMovement = Math.max(
    spread(samples.map((s) => s.features.head.fx)),
    spread(samples.map((s) => s.features.head.fy)),
  );
  return { best, all, headMovement };
}

const storageKey = (mode: PointerMode) => `las.pointer.${mode}`;

export function saveModel(m: GazeModel): void {
  try {
    window.localStorage.setItem(storageKey(m.mode), JSON.stringify(m));
  } catch {
    // Not saved; the model still works until the page is reloaded.
  }
}

export function loadModel(mode: PointerMode): GazeModel | null {
  try {
    const raw = window.localStorage.getItem(storageKey(mode));
    return raw ? (JSON.parse(raw) as GazeModel) : null;
  } catch {
    return null;
  }
}
