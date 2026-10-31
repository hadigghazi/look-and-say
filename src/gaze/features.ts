// Gaze features from MediaPipe face landmarks: where each iris sits inside its eye, plus head pose and
// position. A per-user regression (model.ts) maps these to a point on the screen.

export interface Point {
  x: number;
  y: number;
}

export interface EyeFeatures {
  h: number; // iris position from the eye's image-left corner (0) to its image-right corner (1)
  v: number; // from the upper lid (0) to the lower lid (1)
  open: number; // lid gap relative to eye width; small values mean a blink
}

export interface FaceFeatures {
  right: EyeFeatures; // the person's right eye
  left: EyeFeatures;
  yaw: number; // nose offset from the eyes' midpoint, sideways, relative to eye spacing
  pitch: number; // the same, vertically
  headX: number; // eyes' midpoint in the camera image (0..1)
  headY: number;
}

export type EyeChoice = 'right' | 'left' | 'both';

// Face-mesh indices with iris refinement. Corners are ordered image-left then image-right,
// so both eyes' h grows in the same direction.
const RIGHT_EYE = { iris: 468, a: 33, b: 133, top: 159, bottom: 145 };
const LEFT_EYE = { iris: 473, a: 362, b: 263, top: 386, bottom: 374 };
const NOSE_TIP = 1;

function eyeFeatures(lm: Point[], eye: typeof RIGHT_EYE): EyeFeatures {
  const iris = lm[eye.iris];
  const a = lm[eye.a];
  const b = lm[eye.b];
  const top = lm[eye.top];
  const bottom = lm[eye.bottom];
  const width = b.x - a.x || 1e-6;
  const height = bottom.y - top.y || 1e-6;
  return { h: (iris.x - a.x) / width, v: (iris.y - top.y) / height, open: Math.abs(height / width) };
}

export function faceFeatures(lm: Point[]): FaceFeatures {
  const right = eyeFeatures(lm, RIGHT_EYE);
  const left = eyeFeatures(lm, LEFT_EYE);
  const rightOuter = lm[RIGHT_EYE.a];
  const leftOuter = lm[LEFT_EYE.b];
  const midX = (rightOuter.x + leftOuter.x) / 2;
  const midY = (rightOuter.y + leftOuter.y) / 2;
  const span = Math.hypot(leftOuter.x - rightOuter.x, leftOuter.y - rightOuter.y) || 1e-6;
  const nose = lm[NOSE_TIP];
  return { right, left, yaw: (nose.x - midX) / span, pitch: (nose.y - midY) / span, headX: midX, headY: midY };
}

export function isBlink(f: FaceFeatures, threshold = 0.12): boolean {
  return f.right.open < threshold || f.left.open < threshold;
}

/** Inputs for one regression: eye position with curvature terms, plus head pose and position. */
export function featureVector(f: FaceFeatures, choice: EyeChoice): number[] {
  const e = choice === 'both' ? { h: (f.right.h + f.left.h) / 2, v: (f.right.v + f.left.v) / 2 } : f[choice];
  return [e.h, e.v, e.h * e.h, e.v * e.v, e.h * e.v, f.yaw, f.pitch, f.headX, f.headY];
}
