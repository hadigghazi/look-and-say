// Pointer features from MediaPipe face landmarks: where each iris sits inside its eye, and the head's
// rotation and position. A per-user regression (model.ts) maps these to a point on the screen.

export interface Point {
  x: number;
  y: number;
}

export interface EyeFeatures {
  dx: number; // iris centre minus eye centre, across, in eye widths
  dy: number; // the same, down
  open: number; // lid gap in eye widths; small values mean a blink
}

export interface HeadPose {
  fx: number; // direction the face points (its z axis), sideways component
  fy: number; // the same, vertical component
  tx: number; // head position relative to the camera
  ty: number;
  tz: number;
}

export interface FaceFeatures {
  right: EyeFeatures; // the person's right eye
  left: EyeFeatures;
  head: HeadPose;
}

export type PointerChoice = 'right' | 'left' | 'both' | 'head';

// Face-mesh indices with iris refinement: the 5 iris points, the eye corners and lids, and the 16-point
// eye outline whose centroid is a steadier reference than the corners alone.
const RIGHT_EYE = {
  iris: [468, 469, 470, 471, 472],
  corners: [33, 133],
  lids: [159, 145],
  outline: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
};
const LEFT_EYE = {
  iris: [473, 474, 475, 476, 477],
  corners: [362, 263],
  lids: [386, 374],
  outline: [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
};

function centroid(lm: Point[], indices: number[]): Point {
  let x = 0;
  let y = 0;
  for (const i of indices) {
    x += lm[i].x;
    y += lm[i].y;
  }
  return { x: x / indices.length, y: y / indices.length };
}

function distance(a: Point, b: Point): number {
  return Math.hypot(b.x - a.x, b.y - a.y);
}

function eyeFeatures(lm: Point[], eye: typeof RIGHT_EYE): EyeFeatures {
  const iris = centroid(lm, eye.iris);
  const centre = centroid(lm, eye.outline);
  const width = distance(lm[eye.corners[0]], lm[eye.corners[1]]) || 1e-6;
  return {
    dx: (iris.x - centre.x) / width,
    dy: (iris.y - centre.y) / width,
    open: distance(lm[eye.lids[0]], lm[eye.lids[1]]) / width,
  };
}

/**
 * Head pose from MediaPipe's facial transformation matrix (4 x 4, column-major, as three.js reads it):
 * the face's z axis is the third column and the translation the fourth. Without a matrix, landmark
 * estimates stand in.
 */
export function headPose(lm: Point[], matrix?: number[]): HeadPose {
  if (matrix && matrix.length === 16) {
    return { fx: matrix[8], fy: matrix[9], tx: matrix[12], ty: matrix[13], tz: matrix[14] };
  }
  const right = lm[33];
  const left = lm[263];
  const nose = lm[1];
  const span = distance(right, left) || 1e-6;
  return {
    fx: (nose.x - (right.x + left.x) / 2) / span,
    fy: (nose.y - (right.y + left.y) / 2) / span,
    tx: (right.x + left.x) / 2,
    ty: (right.y + left.y) / 2,
    tz: 1 / span,
  };
}

export function faceFeatures(lm: Point[], matrix?: number[]): FaceFeatures {
  return { right: eyeFeatures(lm, RIGHT_EYE), left: eyeFeatures(lm, LEFT_EYE), head: headPose(lm, matrix) };
}

export function isBlink(f: FaceFeatures, threshold = 0.12): boolean {
  return f.right.open < threshold || f.left.open < threshold;
}

/**
 * Regression inputs. Eye models use the iris offset (with curvature terms) plus head pose and position
 * to correct for posture; the head model uses only the head.
 */
export function featureVector(f: FaceFeatures, choice: PointerChoice): number[] {
  const h = f.head;
  if (choice === 'head') return [h.fx, h.fy, h.fx * h.fx, h.fy * h.fy, h.fx * h.fy, h.tx, h.ty, h.tz];
  const e = choice === 'both' ? { dx: (f.right.dx + f.left.dx) / 2, dy: (f.right.dy + f.left.dy) / 2 } : f[choice];
  return [e.dx, e.dy, e.dx * e.dx, e.dy * e.dy, e.dx * e.dy, h.fx, h.fy, h.tx, h.ty, h.tz];
}

// While calibrating, people turn their head slightly towards each dot. Left unchecked, the regression
// credits the head and the eyes alone then barely move the cursor, so eye models penalise head terms
// far more: the eyes explain where you look, and the head only corrects for posture.
const HEAD_PENALTY = 100;

/** Ridge penalty for each input of featureVector (the bias is never penalised). */
export function penalties(choice: PointerChoice): number[] {
  if (choice === 'head') return new Array<number>(8).fill(1);
  return [1, 1, 1, 1, 1, HEAD_PENALTY, HEAD_PENALTY, HEAD_PENALTY, HEAD_PENALTY, HEAD_PENALTY];
}
