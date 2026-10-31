import { useEffect, useRef, useState } from 'react';
import type { GazeTracker } from './tracker';
import { isBlink } from './features';
import { fitGaze, type FitResult, type PointerMode, type Sample } from './model';

// Thirteen dots across the window: a 3 x 3 grid plus four in between. While the user looks at (or points
// their head at) each one, the camera features become training data for the pointer model. The order
// snakes across rows so the jumps between dots stay short.
const DOTS: [number, number][] = [
  [0.1, 0.1], [0.5, 0.1], [0.9, 0.1],
  [0.7, 0.3], [0.3, 0.3],
  [0.1, 0.5], [0.5, 0.5], [0.9, 0.5],
  [0.7, 0.7], [0.3, 0.7],
  [0.1, 0.9], [0.5, 0.9], [0.9, 0.9],
];
const SETTLE_MS = 600; // time to land on the new dot before recording
const COLLECT_MS = 1100; // recording time per dot
const MIN_SAMPLES = 80; // fewer usable frames than this means the camera did not see the face

export type CalibrationEnd = 'done' | 'cancelled' | 'failed';

const INSTRUCTIONS: Record<PointerMode, string> = {
  eyes: 'Rest your head (a hand under the chin helps), keep it still, and follow the blue dot with your eyes only.',
  head: 'Point your nose at the blue dot, as if your head were a pointer, and follow it as it moves.',
};

interface Props {
  tracker: GazeTracker;
  mode: PointerMode;
  onDone: (result: FitResult | null, end: CalibrationEnd) => void;
}

export function Calibration({ tracker, mode, onDone }: Props) {
  const [index, setIndex] = useState(-1); // -1 shows the instructions
  const samples = useRef<Sample[]>([]);
  const recording = useRef<number | null>(null); // the dot being recorded, if any
  const onDoneRef = useRef(onDone);
  onDoneRef.current = onDone;

  useEffect(
    () =>
      tracker.subscribe((features) => {
        const dot = recording.current;
        if (dot === null || !features || isBlink(features)) return;
        const [x, y] = DOTS[dot];
        samples.current.push({ features, target: { x, y }, point: dot });
      }),
    [tracker],
  );

  useEffect(() => {
    if (index < 0) return;
    if (index >= DOTS.length) {
      const enough = samples.current.length >= MIN_SAMPLES;
      onDoneRef.current(enough ? fitGaze(samples.current, mode) : null, enough ? 'done' : 'failed');
      return;
    }
    recording.current = null;
    const startRecording = window.setTimeout(() => {
      recording.current = index;
    }, SETTLE_MS);
    const nextDot = window.setTimeout(() => {
      recording.current = null;
      setIndex((i) => i + 1);
    }, SETTLE_MS + COLLECT_MS);
    return () => {
      window.clearTimeout(startRecording);
      window.clearTimeout(nextDot);
    };
  }, [index, mode]);

  if (index < 0) {
    return (
      <div className="calibration">
        <div className="calib-intro">
          <h2>{mode === 'eyes' ? 'Eye-gaze calibration' : 'Head-pointer calibration'}</h2>
          <p>
            {INSTRUCTIONS[mode]} It visits {DOTS.length} places and takes about 25 seconds.
          </p>
          <div className="row calib-buttons">
            <button type="button" onClick={() => setIndex(0)}>
              Start
            </button>
            <button type="button" onClick={() => onDone(null, 'cancelled')}>
              Cancel
            </button>
          </div>
        </div>
      </div>
    );
  }

  const [x, y] = DOTS[Math.min(index, DOTS.length - 1)];
  return (
    <div className="calibration">
      <div className="calib-dot" style={{ left: `${x * 100}%`, top: `${y * 100}%` }} />
      <div className="calib-progress">
        {Math.min(index + 1, DOTS.length)} / {DOTS.length}
      </div>
    </div>
  );
}
