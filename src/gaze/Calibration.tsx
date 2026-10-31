import { useEffect, useRef, useState } from 'react';
import type { GazeTracker } from './tracker';
import { isBlink } from './features';
import { fitGaze, type FitResult, type Sample } from './model';

// Nine dots across the window. The eyes follow each one, and the camera features recorded while looking
// at it become training data for the gaze model. Snake order keeps the jumps between dots short.
const DOTS: [number, number][] = [
  [0.1, 0.1], [0.5, 0.1], [0.9, 0.1],
  [0.9, 0.5], [0.5, 0.5], [0.1, 0.5],
  [0.1, 0.9], [0.5, 0.9], [0.9, 0.9],
];
const SETTLE_MS = 700; // time for the eyes to land on the new dot before recording
const COLLECT_MS = 1000; // recording time per dot
const MIN_SAMPLES = 60; // fewer usable frames than this means the camera did not see the eyes

export type CalibrationEnd = 'done' | 'cancelled' | 'failed';

interface Props {
  tracker: GazeTracker;
  onDone: (result: FitResult | null, end: CalibrationEnd) => void;
}

export function Calibration({ tracker, onDone }: Props) {
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
      onDoneRef.current(enough ? fitGaze(samples.current) : null, enough ? 'done' : 'failed');
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
  }, [index]);

  if (index < 0) {
    return (
      <div className="calibration">
        <div className="calib-intro">
          <h2>Eye-gaze calibration</h2>
          <p>
            Sit the way you will use the app, keep your head still, and follow the blue dot with your eyes. It visits 9
            places and takes about 15 seconds.
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
