import { useEffect, useRef, type MutableRefObject } from 'react';
import type { GazeTracker } from './tracker';
import { isBlink } from './features';
import { predict, type GazeModel } from './model';
import { OneEuro } from './oneEuro';
import { mostLikely, pointerProbabilities, type Rect } from '../fusion/pointer';

interface Props {
  tracker: GazeTracker;
  model: GazeModel;
  targets: MutableRefObject<Map<string, HTMLElement>>;
  layoutKey: string; // changes when the targets move, so they are measured again at once
  dwellMs: number | null; // null: pointing alone never selects
  canDwell: () => boolean; // false holds the dwell clock (while the user is speaking, say)
  onHover: (key: string | null, dwelling: boolean) => void;
  onDwell: (key: string) => void;
  onFrame: (x: number, y: number, sx: number, sy: number, timeMs: number) => void;
}

/** Shows where the pointer is (a circle as wide as the calibration error) and reports the target under it. */
export function GazeOverlay({ tracker, model, targets, layoutKey, dwellMs, canDwell, onHover, onDwell, onFrame }: Props) {
  const cursor = useRef<HTMLDivElement>(null);
  const rects = useRef(new Map<string, Rect>());
  const handlers = useRef({ canDwell, onHover, onDwell, onFrame });
  handlers.current = { canDwell, onHover, onDwell, onFrame };

  useEffect(() => {
    const measure = () => {
      rects.current = new Map([...targets.current].map(([key, el]): [string, Rect] => [key, el.getBoundingClientRect()]));
    };
    measure();
    window.addEventListener('resize', measure);
    window.addEventListener('scroll', measure, { passive: true });
    const remeasure = window.setInterval(measure, 1000);
    return () => {
      window.removeEventListener('resize', measure);
      window.removeEventListener('scroll', measure);
      window.clearInterval(remeasure);
    };
  }, [targets, layoutKey]);

  useEffect(() => {
    const fx = new OneEuro();
    const fy = new OneEuro();
    // Dwell: pointing at one target for dwellMs selects it. The clock restarts when the target changes or
    // canDwell() turns false; the page blocks dwell for a moment after every choice, so a repeat takes longer.
    let hovered: string | null = null;
    let dwelling = false;
    let since = 0;
    const unsubscribe = tracker.subscribe((features, now) => {
      if (!features || isBlink(features)) return; // hold the last position through blinks
      const g = predict(model, features);
      const x = fx.filter(g.x, now) * window.innerWidth;
      const y = fy.filter(g.y, now) * window.innerHeight;
      if (cursor.current) cursor.current.style.transform = `translate(${x}px, ${y}px)`;
      const sx = Math.max(20, model.errorX * window.innerWidth);
      const sy = Math.max(20, model.errorY * window.innerHeight);
      handlers.current.onFrame(x, y, sx, sy, now);
      const key = mostLikely(pointerProbabilities(rects.current, x, y, sx, sy));
      const allowed = key !== null && dwellMs !== null && handlers.current.canDwell();
      if (key !== hovered) {
        hovered = key;
        dwelling = allowed;
        since = now;
        handlers.current.onHover(key, dwelling);
      } else if (allowed !== dwelling) {
        dwelling = allowed;
        since = now;
        handlers.current.onHover(key, dwelling);
      } else if (dwelling && key !== null && now - since >= (dwellMs ?? Number.POSITIVE_INFINITY)) {
        since = now;
        handlers.current.onDwell(key);
      }
    });
    return () => {
      unsubscribe();
      handlers.current.onHover(null, false);
    };
  }, [tracker, model, dwellMs]);

  const rx = Math.min(window.innerWidth * 0.4, Math.max(20, model.errorX * window.innerWidth));
  const ry = Math.min(window.innerHeight * 0.4, Math.max(20, model.errorY * window.innerHeight));
  return <div ref={cursor} className="gaze-cursor" style={{ width: 2 * rx, height: 2 * ry, marginLeft: -rx, marginTop: -ry }} />;
}
