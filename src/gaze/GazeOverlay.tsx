import { useEffect, useRef, type MutableRefObject } from 'react';
import type { GazeTracker } from './tracker';
import { isBlink } from './features';
import { predict, type GazeModel } from './model';
import { OneEuro } from './oneEuro';
import { mostLikely, pointerProbabilities, type Rect } from '../fusion/pointer';

interface Props {
  tracker: GazeTracker;
  model: GazeModel;
  cells: MutableRefObject<Map<string, HTMLElement>>;
  dwellMs: number | null; // null: looking alone never selects
  onHover: (key: string | null) => void;
  onDwell: (key: string) => void;
}

/** Shows where the eyes look (a circle as wide as the calibration error) and reports the cell looked at. */
export function GazeOverlay({ tracker, model, cells, dwellMs, onHover, onDwell }: Props) {
  const cursor = useRef<HTMLDivElement>(null);
  const handlers = useRef({ onHover, onDwell });
  handlers.current = { onHover, onDwell };

  useEffect(() => {
    let rects = new Map<string, Rect>();
    const measure = () => {
      rects = new Map([...cells.current].map(([key, el]): [string, Rect] => [key, el.getBoundingClientRect()]));
    };
    measure();
    window.addEventListener('resize', measure);
    window.addEventListener('scroll', measure, { passive: true });
    const remeasure = window.setInterval(measure, 1000);

    const fx = new OneEuro();
    const fy = new OneEuro();
    let hovered: string | null = null;
    let since = 0;
    const unsubscribe = tracker.subscribe((features, now) => {
      if (!features || isBlink(features)) return; // hold the last position through blinks
      const g = predict(model, features);
      const x = fx.filter(g.x, now) * window.innerWidth;
      const y = fy.filter(g.y, now) * window.innerHeight;
      if (cursor.current) cursor.current.style.transform = `translate(${x}px, ${y}px)`;
      const sx = Math.max(20, model.errorX * window.innerWidth);
      const sy = Math.max(20, model.errorY * window.innerHeight);
      const key = mostLikely(pointerProbabilities(rects, x, y, sx, sy));
      if (key !== hovered) {
        hovered = key;
        since = now;
        handlers.current.onHover(key);
      } else if (key !== null && dwellMs !== null && now - since >= dwellMs) {
        since = now + dwellMs; // a repeat needs twice as long, so a resting gaze does not stutter
        handlers.current.onDwell(key);
      }
    });

    return () => {
      unsubscribe();
      window.removeEventListener('resize', measure);
      window.removeEventListener('scroll', measure);
      window.clearInterval(remeasure);
      handlers.current.onHover(null);
    };
  }, [tracker, model, cells, dwellMs]);

  const rx = Math.min(window.innerWidth * 0.4, Math.max(20, model.errorX * window.innerWidth));
  const ry = Math.min(window.innerHeight * 0.4, Math.max(20, model.errorY * window.innerHeight));
  return <div ref={cursor} className="gaze-cursor" style={{ width: 2 * rx, height: 2 * ry, marginLeft: -rx, marginTop: -ry }} />;
}
