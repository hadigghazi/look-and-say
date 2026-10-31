import { useLayoutEffect, useState, type RefObject } from 'react';

// Grid layout that fits the whole page on one screen with the biggest pictures possible: a pointer
// (eyes, head) can only reach what is on screen, and bigger targets forgive a rough pointer.

export const GRID_GAP = 10; // matches .cells { gap } in styles.css
const MIN_PICTURE = 70; // below this the page scrolls rather than shrinking pictures further
const CELL_RATIO = 1.25; // a cell is a square picture with its label below
const BELOW = 40; // room under the grid for the attribution line

export interface GridLayout {
  cols: number;
  rowHeight: number;
  cellWidth: number;
  top: number; // where the grid starts on the page, so the pointer re-measures when it moves
}

/** Columns and row height that fit `count` cells into width × height with the biggest pictures. */
export function fitGrid(count: number, width: number, height: number, gap = GRID_GAP): Omit<GridLayout, 'top'> {
  const n = Math.max(1, count);
  let best = { cols: 1, cellWidth: width, rowHeight: height, picture: 0 };
  for (let cols = 1; cols <= n; cols++) {
    const rows = Math.ceil(n / cols);
    const cellWidth = (width - gap * (cols - 1)) / cols;
    const cellHeight = (height - gap * (rows - 1)) / rows;
    const picture = Math.min(cellWidth, cellHeight / CELL_RATIO);
    if (picture > best.picture) best = { cols, cellWidth, rowHeight: Math.min(cellHeight, cellWidth * CELL_RATIO), picture };
  }
  if (best.picture < MIN_PICTURE) {
    // Too many cells for the screen: keep the pictures usable and let the page scroll.
    const cols = Math.max(1, Math.min(n, Math.floor((width + gap) / (MIN_PICTURE + gap))));
    const cellWidth = (width - gap * (cols - 1)) / cols;
    return { cols, rowHeight: Math.floor(cellWidth * CELL_RATIO), cellWidth: Math.floor(cellWidth) };
  }
  return { cols: best.cols, rowHeight: Math.floor(best.rowHeight), cellWidth: Math.floor(best.cellWidth) };
}

/** Keeps the grid fitted to the space left under `head` as the window or the header changes size. */
export function useFitGrid(grid: RefObject<HTMLElement>, head: RefObject<HTMLElement>, count: number): GridLayout {
  const [layout, setLayout] = useState<GridLayout>({ cols: 6, rowHeight: 150, cellWidth: 150, top: 0 });
  useLayoutEffect(() => {
    const el = grid.current;
    if (!el) return;
    const update = () => {
      const top = Math.round(el.getBoundingClientRect().top + window.scrollY);
      const next = { ...fitGrid(count, el.clientWidth, window.innerHeight - top - BELOW), top };
      setLayout((prev) =>
        prev.cols === next.cols && prev.rowHeight === next.rowHeight && prev.cellWidth === next.cellWidth && prev.top === next.top
          ? prev
          : next,
      );
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(el);
    if (head.current) observer.observe(head.current);
    window.addEventListener('resize', update);
    return () => {
      observer.disconnect();
      window.removeEventListener('resize', update);
    };
  }, [grid, head, count]);
  return layout;
}
