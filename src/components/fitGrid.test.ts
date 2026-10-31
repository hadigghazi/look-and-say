import { describe, expect, it } from 'vitest';
import { fitGrid, GRID_GAP } from './fitGrid';

describe('fitGrid', () => {
  it('picks the columns that give the biggest pictures', () => {
    expect(fitGrid(6, 1000, 500)).toEqual({ cols: 3, rowHeight: 245, cellWidth: 326 });
  });

  it('fits a whole topic page on one screen', () => {
    const { cols, rowHeight } = fitGrid(12, 1200, 600);
    const rows = Math.ceil(12 / cols);
    expect(rows * rowHeight + (rows - 1) * GRID_GAP).toBeLessThanOrEqual(600);
  });

  it('scrolls rather than shrinking pictures below a usable size', () => {
    const { cellWidth, rowHeight } = fitGrid(34, 1200, 200);
    expect(cellWidth).toBeGreaterThanOrEqual(70);
    expect(rowHeight).toBeGreaterThanOrEqual(87);
  });
});
