// One Euro filter (Casiez, Roussel and Vogel, CHI 2012): smooths jitter strongly while the gaze rests,
// and lightly while it moves, so the cursor is steady on a picture but still follows quick looks.

export class OneEuro {
  private previous: number | null = null;
  private previousDerivative = 0;
  private previousTime = 0;

  constructor(
    private readonly minCutoff = 0.8, // Hz: lower means smoother when still
    private readonly beta = 0.5, // how much speed raises the cutoff
    private readonly derivativeCutoff = 1, // Hz
  ) {}

  filter(value: number, timeMs: number): number {
    if (this.previous === null) {
      this.previous = value;
      this.previousTime = timeMs;
      return value;
    }
    const dt = Math.max((timeMs - this.previousTime) / 1000, 1e-3);
    this.previousTime = timeMs;
    const alpha = (cutoff: number) => 1 / (1 + 1 / (2 * Math.PI * cutoff * dt));
    const derivative = (value - this.previous) / dt;
    this.previousDerivative += alpha(this.derivativeCutoff) * (derivative - this.previousDerivative);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.previousDerivative);
    this.previous += alpha(cutoff) * (value - this.previous);
    return this.previous;
  }
}
