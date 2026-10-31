import { TARGET_RATE } from './capture';

// Sound the app makes: recorded words and short tones ("earcons"), through one shared audio context that
// a tap unlocks (iPad Safari only lets sound start from a touch), plus a gate that remembers when the app
// was sounding, so the microphone does not take the app's own voice for the user's.

let context: AudioContext | null = null;

function audioContext(): AudioContext {
  if (!context || context.state === 'closed') context = new AudioContext();
  if (context.state === 'suspended') context.resume().catch(() => undefined);
  return context;
}

/** Call from a click or tap handler, so that sounds started later (after a spoken word) may play. */
export function unlockAudio(): void {
  audioContext();
}

/** Plays 16 kHz samples; resolves when playback ends, so words can be played one after another. */
export function playAudio(samples: Float32Array, rate: number = TARGET_RATE): Promise<void> {
  if (samples.length === 0) return Promise.resolve();
  const ctx = audioContext();
  return new Promise((resolve) => {
    const buffer = ctx.createBuffer(1, samples.length, rate);
    buffer.getChannelData(0).set(samples);
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    // A timer too: a context the browser keeps suspended never reports the end.
    const fallback = window.setTimeout(resolve, (samples.length / rate) * 1000 + 500);
    source.onended = () => {
      window.clearTimeout(fallback);
      resolve();
    };
    source.start();
  });
}

export type Earcon = 'got' | 'unsure' | 'missed';

// Each earcon is a few tones: [frequency in Hz, length in ms].
const TONES: Record<Earcon, [number, number][]> = {
  got: [[880, 70]], // a short tick: recorded
  unsure: [[523, 110], [784, 150]], // rising, like a question: did you mean…?
  missed: [[262, 220]], // low: not understood
};

/** Plays a short non-verbal signal; resolves when it has finished. */
export function earcon(kind: Earcon): Promise<void> {
  const ctx = audioContext();
  const start = ctx.currentTime + 0.02;
  let t = start;
  for (const [frequency, ms] of TONES[kind]) {
    const end = t + ms / 1000;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.frequency.value = frequency;
    gain.gain.setValueAtTime(0, t);
    gain.gain.linearRampToValueAtTime(0.2, t + 0.015);
    gain.gain.linearRampToValueAtTime(0, end);
    osc.connect(gain).connect(ctx.destination);
    osc.start(t);
    osc.stop(end);
    t = end + 0.04;
  }
  return new Promise((resolve) => window.setTimeout(resolve, (t - start) * 1000 + 20));
}

/** Remembers when the app itself made sound, so that sound is not taken for the user speaking. */
export class OutputGate {
  private playing = 0;
  private lastEnd = Number.NEGATIVE_INFINITY;

  async run(play: () => Promise<void>): Promise<void> {
    this.playing++;
    try {
      await play();
    } finally {
      this.playing--;
      this.lastEnd = performance.now();
    }
  }

  /** Whether the app was sounding at any moment since `time` (a performance.now() time). */
  soundedSince(time: number): boolean {
    return this.playing > 0 || this.lastEnd >= time;
  }
}
