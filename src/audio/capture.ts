// Microphone capture, resampling to 16 kHz, and energy-based detection of single spoken words.

export const TARGET_RATE = 16000;

/** Linear-interpolation resampler, with a moving-average low-pass first when downsampling. */
export function resample(input: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (fromRate === toRate) return input.slice();
  const ratio = fromRate / toRate;
  let source = input;
  if (ratio > 1) {
    const width = Math.ceil(ratio);
    const smoothed = new Float32Array(input.length);
    let sum = 0;
    for (let i = 0; i < input.length; i++) {
      sum += input[i];
      if (i >= width) sum -= input[i - width];
      smoothed[i] = sum / Math.min(i + 1, width);
    }
    source = smoothed;
  }
  const outLength = Math.floor(source.length / ratio);
  const out = new Float32Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, source.length - 1);
    const frac = pos - i0;
    out[i] = source[i0] * (1 - frac) + source[i1] * frac;
  }
  return out;
}

export function concat(parts: Float32Array[]): Float32Array {
  const total = parts.reduce((n, p) => n + p.length, 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const p of parts) {
    out.set(p, offset);
    offset += p.length;
  }
  return out;
}

function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
}

export interface DetectorOptions {
  frameMs: number; // analysis frame length
  startWindowMs: number; // a word starts when enough of this recent window is loud...
  startMs: number; // ...namely this much of it, so dips between syllables do not matter
  hangoverMs: number; // this much quiet ends a word
  preRollMs: number; // audio kept from just before the start
  tailMs: number; // quiet kept after the last loud frame, for a natural ending
  minVoicedMs: number; // sounds with less speech than this are clicks or bumps, not words
  maxUtteranceMs: number; // a word is cut here even if sound continues
  absoluteFloor: number; // frame loudness (RMS) below this is always silence
  startRatio: number; // to start, a frame must be this many times louder than the background
  stayRatio: number; // to keep going, a lower ratio (hysteresis)
  noiseWindowMs: number; // background noise is estimated over this much recent audio...
  noisePercentile: number; // ...as this percentile of frame loudness, i.e. its quiet moments
}

export const DEFAULT_DETECTOR: DetectorOptions = {
  frameMs: 10,
  startWindowMs: 80,
  startMs: 50,
  hangoverMs: 400,
  preRollMs: 250,
  tailMs: 60,
  minVoicedMs: 100,
  maxUtteranceMs: 3000,
  absoluteFloor: 0.002,
  startRatio: 3,
  stayRatio: 1.8,
  noiseWindowMs: 3000,
  noisePercentile: 0.15,
};

/** Splits a 16 kHz stream into single utterances: speech clearly above the background starts one, a pause ends it. */
export class UtteranceDetector {
  /** Loudness (RMS) of the latest frame, for a level meter. */
  level = 0;
  /** Estimated background loudness (RMS). */
  noise = 0;
  private readonly frameLen: number;
  private pending: Float32Array = new Float32Array(0);
  private preRoll: Float32Array[] = [];
  private current: Float32Array[] | null = null;
  private recent: boolean[] = [];
  private history: number[] = [];
  private sinceNoiseUpdate = 0;
  private silenceRun = 0;
  private voiced = 0;

  constructor(
    private readonly onUtterance: (audio: Float32Array) => void,
    private readonly opts: DetectorOptions = DEFAULT_DETECTOR,
    rate: number = TARGET_RATE,
  ) {
    this.frameLen = Math.round((opts.frameMs / 1000) * rate);
  }

  /** Loudness a frame needs to count towards starting a word. */
  get startThreshold(): number {
    return Math.max(this.opts.absoluteFloor, this.noise * this.opts.startRatio);
  }

  /** Loudness a frame needs to keep a word going. */
  get stayThreshold(): number {
    return Math.max(this.opts.absoluteFloor, this.noise * this.opts.stayRatio);
  }

  push(chunk: Float32Array): void {
    const data = concat([this.pending, chunk]);
    let offset = 0;
    while (offset + this.frameLen <= data.length) {
      this.frame(data.slice(offset, offset + this.frameLen));
      offset += this.frameLen;
    }
    this.pending = data.slice(offset);
  }

  private frames(ms: number): number {
    return Math.max(1, Math.round(ms / this.opts.frameMs));
  }

  private frame(frame: Float32Array): void {
    let sum = 0;
    for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
    const rms = Math.sqrt(sum / frame.length);
    this.level = rms;

    if (this.current === null) {
      // Background noise: the quiet moments of the last few seconds, measured only between words.
      this.history.push(rms);
      if (this.history.length > this.frames(this.opts.noiseWindowMs)) this.history.shift();
      if (++this.sinceNoiseUpdate >= 10 || this.history.length < 20) {
        this.sinceNoiseUpdate = 0;
        this.noise = percentile(this.history, this.opts.noisePercentile);
      }
      this.preRoll.push(frame);
      if (this.preRoll.length > this.frames(this.opts.preRollMs)) this.preRoll.shift();
      this.recent.push(rms > this.startThreshold);
      if (this.recent.length > this.frames(this.opts.startWindowMs)) this.recent.shift();
      if (this.recent.filter(Boolean).length >= this.frames(this.opts.startMs)) {
        this.current = this.preRoll;
        this.preRoll = [];
        this.recent = [];
        this.silenceRun = 0;
        this.voiced = this.frames(this.opts.startMs); // the frames that started it were speech
      }
      return;
    }

    this.current.push(frame);
    if (rms > this.stayThreshold) {
      this.voiced++;
      this.silenceRun = 0;
    } else {
      this.silenceRun++;
    }
    const length = this.current.length;
    if (this.silenceRun >= this.frames(this.opts.hangoverMs) || length >= this.frames(this.opts.maxUtteranceMs)) {
      const tail = Math.min(this.silenceRun, this.frames(this.opts.tailMs));
      const kept = this.current.slice(0, length - this.silenceRun + tail);
      const voiced = this.voiced;
      this.current = null;
      this.silenceRun = 0;
      this.voiced = 0;
      if (voiced >= this.frames(this.opts.minVoicedMs)) this.onUtterance(concat(kept));
    }
  }
}

export interface MicHandle {
  stop(): void;
  inputRate: number;
}

/** Starts the microphone and hands 16 kHz chunks to onChunk. Call from a click, so audio may start. */
export async function startMic(onChunk: (chunk16k: Float32Array) => void): Promise<MicHandle> {
  // Automatic gain keeps quiet laptop microphones usable; the MFCC mean normalisation cancels the
  // loudness differences it causes. Noise suppression stays off because it distorts atypical speech.
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: true },
  });
  const ctx = new AudioContext();
  await ctx.resume();
  const source = ctx.createMediaStreamSource(stream);
  // ScriptProcessorNode is deprecated but works everywhere, including Safari; an AudioWorklet can replace it later.
  const processor = ctx.createScriptProcessor(4096, 1, 1);
  processor.onaudioprocess = (event) => {
    onChunk(resample(event.inputBuffer.getChannelData(0), ctx.sampleRate, TARGET_RATE));
  };
  source.connect(processor);
  processor.connect(ctx.destination);
  return {
    inputRate: ctx.sampleRate,
    stop: () => {
      processor.disconnect();
      source.disconnect();
      stream.getTracks().forEach((t) => t.stop());
      void ctx.close();
    },
  };
}

/** Plays 16 kHz samples; resolves when playback ends, so words can be played one after another. */
export function playAudio(samples: Float32Array, rate: number = TARGET_RATE): Promise<void> {
  return new Promise((resolve) => {
    const ctx = new AudioContext();
    const buffer = ctx.createBuffer(1, samples.length, rate);
    buffer.getChannelData(0).set(samples);
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    source.onended = () => {
      void ctx.close();
      resolve();
    };
    source.start();
  });
}
