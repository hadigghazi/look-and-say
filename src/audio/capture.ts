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

export interface DetectorOptions {
  frameMs: number; // analysis frame length
  minSpeechMs: number; // speech needed before a word starts
  hangoverMs: number; // silence that ends a word
  preRollMs: number; // audio kept from just before the start
  minUtteranceMs: number; // shorter detections are discarded as noise
  maxUtteranceMs: number; // a word is cut here even if speech continues
  absoluteFloor: number; // frame loudness (RMS) below this is always silence
  snrFactor: number; // speech must be this many times louder than the noise floor
}

export const DEFAULT_DETECTOR: DetectorOptions = {
  frameMs: 10,
  minSpeechMs: 60,
  hangoverMs: 350,
  preRollMs: 200,
  minUtteranceMs: 200,
  maxUtteranceMs: 3000,
  absoluteFloor: 0.008,
  snrFactor: 3,
};

/** Splits a 16 kHz stream into single utterances: loud frames start one, a pause ends it. */
export class UtteranceDetector {
  /** Loudness (RMS) of the latest frame, for a level meter. */
  level = 0;
  private readonly frameLen: number;
  private pending: Float32Array = new Float32Array(0);
  private preRoll: Float32Array[] = [];
  private current: Float32Array[] | null = null;
  private speechRun = 0;
  private silenceRun = 0;
  private noiseFloor = 0.01;

  constructor(
    private readonly onUtterance: (audio: Float32Array) => void,
    private readonly opts: DetectorOptions = DEFAULT_DETECTOR,
    rate: number = TARGET_RATE,
  ) {
    this.frameLen = Math.round((opts.frameMs / 1000) * rate);
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
    return Math.round(ms / this.opts.frameMs);
  }

  private frame(frame: Float32Array): void {
    let sum = 0;
    for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
    const rms = Math.sqrt(sum / frame.length);
    this.level = rms;
    const threshold = Math.max(this.opts.absoluteFloor, this.noiseFloor * this.opts.snrFactor);
    const isSpeech = rms > threshold;

    if (this.current === null) {
      // Track background noise only while nobody is speaking.
      if (!isSpeech) this.noiseFloor = 0.95 * this.noiseFloor + 0.05 * rms;
      this.preRoll.push(frame);
      if (this.preRoll.length > this.frames(this.opts.preRollMs)) this.preRoll.shift();
      this.speechRun = isSpeech ? this.speechRun + 1 : 0;
      if (this.speechRun >= this.frames(this.opts.minSpeechMs)) {
        this.current = this.preRoll;
        this.preRoll = [];
        this.silenceRun = 0;
      }
      return;
    }

    this.current.push(frame);
    this.silenceRun = isSpeech ? 0 : this.silenceRun + 1;
    const length = this.current.length;
    if (this.silenceRun >= this.frames(this.opts.hangoverMs) || length >= this.frames(this.opts.maxUtteranceMs)) {
      const kept = this.current.slice(0, length - this.silenceRun); // drop the trailing pause
      this.current = null;
      this.speechRun = 0;
      this.silenceRun = 0;
      if (kept.length >= this.frames(this.opts.minUtteranceMs)) this.onUtterance(concat(kept));
    }
  }
}

export interface MicHandle {
  stop(): void;
  inputRate: number;
}

/** Starts the microphone and hands 16 kHz chunks to onChunk. Call from a click, so audio may start. */
export async function startMic(onChunk: (chunk16k: Float32Array) => void): Promise<MicHandle> {
  // Browser noise suppression and gain control would change each recording differently, so they are off.
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
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
