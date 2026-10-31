// MFCC features: the standard compact description of how a short slice of speech sounds.
// 25 ms frames every 10 ms -> power spectrum -> 26 mel bands -> log -> DCT -> 13 coefficients,
// then cepstral mean normalisation so the microphone and room matter less.

const FRAME = 400; // 25 ms at 16 kHz
const HOP = 160; // 10 ms
const NFFT = 512;
const NMELS = 26;
const NCEPS = 13;

/** In-place iterative radix-2 FFT; the length must be a power of two. */
export function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const angle = (-2 * Math.PI) / len;
    const wr = Math.cos(angle);
    const wi = Math.sin(angle);
    for (let start = 0; start < n; start += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < half; k++) {
        const a = start + k;
        const b = a + half;
        const vr = re[b] * cr - im[b] * ci;
        const vi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - vr;
        im[b] = im[a] - vi;
        re[a] += vr;
        im[a] += vi;
        const next = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = next;
      }
    }
  }
}

const hzToMel = (hz: number) => 2595 * Math.log10(1 + hz / 700);
const melToHz = (mel: number) => 700 * (10 ** (mel / 2595) - 1);

function melFilterbank(sampleRate: number): Float64Array[] {
  const melMin = hzToMel(20);
  const melMax = hzToMel(sampleRate / 2);
  const bins = Array.from({ length: NMELS + 2 }, (_, i) => {
    const hz = melToHz(melMin + (i * (melMax - melMin)) / (NMELS + 1));
    return Math.floor(((NFFT + 1) * hz) / sampleRate);
  });
  const filters: Float64Array[] = [];
  for (let m = 1; m <= NMELS; m++) {
    const f = new Float64Array(NFFT / 2 + 1);
    const [left, centre, right] = [bins[m - 1], bins[m], bins[m + 1]];
    for (let k = left; k < centre; k++) f[k] = (k - left) / (centre - left);
    for (let k = centre; k < right; k++) f[k] = (right - k) / (right - centre);
    filters.push(f);
  }
  return filters;
}

function hamming(n: number): Float64Array {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (n - 1));
  return w;
}

/** Orthonormal DCT-II, keeping the first nOut coefficients. */
function dct(input: Float64Array, nOut: number): Float64Array {
  const n = input.length;
  const out = new Float64Array(nOut);
  for (let k = 0; k < nOut; k++) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += input[i] * Math.cos((Math.PI * k * (2 * i + 1)) / (2 * n));
    out[k] = sum * (k === 0 ? Math.sqrt(1 / n) : Math.sqrt(2 / n));
  }
  return out;
}

const filterbanks = new Map<number, Float64Array[]>();
const window400 = hamming(FRAME);

/** One 13-coefficient vector per 10 ms of audio, mean-normalised over the utterance. */
export function mfcc(signal: Float32Array, sampleRate = 16000): Float64Array[] {
  let fb = filterbanks.get(sampleRate);
  if (!fb) {
    fb = melFilterbank(sampleRate);
    filterbanks.set(sampleRate, fb);
  }
  const frames: Float64Array[] = [];
  for (let start = 0; start + FRAME <= signal.length; start += HOP) {
    const re = new Float64Array(NFFT);
    const im = new Float64Array(NFFT);
    for (let i = 0; i < FRAME; i++) {
      const previous = start + i > 0 ? signal[start + i - 1] : 0;
      re[i] = (signal[start + i] - 0.97 * previous) * window400[i]; // pre-emphasis, then window
    }
    fft(re, im);
    const logMel = new Float64Array(NMELS);
    for (let m = 0; m < NMELS; m++) {
      const f = fb[m];
      let energy = 0;
      for (let k = 0; k < f.length; k++) energy += (f[k] * (re[k] * re[k] + im[k] * im[k])) / NFFT;
      logMel[m] = Math.log(Math.max(energy, 1e-10));
    }
    frames.push(dct(logMel, NCEPS));
  }
  return meanNormalise(frames);
}

function meanNormalise(frames: Float64Array[]): Float64Array[] {
  if (frames.length === 0) return frames;
  const mean = new Float64Array(frames[0].length);
  for (const f of frames) for (let i = 0; i < f.length; i++) mean[i] += f[i] / frames.length;
  return frames.map((f) => f.map((v, i) => v - mean[i]));
}
