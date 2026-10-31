import { useEffect, useRef, useState } from 'react';
import { playAudio, startMic, TARGET_RATE, UtteranceDetector, type MicHandle } from '../audio/capture';

// Week-1 check: does the browser give us the microphone, and does speech detection cut single words cleanly?

interface Clip {
  id: number;
  audio: Float32Array;
}

export function MicCheck() {
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [level, setLevel] = useState(0);
  const [clips, setClips] = useState<Clip[]>([]);
  const [inputRate, setInputRate] = useState<number | null>(null);
  const micRef = useRef<MicHandle | null>(null);
  const detectorRef = useRef<UtteranceDetector | null>(null);
  const nextId = useRef(1);

  useEffect(() => () => micRef.current?.stop(), []);

  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => setLevel(detectorRef.current?.level ?? 0), 100);
    return () => window.clearInterval(timer);
  }, [running]);

  async function start() {
    setError(null);
    const detector = new UtteranceDetector((audio) => {
      const id = nextId.current++;
      setClips((c) => [{ id, audio }, ...c].slice(0, 10));
    });
    detectorRef.current = detector;
    try {
      const mic = await startMic((chunk) => detector.push(chunk));
      micRef.current = mic;
      setInputRate(mic.inputRate);
      setRunning(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  function stop() {
    micRef.current?.stop();
    micRef.current = null;
    setRunning(false);
  }

  return (
    <section className="check">
      <h1>Check 2: microphone and speech detection</h1>
      <p>Press start, then say single words with a pause between them. Each detected word appears below; play it back to check the recording.</p>
      <div className="row">
        {running ? (
          <button type="button" onClick={stop}>
            Stop
          </button>
        ) : (
          <button type="button" onClick={start}>
            Start
          </button>
        )}
        {inputRate !== null && (
          <span>
            Microphone at {inputRate} Hz, analysed at {TARGET_RATE} Hz
          </span>
        )}
      </div>
      {error && <p className="error">{error}</p>}
      <div className="meter" aria-label="Microphone level">
        <div className="meter-fill" style={{ width: `${Math.min(100, (level / 0.1) * 100)}%` }} />
      </div>
      <ul className="clips">
        {clips.map((c) => (
          <li key={c.id}>
            Word {c.id}: {(c.audio.length / TARGET_RATE).toFixed(2)} s{' '}
            <button type="button" onClick={() => playAudio(c.audio)}>
              Play
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
