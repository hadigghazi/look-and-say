import { useEffect, useRef, useState } from 'react';
import { playAudio, startMic, TARGET_RATE, UtteranceDetector, type MicHandle } from '../audio/capture';
import { LevelMeter, useLevels } from '../components/LevelMeter';

// Week-1 check: does the browser give us the microphone, and does speech detection cut single words cleanly?

interface Clip {
  id: number;
  audio: Float32Array;
}

export function MicCheck() {
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [clips, setClips] = useState<Clip[]>([]);
  const [inputRate, setInputRate] = useState<number | null>(null);
  const micRef = useRef<MicHandle | null>(null);
  const detectorRef = useRef<UtteranceDetector | null>(null);
  const nextId = useRef(1);
  const levels = useLevels(detectorRef, running);

  useEffect(() => () => micRef.current?.stop(), []);

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
      <p>
        Press start, stay quiet for two seconds so the app can measure the room, then say single words with a short pause
        between them. Each detected word appears below; play it back to check the recording. The bar turns green while your
        voice is above the orange line.
      </p>
      <div className="row">
        {running ? (
          <button type="button" onClick={stop}>
            Stop
          </button>
        ) : (
          <button type="button" onClick={() => void start()}>
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
      {running && <LevelMeter {...levels} />}
      <ul className="clips">
        {clips.map((c) => (
          <li key={c.id}>
            Word {c.id}: {(c.audio.length / TARGET_RATE).toFixed(2)} s{' '}
            <button type="button" onClick={() => void playAudio(c.audio)}>
              Play
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
