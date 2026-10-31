import { useEffect, useMemo, useRef, useState } from 'react';
import { startMic, TARGET_RATE, UtteranceDetector, type MicHandle } from '../audio/capture';
import { LevelMeter, useLevels } from '../components/LevelMeter';
import { ranked } from '../fusion/fuse';
import { cells } from '../vocab';
import { VoiceModel, wordFeatures } from '../voice/voiceModel';

// Week-1 check: record five words three times each, then test whether the user's own recordings
// recognise them (MFCC + DTW). The confusion matrix shows which words get mixed up.

const WORD_IDS = ['water', 'juice', 'mum', 'dad', 'ball'];
const PER_WORD = 3;

type Mode = { kind: 'idle' } | { kind: 'enrol'; word: string } | { kind: 'test'; word: string };

interface Guess {
  word: string;
  probability: number;
  distance: number;
}

interface Result {
  truth: string;
  ranking: Guess[];
}

export function TemplateCheck() {
  const words = useMemo(() => cells().filter((c) => WORD_IDS.includes(c.wordId)), []);
  const name = (id: string) => {
    const c = words.find((w) => w.wordId === id);
    return c ? `${c.lb} (${c.en})` : id;
  };

  const model = useRef(new VoiceModel());
  const nextId = useRef(1);
  const [counts, setCounts] = useState<Map<string, number>>(() => new Map());
  const [mode, setMode] = useState<Mode>({ kind: 'idle' });
  const modeRef = useRef<Mode>({ kind: 'idle' });
  const [results, setResults] = useState<Result[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastHeard, setLastHeard] = useState<string | null>(null);
  const micRef = useRef<MicHandle | null>(null);
  const detectorRef = useRef<UtteranceDetector | null>(null);
  const levels = useLevels(detectorRef, running);

  useEffect(() => () => micRef.current?.stop(), []);

  const changeMode = (m: Mode) => {
    modeRef.current = m;
    setMode(m);
  };

  // Called by the detector for every word heard; refs keep it current across renders.
  function onUtterance(audio: Float32Array) {
    const m = modeRef.current;
    const seconds = (audio.length / TARGET_RATE).toFixed(2);
    setLastHeard(m.kind === 'idle' ? `Heard a ${seconds} s sound (not recording right now)` : `Heard a ${seconds} s word`);
    if (m.kind === 'idle') return;
    const feats = wordFeatures(audio);
    if (m.kind === 'enrol') {
      model.current.add({ id: nextId.current++, key: m.word, source: 'taught', feats, audio });
      setCounts(new Map(WORD_IDS.map((id): [string, number] => [id, model.current.count(id)])));
    } else {
      const r = model.current.recognise(feats, WORD_IDS);
      const ranking = r
        ? ranked(r.dist).map(([word, probability]) => ({ word, probability, distance: r.distances.get(word) ?? Number.NaN }))
        : [];
      setResults((rs) => [{ truth: m.word, ranking }, ...rs]);
    }
    changeMode({ kind: 'idle' });
  }

  async function start() {
    setError(null);
    const detector = new UtteranceDetector(onUtterance);
    detectorRef.current = detector;
    try {
      micRef.current = await startMic((chunk) => detector.push(chunk));
      setRunning(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  function stop() {
    micRef.current?.stop();
    micRef.current = null;
    setRunning(false);
    changeMode({ kind: 'idle' });
  }

  const count = (id: string) => counts.get(id) ?? 0;
  const ready = WORD_IDS.every((id) => count(id) > 0);

  function nextTest() {
    const word = WORD_IDS[Math.floor(Math.random() * WORD_IDS.length)];
    changeMode({ kind: 'test', word });
  }

  const correct = results.filter((r) => r.ranking[0]?.word === r.truth).length;
  const matrix = (truth: string, guess: string) =>
    results.filter((r) => r.truth === truth && r.ranking[0]?.word === guess).length;

  return (
    <section className="check">
      <h1>Check 3: voice templates</h1>
      <p>
        Start the microphone and stay quiet for two seconds so the app can measure the room. Then record each word {PER_WORD}{' '}
        times in your own way: press Record, then say the word once. Finally test: the app shows a word, you say it, and it
        guesses from your recordings alone.
      </p>
      <div className="row">
        {running ? (
          <button type="button" onClick={stop}>
            Stop microphone
          </button>
        ) : (
          <button type="button" onClick={() => void start()}>
            Start microphone
          </button>
        )}
      </div>
      {error && <p className="error">{error}</p>}
      {running && <LevelMeter {...levels} />}
      {lastHeard && <p className="heard">{lastHeard}</p>}

      <h2>1. Record</h2>
      <table className="table">
        <tbody>
          {WORD_IDS.map((id) => (
            <tr key={id}>
              <td>{name(id)}</td>
              <td>
                {count(id)} / {PER_WORD}
              </td>
              <td>
                <button
                  type="button"
                  disabled={!running || mode.kind !== 'idle' || count(id) >= PER_WORD}
                  onClick={() => changeMode({ kind: 'enrol', word: id })}
                >
                  Record
                </button>{' '}
                {mode.kind === 'enrol' && mode.word === id && <span className="listening">Listening… say it now</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2>2. Test</h2>
      <div className="row">
        <button type="button" disabled={!running || !ready || mode.kind !== 'idle'} onClick={nextTest}>
          Next test word
        </button>
        {mode.kind === 'test' && (
          <span className="prompt">
            Say: <strong>{name(mode.word)}</strong>
          </span>
        )}
      </div>
      {results[0] && (
        <p>
          Last guess:{' '}
          {results[0].ranking
            .slice(0, 3)
            .map((r) => `${name(r.word)} ${(r.probability * 100).toFixed(0)}% (distance ${r.distance.toFixed(2)})`)
            .join(' · ')}
        </p>
      )}

      <h2>
        Results: {correct} of {results.length} correct
      </h2>
      {results.length > 0 && (
        <table className="table confusion">
          <thead>
            <tr>
              <th>Said \ guessed</th>
              {WORD_IDS.map((id) => (
                <th key={id}>{name(id)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {WORD_IDS.map((truth) => (
              <tr key={truth}>
                <th>{name(truth)}</th>
                {WORD_IDS.map((guess) => (
                  <td key={guess} className={truth === guess ? 'diag' : undefined}>
                    {matrix(truth, guess) || ''}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
