import { useEffect, useRef, useState } from 'react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

// Week-1 check: does MediaPipe's face model run in this browser, how fast, and do the irises move
// sensibly inside each eye? Calibration to screen positions comes later.

const WASM_PATH = '/mediapipe/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

interface Point {
  x: number;
  y: number;
}

// Face-mesh indices with iris refinement: iris centre, eye corners and lids, per eye.
const RIGHT_EYE = { iris: 468, outer: 33, inner: 133, top: 159, bottom: 145 };
const LEFT_EYE = { iris: 473, outer: 263, inner: 362, top: 386, bottom: 374 };
type Eye = typeof RIGHT_EYE;

interface EyePosition {
  h: number; // 0 = outer corner, 1 = inner corner
  v: number; // 0 = upper lid, 1 = lower lid
}

function irisPosition(lm: Point[], eye: Eye): EyePosition {
  const iris = lm[eye.iris];
  const h = (iris.x - lm[eye.outer].x) / (lm[eye.inner].x - lm[eye.outer].x || 1e-6);
  const v = (iris.y - lm[eye.top].y) / (lm[eye.bottom].y - lm[eye.top].y || 1e-6);
  return { h, v };
}

function dot(ctx: CanvasRenderingContext2D, p: Point, w: number, h: number, colour: string, r: number): void {
  ctx.fillStyle = colour;
  ctx.beginPath();
  ctx.arc(p.x * w, p.y * h, r, 0, Math.PI * 2);
  ctx.fill();
}

export function GazeCheck() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cleanupRef = useRef<(() => void) | null>(null);
  const [status, setStatus] = useState('Press start and allow the camera.');
  const [running, setRunning] = useState(false);
  const [fps, setFps] = useState(0);
  const [delegate, setDelegate] = useState<string | null>(null);
  const [eyes, setEyes] = useState<{ right: EyePosition; left: EyePosition } | null>(null);

  useEffect(() => () => cleanupRef.current?.(), []);

  async function start() {
    try {
      setStatus('Loading the face model…');
      const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
      const create = (d: 'GPU' | 'CPU') =>
        FaceLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: d },
          runningMode: 'VIDEO',
          numFaces: 1,
        });
      let landmarker: FaceLandmarker;
      try {
        landmarker = await create('GPU');
        setDelegate('GPU');
      } catch {
        landmarker = await create('CPU');
        setDelegate('CPU');
      }

      setStatus('Starting the camera…');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      const video = videoRef.current!;
      video.srcObject = stream;
      await video.play();
      const canvas = canvasRef.current!;
      const ctx = canvas.getContext('2d')!;

      let raf = 0;
      let lastVideoTime = -1;
      let frames = 0;
      let windowStart = performance.now();
      let lastUi = 0;
      const loop = () => {
        raf = requestAnimationFrame(loop);
        if (video.readyState < 2 || video.currentTime === lastVideoTime) return;
        lastVideoTime = video.currentTime;
        const now = performance.now();
        const result = landmarker.detectForVideo(video, now);
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }
        const lm: Point[] | undefined = result.faceLandmarks[0];
        ctx.save();
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1); // mirror, like a selfie camera
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        if (lm) {
          for (const e of [RIGHT_EYE, LEFT_EYE]) {
            dot(ctx, lm[e.iris], canvas.width, canvas.height, '#00e5ff', 4);
            dot(ctx, lm[e.outer], canvas.width, canvas.height, '#ffea00', 2);
            dot(ctx, lm[e.inner], canvas.width, canvas.height, '#ffea00', 2);
          }
        }
        ctx.restore();

        frames++;
        if (now - windowStart >= 1000) {
          setFps(Math.round((frames * 1000) / (now - windowStart)));
          frames = 0;
          windowStart = now;
        }
        if (now - lastUi > 200) {
          lastUi = now;
          setEyes(lm ? { right: irisPosition(lm, RIGHT_EYE), left: irisPosition(lm, LEFT_EYE) } : null);
        }
      };
      raf = requestAnimationFrame(loop);
      cleanupRef.current = () => {
        cancelAnimationFrame(raf);
        stream.getTracks().forEach((t) => t.stop());
        landmarker.close();
      };
      setRunning(true);
      setStatus('Running. Look left, right, up and down, and watch the numbers follow.');
    } catch (e) {
      setStatus(`Error: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  function stop() {
    cleanupRef.current?.();
    cleanupRef.current = null;
    setRunning(false);
    setStatus('Stopped.');
  }

  const fmt = (n: number) => n.toFixed(2);

  return (
    <section className="check">
      <h1>Check 1: eye tracking</h1>
      <p>
        Checks that the face model runs in this browser and how fast. Try it on the laptop and on the tablet, and note the frame
        rate for each. Each iris should move smoothly between 0 and 1 as you look side to side.
      </p>
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
        <span>{status}</span>
      </div>
      <div className="stats">
        <span>Frames per second: {fps}</span>
        <span>Model runs on: {delegate ?? '…'}</span>
        <span>
          Right eye (h, v): {eyes ? `${fmt(eyes.right.h)}, ${fmt(eyes.right.v)}` : 'no face'}
        </span>
        <span>
          Left eye (h, v): {eyes ? `${fmt(eyes.left.h)}, ${fmt(eyes.left.v)}` : 'no face'}
        </span>
      </div>
      <video ref={videoRef} className="hidden-video" playsInline muted />
      <canvas ref={canvasRef} className="camera" />
      <p className="device">{navigator.userAgent}</p>
    </section>
  );
}
