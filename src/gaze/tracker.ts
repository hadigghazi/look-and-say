import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { faceFeatures, type FaceFeatures } from './features';

const WASM_PATH = '/mediapipe/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

export type FrameListener = (features: FaceFeatures | null, timeMs: number) => void;

/** Runs the webcam and MediaPipe's face model, and hands gaze features for every new frame to listeners. */
export class GazeTracker {
  private landmarker: FaceLandmarker | null = null;
  private stream: MediaStream | null = null;
  private readonly video: HTMLVideoElement;
  private raf = 0;
  private lastVideoTime = -1;
  private readonly listeners = new Set<FrameListener>();

  constructor() {
    this.video = document.createElement('video');
    this.video.playsInline = true;
    this.video.muted = true;
    this.video.className = 'hidden-video'; // in the page (some browsers pause detached videos) but invisible
  }

  async start(): Promise<void> {
    const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
    const create = (delegate: 'GPU' | 'CPU') =>
      FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate },
        runningMode: 'VIDEO',
        numFaces: 1,
        outputFacialTransformationMatrixes: true, // real head rotation and position
      });
    try {
      this.landmarker = await create('GPU');
    } catch {
      this.landmarker = await create('CPU');
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, // more pixels per eye
      audio: false,
    });
    document.body.appendChild(this.video);
    this.video.srcObject = this.stream;
    await this.video.play();

    const loop = () => {
      this.raf = requestAnimationFrame(loop);
      const landmarker = this.landmarker;
      if (!landmarker || this.video.readyState < 2 || this.video.currentTime === this.lastVideoTime) return;
      this.lastVideoTime = this.video.currentTime;
      const now = performance.now();
      const result = landmarker.detectForVideo(this.video, now);
      const landmarks = result.faceLandmarks[0];
      const matrix = result.facialTransformationMatrixes?.[0]?.data;
      const features = landmarks ? faceFeatures(landmarks, matrix) : null;
      this.listeners.forEach((listener) => listener(features, now));
    };
    this.raf = requestAnimationFrame(loop);
  }

  subscribe(listener: FrameListener): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  stop(): void {
    cancelAnimationFrame(this.raf);
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.video.remove();
    this.landmarker?.close();
    this.landmarker = null;
    this.listeners.clear();
  }
}
