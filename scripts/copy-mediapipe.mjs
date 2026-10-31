// Copies MediaPipe's WebAssembly files into public/ so the app serves them itself:
// always the same version as the installed package, and no CDN needed at runtime.
import { cpSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const src = join(root, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
const dst = join(root, 'public', 'mediapipe', 'wasm');

if (!existsSync(src)) {
  console.error(`MediaPipe wasm folder not found at ${src}`);
  process.exit(1);
}
mkdirSync(dst, { recursive: true });
cpSync(src, dst, { recursive: true });
console.log(`Copied MediaPipe wasm to ${dst}`);
