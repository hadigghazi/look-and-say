// Recorded words, kept in the browser's IndexedDB so they survive reloads and work offline.
// Keyed by the exact label spoken, so Lebanese, MSA and feminine forms each get their own recording.

const DB_NAME = 'look-and-say';
const STORE = 'voice';

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function loadRecordings(): Promise<Map<string, Float32Array>> {
  const recordings = new Map<string, Float32Array>();
  try {
    const db = await openDb();
    await new Promise<void>((resolve, reject) => {
      const request = db.transaction(STORE, 'readonly').objectStore(STORE).openCursor();
      request.onsuccess = () => {
        const cursor = request.result;
        if (cursor) {
          recordings.set(String(cursor.key), cursor.value as Float32Array);
          cursor.continue();
        } else {
          resolve();
        }
      };
      request.onerror = () => reject(request.error);
    });
    db.close();
  } catch {
    // IndexedDB unavailable (private mode): recordings then last for this visit only.
  }
  return recordings;
}

export async function saveRecording(key: string, audio: Float32Array): Promise<void> {
  try {
    const db = await openDb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, 'readwrite');
      tx.objectStore(STORE).put(audio, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
    db.close();
  } catch {
    // Not saved; the in-memory copy still plays for this visit.
  }
}
