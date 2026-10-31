// The user's own recordings of each word ("templates"), used to recognise how they say it. Kept in
// IndexedDB as raw 16 kHz audio, so the features can be recomputed if the method changes.

const DB_NAME = 'look-and-say-templates';
const STORE = 'templates';

export type TemplateSource = 'taught' | 'learned';

export interface StoredTemplate {
  id: number;
  key: string; // the grid cell the recording belongs to
  source: TemplateSource; // recorded on purpose, or learned from a confirmed choice
  audio: Float32Array;
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE, { keyPath: 'id', autoIncrement: true });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function withStore<T>(mode: IDBTransactionMode, run: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await openDb();
  try {
    return await new Promise<T>((resolve, reject) => {
      const request = run(db.transaction(STORE, mode).objectStore(STORE));
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  } finally {
    db.close();
  }
}

export async function loadTemplates(): Promise<StoredTemplate[]> {
  try {
    return (await withStore('readonly', (s) => s.getAll())) as StoredTemplate[];
  } catch {
    return []; // IndexedDB unavailable (private mode): templates last for this visit only
  }
}

/** Saves a recording and returns its id, or null when it could not be saved. */
export async function addTemplate(key: string, audio: Float32Array, source: TemplateSource): Promise<number | null> {
  try {
    return Number(await withStore('readwrite', (s) => s.add({ key, source, audio })));
  } catch {
    return null;
  }
}

export async function deleteTemplate(id: number): Promise<void> {
  try {
    await withStore('readwrite', (s) => s.delete(id));
  } catch {
    // Already gone or storage unavailable.
  }
}
