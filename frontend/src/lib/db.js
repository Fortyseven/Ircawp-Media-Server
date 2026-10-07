const DB_NAME = "ircawp-media";
const DB_VERSION = 2;
const STORE = "history";
export const MEDIA_BIN_STORE = "media_bin";

export const MAX_HISTORY = 50;

// A version upgrade waits for every lower-version connection (an old tab
// of this app, or a page from before the update) to close. Cap that wait
// so the UI can't stall on a tab that will never release.
const OPEN_TIMEOUT_MS = 5000;

let dbPromise = null;

export function openDB() {
    if (!dbPromise) {
        dbPromise = new Promise((resolve, reject) => {
            const req = indexedDB.open(DB_NAME, DB_VERSION);
            let settled = false;
            req.onupgradeneeded = () => {
                const db = req.result;
                if (!db.objectStoreNames.contains(STORE)) {
                    db.createObjectStore(STORE, {
                        keyPath: "id",
                        autoIncrement: true,
                    });
                }
                if (!db.objectStoreNames.contains(MEDIA_BIN_STORE)) {
                    db.createObjectStore(MEDIA_BIN_STORE, { keyPath: "id" });
                }
            };
            req.onsuccess = () => {
                if (settled) {
                    // The request completed after we timed out — drop the
                    // connection rather than holding a DB nobody uses.
                    req.result.close();
                    return;
                }
                settled = true;
                const db = req.result;
                // Yield to a newer-version request from elsewhere so the
                // upgrade can proceed.
                db.onversionchange = () => db.close();
                resolve(db);
            };
            req.onerror = () => {
                if (settled) return;
                settled = true;
                dbPromise = null; // allow a later retry
                reject(req.error);
            };
            setTimeout(() => {
                if (settled) return;
                settled = true;
                dbPromise = null; // allow a later retry
                reject(
                    new Error(
                        "storage open timed out — another tab may be holding an old version of the app. Close other tabs and reload.",
                    ),
                );
            }, OPEN_TIMEOUT_MS);
        });
    }
    return dbPromise;
}

function store(db, mode) {
    return db.transaction(STORE, mode).objectStore(STORE);
}

function toPromise(req) {
    return new Promise((resolve, reject) => {
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

async function prune(db) {
    const s = store(db, "readwrite");
    const count = await toPromise(s.count());
    if (count <= MAX_HISTORY) return;
    const oldestKeys = await toPromise(s.getAllKeys(null, count - MAX_HISTORY));
    for (const key of oldestKeys) s.delete(key);
}

export async function addGeneration(record) {
    const db = await openDB();
    const id = await toPromise(store(db, "readwrite").add(record));
    await prune(db);
    return id;
}

export async function getGenerations() {
    const db = await openDB();
    const all = await toPromise(store(db, "readonly").getAll());
    return all.sort((a, b) => b.id - a.id);
}

export async function deleteGeneration(id) {
    const db = await openDB();
    await toPromise(store(db, "readwrite").delete(id));
}

export async function clearHistory() {
    const db = await openDB();
    await toPromise(store(db, "readwrite").clear());
}
