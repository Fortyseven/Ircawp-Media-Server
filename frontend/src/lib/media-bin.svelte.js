import { openDB, MEDIA_BIN_STORE } from "./db.js";

// ── media bin ─────────────────────────────────────────────────
// Single source of truth for media blobs. Tray entries are pointers
// ({ binId }) into this store; data URLs are resolved at the edges
// (thumbnails, API calls). One physical blob per piece of media.

let entries = $state([]); // newest first
const byId = new Map(); // id → entry
const hashIndex = new Map(); // content hash → id
let ready = $state(false);
let trayIds = $state([]); // ids currently referenced by the editing tray
let initError = $state("");

let initPromise = null;

function toPromise(req) {
    return new Promise((resolve, reject) => {
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

// 64-bit content hash: two FNV-1a 32-bit runs with different seeds in a
// single pass. Not cryptographic — just needs to be stable and collision-
// free for the scale of user media in the bin.
function contentHash(b64) {
    let lo = 0x811c9dc5;
    let hi = 0x01000193;
    for (let i = 0; i < b64.length; i++) {
        const c = b64.charCodeAt(i);
        lo ^= c;
        lo = Math.imul(lo, 0x01000193) >>> 0;
        hi ^= c;
        hi = Math.imul(hi, 0x01000193) >>> 0;
    }
    return (
        lo.toString(16).padStart(8, "0") +
        hi.toString(16).padStart(8, "0")
    );
}

function mimeFromDataUrl(dataUrl) {
    const end = dataUrl.indexOf(";");
    return dataUrl.slice(5, end > 5 ? end : undefined) || "image";
}

// Load bin entries from IndexedDB and rebuild the in-memory indexes.
// Idempotent: safe to call on every boot. On failure the cached promise is
// cleared so a later call retries (e.g. after the blocking tab closes).
export function initMediaBin() {
    if (!initPromise) {
        initPromise = (async () => {
            try {
                const db = await openDB();
                const store = db
                    .transaction(MEDIA_BIN_STORE, "readonly")
                    .objectStore(MEDIA_BIN_STORE);
                const all = await toPromise(store.getAll());
                for (const entry of all) {
                    byId.set(entry.id, entry);
                    hashIndex.set(entry.hash, entry.id);
                }
                entries = all.sort((a, b) => b.addedAt - a.addedAt);
                ready = true;
            } catch (e) {
                initError = e.message;
                throw e;
            }
        })();
        initPromise.catch(() => {
            initPromise = null;
        });
    }
    return initPromise;
}

// Exposed for the UI: non-empty when the last init attempt failed.
export function getInitError() {
    return initError;
}

// Retry a failed init (e.g. after the user closes the stale tab).
export function retryInit() {
    initError = "";
    return initMediaBin();
}

export function isReady() {
    return ready;
}


export function getEntries() {
    return entries;
}

export function getEntry(id) {
    // Read `entries` (reactive state) so template calls track bin
    // mutations; the Map itself keeps the lookup O(1) but isn't reactive.
    entries;
    return byId.get(id) ?? null;
}

export function has(id) {
    return byId.has(id);
}

// Add a media blob, deduped by content hash. Returns { id, added }:
// existing content reuses its entry instead of creating a copy.
export function addDataUrl(dataUrl, mime) {
    if (!dataUrl || !dataUrl.startsWith("data:")) {
        throw new Error("addDataUrl: expected a data URL");
    }
    const b64 = dataUrl.slice(dataUrl.indexOf(",") + 1);
    const hash = contentHash(b64);
    const existing = hashIndex.get(hash);
    if (existing) return { id: existing, added: false };

    const entry = {
        id: crypto.randomUUID(),
        dataUrl,
        hash,
        mime: mime || mimeFromDataUrl(dataUrl),
        addedAt: Math.floor(Date.now() / 1000),
    };
    byId.set(entry.id, entry);
    hashIndex.set(hash, entry.id);
    entries = [entry, ...entries];
    persist(entry);
    return { id: entry.id, added: true };
}

function readFile(file) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => resolve(null);
        reader.readAsDataURL(file);
    });
}

// Ingest dropped/pasted/browsed files (image types only).
export async function addFiles(files) {
    const imageFiles = [...files].filter((file) =>
        file.type.startsWith("image/"),
    );
    const loaded = (await Promise.all(imageFiles.map(readFile))).filter(Boolean);
    for (const dataUrl of loaded) addDataUrl(dataUrl);
    return loaded.length;
}

// Import a history image (raw b64) as a bin copy. Deliberately a copy:
// clearing history must not lose bin entries.
export function importHistoryImage(b64Json) {
    return addDataUrl(`data:image/png;base64,${b64Json}`, "image/png");
}

function persist(entry) {
    openDB()
        .then((db) =>
            toPromise(
                db
                    .transaction(MEDIA_BIN_STORE, "readwrite")
                    .objectStore(MEDIA_BIN_STORE)
                    .put(entry),
            ),
        )
        .catch((e) => console.error("media bin persist failed", e));
}

export async function remove(id) {
    const entry = byId.get(id);
    if (!entry) return;
    byId.delete(id);
    hashIndex.delete(entry.hash);
    entries = entries.filter((e) => e.id !== id);
    const db = await openDB();
    await toPromise(
        db
            .transaction(MEDIA_BIN_STORE, "readwrite")
            .objectStore(MEDIA_BIN_STORE)
            .delete(id),
    );
}

// Clear the bin except for the given ids (e.g. entries still referenced
// by the editing tray, which are protected from deletion).
export async function clear(excludeIds = new Set()) {
    const kept = [...byId.values()].filter((e) => excludeIds.has(e.id));
    byId.clear();
    hashIndex.clear();
    for (const entry of kept) {
        byId.set(entry.id, entry);
        hashIndex.set(entry.hash, entry.id);
    }
    entries = kept.sort((a, b) => b.addedAt - a.addedAt);
    const db = await openDB();
    const store = db
        .transaction(MEDIA_BIN_STORE, "readwrite")
        .objectStore(MEDIA_BIN_STORE);
    await toPromise(store.clear());
    for (const entry of kept) {
        await toPromise(store.put(entry));
    }
}

// Resolve tray pointers to data URLs for API calls / rendering.
// Missing ids (deleted before resolution) are skipped.
export function resolve(ids) {
    return ids.map((id) => byId.get(id)?.dataUrl).filter(Boolean);
}

// Tray registry: the editing tray publishes its current bin ids so the
// bin grid can badge "in queue" frames without sharing component state.
export function setTrayIds(ids) {
    trayIds = [...ids];
}

export function getTrayIds() {
    return trayIds;
}
