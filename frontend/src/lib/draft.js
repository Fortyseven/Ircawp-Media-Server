import { loadStoredObject, saveStoredObject } from "./storage.js";
import { addDataUrl } from "./media-bin.svelte.js";

const KEY = "ircawp-media-draft";

export function loadDraft() {
    return loadStoredObject(KEY);
}

export function saveDraft(draft) {
    saveStoredObject(KEY, draft);
}

// Legacy drafts (pre-media-bin) stored raw data URLs in `images`.
export function hasLegacyImages(draft = loadDraft()) {
    return (
        Array.isArray(draft.images) &&
        draft.images.some((i) => typeof i === "string")
    );
}

// One-shot migration: import legacy blobs into the media bin (deduped) and
// rewrite the draft to pointer entries. Must run after initMediaBin().
// Returns true if anything was migrated.
export function migrateDraft() {
    const draft = loadDraft();
    if (!hasLegacyImages(draft)) return false;
    const images = draft.images.map((image) =>
        typeof image === "string" ? { binId: addDataUrl(image).id } : image,
    );
    saveDraft({ ...draft, images });
    return true;
}
