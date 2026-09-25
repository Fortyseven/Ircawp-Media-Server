const KEY = "ircawp-media-templates";

/** Marker spliced with the prompt box text inside a template body. */
export const PROMPT_PLACEHOLDER = "{prompt}";

/** Starter templates, seeded on first run (deletable by the user). */
export const DEFAULT_TEMPLATES = [
    {
        name: "comic book",
        body:
            "Transform this photo into a comic book illustration: bold ink outlines, cel shading, halftone textures. " +
            PROMPT_PLACEHOLDER,
    },
    {
        name: "photo restoration",
        body:
            "remaster this image into a modern, realistic, natural color photograph; a clear, focused, raw DSLR photograph; it should be a masterpiece of cinematography, deblur the image, reduce it's film noise, color correct, remove haze, masterpiece of cinematography, preserve composition and preserve aspect ratio; preserve clothing and facial structure\n" +
            PROMPT_PLACEHOLDER,
    },
];

/**
 * Resolve the storage backend: injected argument wins (tests), otherwise the
 * real localStorage when available, otherwise null (SSR / unsupported env).
 */
function resolveStorage(storage) {
    try {
        if (storage) return storage;
        return typeof localStorage !== "undefined" ? localStorage : null;
    } catch {
        return null;
    }
}

function isValidTemplate(t) {
    return (
        t &&
        typeof t === "object" &&
        typeof t.name === "string" &&
        t.name.trim() !== "" &&
        typeof t.body === "string" &&
        t.body.trim() !== ""
    );
}

/** Parse a stored value; returns null when the shape is wrong. */
function parseStored(raw) {
    try {
        const parsed = JSON.parse(raw);
        const list = parsed?.templates;
        if (!Array.isArray(list)) return null;
        return list
            .filter(isValidTemplate)
            .map((t) => ({ name: t.name.trim(), body: t.body.trim() }));
    } catch {
        return null;
    }
}

/**
 * Load the template list. When the key has never been set, seed the defaults
 * and persist them. Invalid or partially-invalid stored data degrades to the
 * valid subset (or an empty list) instead of throwing.
 */
export function loadTemplates(storage) {
    const s = resolveStorage(storage);
    if (!s) return DEFAULT_TEMPLATES.map((t) => ({ ...t }));
    const raw = s.getItem(KEY);
    if (raw === null) {
        saveTemplates(DEFAULT_TEMPLATES, s);
        return DEFAULT_TEMPLATES.map((t) => ({ ...t }));
    }
    return parseStored(raw) ?? [];
}

/**
 * Persist the template list. Invalid entries (missing/blank name or body) are
 * dropped; names are trimmed. Writes to the resolved storage directly (not
 * via storage.js) so an injected storage is honored.
 */
export function saveTemplates(list, storage) {
    const s = resolveStorage(storage);
    if (!s) return;
    try {
        s.setItem(KEY, JSON.stringify({ templates: normalizeList(list) }));
    } catch {
        // Storage unavailable (e.g. quota exceeded) — ignore, like storage.js.
    }
}

/**
 * Build the prompt actually sent to the server:
 * - template has {prompt} → box text substituted at the first occurrence
 * - no placeholder → the template body alone (box text ignored)
 * - no template → the trimmed box text (unchanged behavior)
 */
export function applyTemplate(template, promptText) {
    const body = (template?.body ?? "").trim();
    const text = (promptText ?? "").trim();
    if (!body) return text;
    if (!body.includes(PROMPT_PLACEHOLDER)) return body;
    return body.replace(PROMPT_PLACEHOLDER, text).trim();
}

/**
 * A copy of the list sorted by name (case-insensitive), for display in the
 * dialog list and the dropdown. Storage order is left untouched.
 */
export function sortedTemplates(list) {
    return [...list].sort((a, b) =>
        a.name.localeCompare(b.name, undefined, { sensitivity: "base" }),
    );
}

/**
 * Minimal RFC 4180 CSV parser: returns an array of rows (string arrays).
 * Handles quoted fields, doubled embedded quotes, and fields containing
 * commas/newlines. A quote only opens a quoted field at the start of the
 * field; stray quotes elsewhere are kept literally. Throws on an
 * unterminated quoted field.
 */
export function parseCsv(text) {
    const rows = [];
    let row = [];
    let field = "";
    let inQuotes = false;
    let i = 0;
    const n = text.length;
    const pushField = () => {
        row.push(field);
        field = "";
    };
    const pushRow = () => {
        pushField();
        rows.push(row);
        row = [];
    };

    while (i < n) {
        const ch = text[i];
        if (inQuotes) {
            if (ch === '"') {
                if (text[i + 1] === '"') {
                    field += '"';
                    i += 2;
                    continue;
                }
                inQuotes = false;
                i++;
                continue;
            }
            field += ch;
            i++;
            continue;
        }
        if (ch === '"' && field === "") {
            inQuotes = true;
            i++;
            continue;
        }
        if (ch === ",") {
            pushField();
            i++;
            continue;
        }
        if (ch === "\r") {
            if (text[i + 1] === "\n") i++;
            pushRow();
            i++;
            continue;
        }
        if (ch === "\n") {
            pushRow();
            i++;
            continue;
        }
        field += ch;
        i++;
    }
    if (inQuotes) throw new Error("unterminated quote in CSV file");
    if (field !== "" || row.length > 0) pushRow();
    return rows;
}

/** Quote a CSV field only when it contains a comma, quote, or newline. */
function csvEscape(value) {
    if (/[",\n\r]/.test(value)) {
        return '"' + value.replace(/"/g, '""') + '"';
    }
    return value;
}

/** Serialize rows as RFC 4180 CSV with CRLF line endings and a trailing newline. */
export function serializeCsv(rows) {
    if (!rows.length) return "";
    return rows.map((r) => r.map(csvEscape).join(",")).join("\r\n") + "\r\n";
}

function normalizeList(list) {
    return (Array.isArray(list) ? list : [])
        .filter(isValidTemplate)
        .map((t) => ({ name: t.name.trim(), body: t.body.trim() }));
}

/**
 * Serialize the list as the backup CSV file.
 * Format (see prompt-templates.csv): header `name,prompt,negative_prompt`.
 * We do not track negative prompts yet, so that column is always exported
 * empty — the column is kept so files stay compatible with the format.
 */
export function exportTemplatesCsv(list) {
    const rows = [
        ["name", "prompt", "negative_prompt"],
        ...normalizeList(list).map((t) => [t.name, t.body, ""]),
    ];
    return serializeCsv(rows);
}

/**
 * Parse and validate a backup CSV file. Returns the template list, or throws
 * an Error with a human-readable reason (malformed CSV, blank name/prompt,
 * duplicate names) that the UI can display. A leading header row (first cell
 * "name") is skipped when present; the negative_prompt column is accepted
 * and ignored. Blank lines are skipped.
 */
export function importTemplatesCsv(text) {
    let rows;
    try {
        rows = parseCsv(String(text ?? "").replace(/^\uFEFF/, ""));
    } catch (e) {
        throw new Error(e.message);
    }
    const nonEmpty = rows.filter((r) => r.some((c) => c.trim() !== ""));
    if (nonEmpty.length === 0) return [];

    // Header detection: first cell is literally "name".
    const dataRows =
        nonEmpty[0][0]?.trim().toLowerCase() === "name"
            ? nonEmpty.slice(1)
            : nonEmpty;
    if (dataRows.length === 0) return [];

    const seen = new Set();
    const result = [];
    dataRows.forEach((cells, idx) => {
        const name = (cells[0] ?? "").trim();
        const body = (cells[1] ?? "").trim();
        if (!name || !body) {
            throw new Error(`entry ${idx + 1} has a blank name or prompt`);
        }
        if (seen.has(name)) {
            throw new Error(`duplicate template name "${name}"`);
        }
        seen.add(name);
        result.push({ name, body });
    });
    return result;
}
