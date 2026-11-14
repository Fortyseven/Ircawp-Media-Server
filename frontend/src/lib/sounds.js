// Short audio cues for generation outcomes.
//
// Drop the real files into frontend/public/sounds/ — either format works,
// the first candidate that exists is used:
//   success.mp3  or  success.wav
//   error.mp3    or  error.wav
//
// Until the files exist, playCue() is a harmless no-op.

const CUES = {
    success: ["/sounds/success.mp3", "/sounds/success.wav"],
    error: ["/sounds/error.mp3", "/sounds/error.wav"],
};

const cache = new Map();

// Reuse one <audio> per candidate so each file is fetched only once.
function audioFor(url) {
    if (!cache.has(url)) cache.set(url, new Audio(url));
    return cache.get(url);
}

// Play a cue ("success" | "error"), falling back across candidate files
// (mp3 → wav) if the first is missing. Failures (missing file, blocked
// autoplay) are swallowed so they never surface as an error.
export function playCue(kind) {
    const urls = CUES[kind];
    if (!urls) return;

    const tryIndex = (i) => {
        if (i >= urls.length) return;
        const audio = audioFor(urls[i]);
        audio.currentTime = 0;
        audio.play()?.catch(() => tryIndex(i + 1));
    };

    tryIndex(0);
}
