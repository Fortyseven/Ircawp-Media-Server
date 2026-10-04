export const DEFAULT_SIZE = "1024x1024";
export const MATCH_SOURCE = "match-source";
export const DEFAULT_ASPECT_RATIO = "1:1";
export const DEFAULT_OUTPUT_SIZE = 1024;

const aspectRatioGroups = [
    {
        label: "Square",
        options: [{ label: "1:1", value: "1:1" }],
    },
    {
        label: "Landscape",
        options: ["8:1", "4:1", "21:9", "16:9", "4:3", "3:2"].map((value) => ({
            label: value,
            value,
        })),
    },
    {
        label: "Portrait",
        options: ["1:8", "1:4", "9:21", "9:16", "3:4", "2:3"].map((value) => ({
            label: value,
            value,
        })),
    },
];

export function getAspectRatioGroups(hasSourceImage = false) {
    if (hasSourceImage) {
        return [
            {
                label: "Source",
                options: [{ label: "Match source", value: MATCH_SOURCE }],
            },
            ...aspectRatioGroups,
        ];
    }
    return aspectRatioGroups;
}

export function dimensionsForAspect(aspectRatio, outputSize) {
    if (aspectRatio === MATCH_SOURCE) return undefined;

    const [ratioWidth, ratioHeight] = aspectRatio.split(":").map(Number);
    const edge = Number(outputSize);
    if (!ratioWidth || !ratioHeight || !edge) return DEFAULT_SIZE;

    const landscape = ratioWidth >= ratioHeight;
    const width = landscape
        ? edge
        : Math.max(16, Math.round((edge * ratioWidth) / ratioHeight / 16) * 16);
    const height = landscape
        ? Math.max(16, Math.round((edge * ratioHeight) / ratioWidth / 16) * 16)
        : edge;
    return `${width}x${height}`;
}
