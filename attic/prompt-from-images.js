import { chatCompletion } from "./prompt-rewrite.js";

// System prompt for the "prompt from images" button: the model receives only
// the attached edit images (no instruction text) and writes the prompt that
// lands in the prompt box. Replace this const to change that behavior.
export const PROMPT_FROM_IMAGES_SYSTEM_PROMPT = `You are an expert image description AI. Analyze the images supplied by the user and provide a detailed prompt for it. Write an extensive, complex in-depth multi-paragraph description that describes only what is clearly visible: the main subject(s), key objects, setting, spatial relationships, colors/materials, lighting, style, and overall mood. Keep it factual and coherent. Prioritize the subject's visible identity cues: gender presentation, face and expression, hairstyle and hair color, distinctive accessories, body pose, outfit details (materials, layers, patterns), and any signature traits that help recognize the same character across images; For illustration, emphasize the composition and framing, line quality, brush/ink style, shading approach, color palette, texture, and the overall artistic mood. Do not use tag lists, prompt commands, weights, or meta phrases (e.g., "this image shows"). Do not guess hidden details. Avoid speculative words like "maybe" or "probably." Output only the description.`;

// Run inference on the attached images with the swappable system prompt
// above and return the completed prompt text.
export async function promptFromImages({
    images,
    endpoint,
    apiKey,
    model,
    signal,
}) {
    const userContent = [
        // { type: "text", text: "Write the prompt for these image(s)." },
        ...images.map((url) => ({
            type: "image_url",
            image_url: { url },
        })),
    ];

    console.log("endpoint", endpoint);
    console.log("systemPrompt", PROMPT_FROM_IMAGES_SYSTEM_PROMPT);
    console.log("userContent", userContent);
    console.log("signal", signal);

    return chatCompletion({
        endpoint,
        apiKey,
        model,
        systemPrompt: PROMPT_FROM_IMAGES_SYSTEM_PROMPT,
        userContent,
        signal,
    });
}
