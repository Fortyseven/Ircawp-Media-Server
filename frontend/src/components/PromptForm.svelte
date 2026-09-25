<script>
    import { untrack } from "svelte";
    import ImageUpload from "./ImageUpload.svelte";
    import PromptTemplates from "./PromptTemplates.svelte";
    import {
        applyTemplate,
        loadTemplates,
        sortedTemplates,
    } from "../lib/prompt-templates.js";
    import {
        DEFAULT_ASPECT_RATIO,
        DEFAULT_OUTPUT_SIZE,
        MATCH_SOURCE,
        OUTPUT_SIZES,
        dimensionsForAspect,
        getAspectRatioGroups,
    } from "../lib/size-options.js";
    import { unloadBackends } from "../lib/api.js";
    import { loadDraft, saveDraft } from "../lib/draft.js";

    let {
        backends = [],
        defaultBackend = "",
        settings = {},
        generating = false,
        ongenerate,
        onabort,
    } = $props();

    const initialSettings = untrack(() => settings);
    const initialDraft = untrack(() => loadDraft());
    const initialDraftSettings =
        initialDraft.settings && typeof initialDraft.settings === "object"
            ? initialDraft.settings
            : {};
    const initialImages = Array.isArray(initialDraft.images)
        ? initialDraft.images
        : [];

    let prompt = $state(initialDraft.prompt ?? "");
    let templates = $state(loadTemplates());
    let templateName = $state(initialDraftSettings.templateName ?? "");
    let showTemplatesDialog = $state(false);
    // Drop a stale selection (template deleted since the last draft save).
    if (templateName && !templates.some((t) => t.name === templateName)) {
        templateName = "";
    }
    let model = $state(
        initialDraftSettings.model ?? initialSettings.model ?? "",
    );
    let aspectRatio = $state(
        initialDraftSettings.aspectRatio ??
            initialSettings.aspectRatio ??
            DEFAULT_ASPECT_RATIO,
    );
    let outputSize = $state(
        initialDraftSettings.outputSize ??
            initialSettings.outputSize ??
            DEFAULT_OUTPUT_SIZE,
    );
    let trueCfgScale = $state(
        initialDraftSettings.trueCfgScale ??
            initialSettings.trueCfgScale ??
            1.0,
    );
    let seed = $state(
        initialDraftSettings.seed ?? initialSettings.seed ?? undefined,
    );
    let hadImages = initialImages.length > 0;
    let unloading = $state(false);
    let unloadNote = $state("");
    let n = $state(initialDraftSettings.n ?? initialSettings.n ?? 1);
    let steps = $state(
        initialDraftSettings.steps ?? initialSettings.steps ?? undefined,
    );
    let images = $state(initialImages);
    let rewritePrompt = $state(
        initialDraftSettings.rewritePrompt ??
            initialSettings.rewritePrompt ??
            false,
    );

    const aspectRatioGroups = $derived(getAspectRatioGroups(images.length > 0));
    const supportsQwenControls = $derived(
        (model || defaultBackend) === "qwenimage21",
    );

    // Dropdown display order: name-sorted, independent of storage order.
    const sorted = $derived(sortedTemplates(templates));
    const activeTemplate = $derived(
        templates.find((t) => t.name === templateName) ?? null,
    );
    // The prompt actually sent: template body with the box text spliced into
    // {prompt} when present, the body alone otherwise, raw box text for none.
    const finalPrompt = $derived(
        activeTemplate ? applyTemplate(activeTemplate, prompt) : prompt.trim(),
    );
    const canSubmit = $derived(!generating && finalPrompt.trim() !== "");

    $effect(() => {
        const hasImages = images.length > 0;

        if (hasImages && !hadImages) {
            aspectRatio = MATCH_SOURCE;
        } else if (!hasImages && aspectRatio === MATCH_SOURCE) {
            aspectRatio = DEFAULT_ASPECT_RATIO;
        }

        hadImages = hasImages;
    });

    $effect(() => {
        saveDraft({
            prompt,
            images,
            settings: {
                model,
                aspectRatio,
                outputSize,
                trueCfgScale,
                seed,
                n,
                steps,
                rewritePrompt,
                templateName,
            },
        });
    });

    function handleTemplatesChange(list) {
        templates = list;
        if (templateName && !list.some((t) => t.name === templateName)) {
            templateName = "";
        }
    }

    async function handleUnload() {
        if (unloading) return;
        unloading = true;
        unloadNote = "unloading…";
        try {
            const res = await unloadBackends();
            unloadNote =
                res.unloaded?.length > 0
                    ? `unloaded ${res.unloaded.join(", ")}`
                    : "nothing loaded";
        } catch (e) {
            unloadNote = `unload failed: ${e.message}`;
        } finally {
            unloading = false;
        }
    }

    // Transient feedback: clear the note a few seconds after it appears.
    $effect(() => {
        if (!unloadNote) return;
        const t = setTimeout(() => (unloadNote = ""), 5000);
        return () => clearTimeout(t);
    });

    function submit() {
        if (!canSubmit) return;
        ongenerate({
            prompt: finalPrompt,
            model: model || undefined,
            // Match source: size stays undefined; output_size tells the
            // backend to scale the source's longest side, keeping its shape.
            size: dimensionsForAspect(aspectRatio, outputSize),
            outputSize,
            aspectRatio,
            trueCfgScale: supportsQwenControls ? trueCfgScale : undefined,
            seed: supportsQwenControls ? seed : undefined,
            n,
            steps,
            images,
            rewritePrompt,
        });
    }

    export function setPrompt(text) {
        prompt = text;
    }

    export function addImage(dataUrl) {
        images = [...images, dataUrl];
    }

    function handleShortcut(event) {
        if (event.key === "Escape" && generating) {
            event.preventDefault();
            onabort();
            return;
        }

        if (
            event.ctrlKey &&
            (event.key === "Enter" || event.key.toLowerCase() === "e")
        ) {
            event.preventDefault();
            submit();
        }
    }
</script>

<svelte:window onkeydown={handleShortcut} />

<form
    class="form"
    onsubmit={(e) => {
        e.preventDefault();
        submit();
    }}
>
    <div class="template-pick">
        <label class="field">
            <span class="label mono">template</span>
            <select
                bind:value={templateName}
                disabled={generating}
            >
                <option value="">none</option>
                {#each sorted as t (t.name)}
                    <option value={t.name}>{t.name}</option>
                {/each}
            </select>
        </label>
        <button
            class="config-button"
            type="button"
            onclick={() => (showTemplatesDialog = true)}
            disabled={generating}
        >
            manage
        </button>
    </div>

    <label class="field">
        <span class="label mono">prompt</span>
        <textarea
            bind:value={prompt}
            rows="4"
            placeholder="a lighthouse in a storm, oil painting…"
            disabled={generating}
        ></textarea>
        {#if activeTemplate}
            <p
                class="final-prompt mono"
                title={finalPrompt}
            >
                → {finalPrompt}
            </p>
        {/if}
    </label>

    <button
        class="cta"
        type="submit"
        disabled={!canSubmit}
    >
        {generating ? "developing…" : images.length ? "edit" : "generate"}
    </button>

    <ImageUpload bind:images />

    <label class="rewrite-toggle">
        <input
            type="checkbox"
            bind:checked={rewritePrompt}
            disabled={generating}
        />
        <span>rewrite prompt before generating</span>
    </label>

    <div class="row">
        <div class="field grow model-field">
            <span class="label mono">model</span>
            <div class="model-pick">
                <select
                    bind:value={model}
                    disabled={generating}
                >
                    <option value="">default ({defaultBackend || "…"})</option>
                    {#each backends as b}
                        <option value={b}>{b}</option>
                    {/each}
                </select>
                <button
                    class="config-button"
                    type="button"
                    onclick={handleUnload}
                    disabled={generating || unloading}
                    title="Unload all models from the backend server"
                >
                    {unloading ? "unloading…" : "unload"}
                </button>
            </div>
            {#if unloadNote}
                <span class="hint mono">{unloadNote}</span>
            {/if}
        </div>

        <label class="field">
            <span class="label mono">images</span>
            <select
                bind:value={n}
                disabled={generating}
            >
                <option value={1}>1</option>
                <option value={2}>2</option>
                <option value={3}>3</option>
                <option value={4}>4</option>
            </select>
        </label>
    </div>

    <div class="row">
        <label class="field">
            <span class="label mono">aspect</span>
            <select
                bind:value={aspectRatio}
                disabled={generating}
            >
                {#each aspectRatioGroups as group}
                    <optgroup label={group.label}>
                        {#each group.options as option}
                            <option value={option.value}>{option.label}</option>
                        {/each}
                    </optgroup>
                {/each}
            </select>
        </label>

        <label class="field">
            <span class="label mono">output</span>
            <select
                bind:value={outputSize}
                disabled={generating}
            >
                {#each OUTPUT_SIZES as edge}
                    <option value={edge}>{edge}px</option>
                {/each}
            </select>
        </label>

        <label class="field">
            <span class="label mono">steps</span>
            <input
                type="number"
                bind:value={steps}
                min="1"
                max="100"
                disabled={generating}
            />
        </label>

        {#if supportsQwenControls}
            <label class="field">
                <span class="label mono">true cfg</span>
                <input
                    type="number"
                    bind:value={trueCfgScale}
                    min="0"
                    step="0.1"
                    placeholder="default"
                    disabled={generating}
                />
            </label>

            <label class="field">
                <span class="label mono">seed</span>
                <input
                    type="number"
                    bind:value={seed}
                    min="0"
                    max="4294967295"
                    step="1"
                    placeholder="random"
                    disabled={generating}
                />
            </label>
        {/if}
    </div>

    {#if n > 1}
        <p class="hint mono">
            {n} images render sequentially — allow ~{n}× the time
        </p>
    {/if}
</form>

{#if showTemplatesDialog}
    <PromptTemplates
        onchange={handleTemplatesChange}
        onclose={() => (showTemplatesDialog = false)}
    />
{/if}
