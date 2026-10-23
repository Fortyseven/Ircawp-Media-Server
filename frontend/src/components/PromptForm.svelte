<script>
    import { untrack } from "svelte";
    import EditingImages from "./EditingImages.svelte";
    import {
        getEntries,
        has,
        isReady,
        readImageFiles,
        resolve,
        setTrayIds,
    } from "../lib/media-bin.svelte.js";
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
        dimensionsForAspect,
        getAspectRatioGroups,
    } from "../lib/size-options.js";
    import { promptFromImages, unloadBackends } from "../lib/api.js";
    import { loadDraft, saveDraft } from "../lib/draft.js";

    let {
        backends = [],
        defaultBackend = "",
        capabilities = {},
        generating = false,
        ongenerate,
        onabort,
    } = $props();

    const initialDraft = untrack(() => loadDraft());
    const initialDraftSettings =
        initialDraft.settings && typeof initialDraft.settings === "object"
            ? initialDraft.settings
            : {};
    // Tray entries are pointers into the media bin ({ binId }), never
    // copies. Legacy drafts stored raw data URLs; those are dropped here —
    // App migrates them into the bin and rewrites the draft before this
    // form mounts when such a draft is present.
    const initialImages = Array.isArray(initialDraft.images)
        ? initialDraft.images.filter(
              (i) => i && typeof i === "object" && typeof i.binId === "string",
          )
        : [];

    let prompt = $state(initialDraft.prompt ?? "");
    // The toggle only controls whether the box is shown (and sent); the text
    // is kept while hidden so re-enabling restores it.
    let negativePrompt = $state(initialDraftSettings.negativePrompt ?? "");
    let showNegativePrompt = $state(
        initialDraftSettings.showNegativePrompt ?? false,
    );
    let templates = $state(loadTemplates());
    let templateName = $state(initialDraftSettings.templateName ?? "");
    let showTemplatesDialog = $state(false);
    // Drop a stale selection (template deleted since the last draft save).
    if (templateName && !templates.some((t) => t.name === templateName)) {
        templateName = "";
    }
    let model = $state(initialDraftSettings.model ?? "");
    let aspectRatio = $state(
        initialDraftSettings.aspectRatio ?? DEFAULT_ASPECT_RATIO,
    );
    let outputSize = $state(
        initialDraftSettings.outputSize ?? DEFAULT_OUTPUT_SIZE,
    );
    // Empty field ("default" placeholder) means the model's own tuned default.
    let trueCfgScale = $state(initialDraftSettings.trueCfgScale ?? undefined);
    let seed = $state(initialDraftSettings.seed ?? undefined);
    let scheduler = $state(initialDraftSettings.scheduler ?? "");
    let hadImages = initialImages.length > 0;
    let unloading = $state(false);
    let unloadNote = $state("");
    let n = $state(initialDraftSettings.n ?? 1);
    let steps = $state(initialDraftSettings.steps ?? undefined);
    let images = $state(initialImages);
    let rewritePrompt = $state(initialDraftSettings.rewritePrompt ?? false);
    let resizeInputImages = $state(
        initialDraftSettings.resizeInputImages ?? true,
    );
    // Prompt-from-images (image dropped on the prompt box): in-flight state
    // and its error note.
    let prompting = $state(false);
    let promptNote = $state("");
    let promptDragging = $state(false);

    const BIN_TYPE = "application/x-ircawp-bin";

    const aspectRatioGroups = $derived(getAspectRatioGroups(images.length > 0));
    // Per-model knobs, advertised by the server via /backends. Unknown
    // backends expose nothing (safe default: controls hidden, model defaults).
    const activeCaps = $derived(capabilities[model || defaultBackend] ?? {});
    const supportsCfg = $derived(activeCaps.cfg === true);
    const cfgExperimental = $derived(activeCaps.cfg_experimental === true);
    const supportsSeed = $derived(activeCaps.seed === true);
    // Shown unless the backend fixes its step count.
    const supportsSteps = $derived(activeCaps.steps !== false);
    const supportsNegativePrompt = $derived(activeCaps.negative_prompt === true);
    const negativeActive = $derived(
        supportsNegativePrompt && showNegativePrompt,
    );
    const schedulerOptions = $derived(activeCaps.schedulers ?? []);
    // A stale pick (model switched, or option removed) falls back to default.
    const activeScheduler = $derived(
        schedulerOptions.includes(scheduler) ? scheduler : "",
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
    const canSubmit = $derived(
        !generating && !prompting && finalPrompt.trim() !== "",
    );

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
                scheduler,
                n,
                steps,
                rewritePrompt,
                resizeInputImages,
                templateName,
                negativePrompt,
                showNegativePrompt,
            },
        });
    });

    // Publish the tray's bin ids so the media bin can badge "in queue".
    $effect(() => {
        setTrayIds(images.map((i) => i.binId));
    });

    // Cascade: drop tray pointers whose bin entry was deleted (or was
    // already gone before the bin finished loading).
    $effect(() => {
        if (!isReady()) return;
        getEntries(); // track bin mutations
        if (images.some((i) => !has(i.binId))) {
            images = images.filter((i) => has(i.binId));
        }
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

    // dragover can't read payloads, only their types.
    function isImageDrag(event) {
        const types = event.dataTransfer?.types;
        return !!types && (types.includes(BIN_TYPE) || types.includes("Files"));
    }

    // Images dropped on the prompt box (from the media bin or the OS) are
    // described by the LLM and the result replaces the prompt. They are not
    // added to the editing queue.
    function handlePromptDrop(event) {
        if (!isImageDrag(event)) return;
        event.preventDefault();
        promptDragging = false;
        if (prompting || generating) return;

        // DataTransfer is only readable synchronously during the event.
        const raw = event.dataTransfer.getData(BIN_TYPE);
        const files = [...(event.dataTransfer.files ?? [])];
        if (raw) {
            let ids = [];
            try {
                const parsed = JSON.parse(raw);
                if (Array.isArray(parsed)) ids = parsed;
            } catch {
                // malformed payload: nothing to resolve
            }
            runPromptFromImages(async () => resolve(ids));
        } else if (files.length) {
            runPromptFromImages(() => readImageFiles(files));
        }
    }

    async function runPromptFromImages(loadImages) {
        promptNote = "";
        prompting = true;
        try {
            const dataUrls = await loadImages();
            if (!dataUrls.length) {
                promptNote = "prompt from images failed: no usable images";
                return;
            }
            const generated = await promptFromImages({ images: dataUrls });
            prompt = generated;
        } catch (e) {
            if (e.name !== "AbortError") {
                promptNote = `prompt from images failed: ${e.message}`;
            }
        } finally {
            prompting = false;
        }
    }

    // Matches the seed input's max so any generated seed can be re-entered.
    function randomSeed() {
        return crypto.getRandomValues(new Uint32Array(1))[0];
    }

    function submit() {
        if (!canSubmit) return;
        const hasSeed = seed != null && seed !== "";
        ongenerate({
            prompt: finalPrompt,
            model: model || undefined,
            // Match source: size stays undefined; output_size tells the
            // backend to scale the source's longest side, keeping its shape.
            size: dimensionsForAspect(aspectRatio, outputSize),
            outputSize,
            aspectRatio,
            // Empty input (field cleared) means "model default".
            trueCfgScale:
                supportsCfg && trueCfgScale !== "" ? trueCfgScale : undefined,
            seed: supportsSeed ? (hasSeed ? seed : randomSeed()) : undefined,
            scheduler: activeScheduler || undefined,
            negativePrompt:
                negativeActive && negativePrompt.trim()
                    ? negativePrompt.trim()
                    : undefined,
            n,
            steps: supportsSteps ? steps : undefined,
            // Resolve tray pointers to data URLs at the API edge.
            images: resolve(images.map((i) => i.binId)),
            rewritePrompt,
            resizeInputImages,
        });
    }

    export function setPrompt(text) {
        prompt = text;
    }

    export function setNegativePrompt(text) {
        negativePrompt = text;
        showNegativePrompt = true;
    }

    export function setSeed(value) {
        seed = value;
    }

    export function addBinIds(ids) {
        images = [...images, ...ids.map((binId) => ({ binId }))];
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
            class:drop-target={promptDragging}
            bind:value={prompt}
            rows="4"
            placeholder={prompting
                ? "describing image…"
                : "Place prompt here... (drop an image to generate a prompt from it)"}
            disabled={generating || prompting}
            ondragover={(e) => {
                if (isImageDrag(e)) {
                    e.preventDefault();
                    promptDragging = true;
                }
            }}
            ondragleave={() => (promptDragging = false)}
            ondrop={handlePromptDrop}></textarea>
        {#if activeTemplate}
            <p
                class="final-prompt mono"
                title={finalPrompt}
            >
                → {finalPrompt}
            </p>
        {/if}
    </label>

    {#if negativeActive}
        <label class="field">
            <span class="label mono">negative prompt</span>
            <textarea
                bind:value={negativePrompt}
                rows="2"
                placeholder="Things to avoid... (only applies when CFG is above 1)"
                disabled={generating}></textarea>
        </label>
    {/if}

    <button
        class="cta"
        type="submit"
        disabled={!canSubmit}
    >
        {generating
            ? "developing…"
            : prompting
              ? "Describing..."
              : images.length
                ? "edit"
                : "generate"}
    </button>

    <EditingImages
        bind:images
        onbinids={addBinIds}
    />

    <div class="rewrite-row">
        <label class="rewrite-toggle">
            <input
                type="checkbox"
                bind:checked={rewritePrompt}
                disabled={generating}
            />
            <span>rewrite prompt</span>
        </label>
        {#if supportsNegativePrompt}
            <label class="rewrite-toggle">
                <input
                    type="checkbox"
                    bind:checked={showNegativePrompt}
                    disabled={generating}
                />
                <span>negative prompt</span>
            </label>
        {/if}
        <label class="rewrite-toggle">
            <input
                type="checkbox"
                bind:checked={resizeInputImages}
                disabled={generating}
            />
            <span>resize media</span>
        </label>
    </div>

    {#if promptNote}
        <span
            class="hint mono error-note"
            role="alert"
        >
            {promptNote}
        </span>
    {/if}

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
            <input
                type="number"
                placeholder="default"
                bind:value={outputSize}
                min="0"
                max="2048"
                disabled={generating}
            />
        </label>

        {#if supportsSteps}
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
        {/if}

        {#if supportsCfg}
            <label class="field small">
                <span class="label mono">cfg scale</span>
                <input
                    type="number"
                    bind:value={trueCfgScale}
                    min="0"
                    step="0.1"
                    placeholder="default"
                    disabled={generating}
                />
                {#if cfgExperimental}
                    <span class="hint mono">
                        CFG-free distillation — values above 1 run an untrained
                        guidance path (experimental)
                    </span>
                {/if}
            </label>
        {/if}

        {#if supportsSeed}
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

        {#if schedulerOptions.length > 0}
            <label class="field">
                <span class="label mono">scheduler</span>
                <select
                    bind:value={scheduler}
                    disabled={generating}
                >
                    <option value="">default</option>
                    {#each schedulerOptions as option}
                        <option value={option}>{option}</option>
                    {/each}
                </select>
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

<style lang="scss">
    .form {
        display: flex;
        flex-direction: column;
        gap: 0.9rem;

        .field {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
            &.grow {
                flex: 1;
            }
            &.small {
                flex: 0 0 4rem;
            }
            label {
                font-size: 0.68rem;
                letter-spacing: 0.14em;
                text-transform: uppercase;
                color: var(--muted);
            }
        }
        textarea,
        select,
        input[type="number"] {
            font-family: var(--font-display);
            font-size: 0.92rem;
            color: var(--text);
            background: var(--bg-deep);
            border: 1px solid var(--border);
            border-radius: 7px;
            padding: 0.55rem 0.7rem;
            transition: border-color 0.15s;
        }

        textarea {
            resize: vertical;
            min-height: 90px;
        }

        textarea:focus,
        select:focus,
        input[type="number"]:focus {
            outline: none;
            border-color: var(--amber);
        }

        textarea.drop-target {
            border-color: var(--amber);
            background: var(--amber-dim);
        }

        textarea:disabled,
        select:disabled,
        input[type="number"]:disabled {
            opacity: 0.5;
        }
        /* --------------------------- */

        .cta {
            font-family: var(--font-mono);
            font-size: 0.85rem;
            letter-spacing: 0.1em;
            text-transform: lowercase;
            color: var(--bg-deep);
            background: var(--amber);
            border: none;
            border-radius: 8px;
            padding: 0.75rem;
            cursor: pointer;
            transition:
                background 0.15s,
                transform 0.1s;

            &:hover:not(:disabled) {
                background: var(--amber-bright);
            }

            &:active:not(:disabled) {
                transform: translateY(1px);
            }

            &:disabled {
                background: var(--border);
                color: var(--muted);
                cursor: not-allowed;
            }
        }

        /* --------------------------- */

        .model-pick {
            display: flex;
            gap: 0.45rem;
            align-items: center;

            select {
                flex: 1;
            }
        }

        /* --------------------------- */

        .rewrite-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
        }

        .rewrite-toggle {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--muted);
            font-size: 0.8rem;
            cursor: pointer;

            &:has(input:disabled) {
                cursor: not-allowed;
                opacity: 0.5;
            }

            input {
                accent-color: var(--amber);
            }
        }

        /* --------------------------- */

        .upload {
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
        }
    }
</style>
