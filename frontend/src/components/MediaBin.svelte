<script>
    import MediaFrame from "./MediaFrame.svelte";
    import { extractImageFiles } from "../lib/clipboard-images.js";
    import {
        getEntries,
        getTrayIds,
        getInitError,
        retryInit,
        addFiles,
        remove,
        clear,
    } from "../lib/media-bin.svelte.js";

    let { onimportfromhistory } = $props();

    const BIN_TYPE = "application/x-ircawp-bin";
    const HISTORY_TYPE = "application/x-ircawp-history";

    const entries = $derived(getEntries());
    const traySet = $derived(new Set(getTrayIds()));
    const initError = $derived(getInitError());
    let dragging = $state(false);

    let fileInput;

    function handlePaste(event) {
        const files = extractImageFiles(event.clipboardData);
        if (!files.length) return;
        event.preventDefault();
        addFiles(files);
    }

    // dragover can't read payloads, only their types.
    function isRelevantDrag(event) {
        const types = event.dataTransfer?.types;
        if (!types) return false;
        return types.includes(HISTORY_TYPE) || types.includes("Files");
    }

    function handleDrop(event) {
        event.preventDefault();
        dragging = false;
        const historyId = event.dataTransfer.getData(HISTORY_TYPE);
        if (historyId) {
            onimportfromhistory(historyId);
            return;
        }
        if (event.dataTransfer.files?.length) {
            addFiles(event.dataTransfer.files);
        }
    }

    function handleBrowse() {
        addFiles(fileInput.files);
        fileInput.value = "";
    }
</script>

<svelte:window onpaste={handlePaste} />

<section
    class="media-bin panel"
    class:dragging
    ondragover={(e) => {
        if (isRelevantDrag(e)) {
            e.preventDefault();
            dragging = true;
        }
    }}
    ondragleave={(e) => {
        // dragleave also fires when moving between children; only clear
        // when the pointer actually leaves the section.
        if (!e.currentTarget.contains(e.relatedTarget)) dragging = false;
    }}
    ondrop={handleDrop}
>
    <div class="media-bin-head">
        <h2 class="mono">media bin · {entries.length}</h2>
        {#if entries.length}
            <div class="media-bin-actions">
                <button
                    type="button"
                    class="ghost mono"
                    onclick={() => fileInput.click()}
                >
                    add media
                </button>
                <button
                    type="button"
                    class="ghost mono"
                    disabled={!entries.some((e) => !traySet.has(e.id))}
                    title={traySet.size
                        ? "clears everything except images in the editing queue"
                        : undefined}
                    onclick={() => clear(traySet)}
                >
                    clear all
                </button>
            </div>
        {/if}
    </div>

    {#if entries.length}
        <div class="bin-grid">
            {#each entries as entry (entry.id)}
                <MediaFrame
                    src={entry.dataUrl}
                    label={entry.mime.split("/").pop()}
                    title="drag into editing images"
                    draggable
                    ondragstart={(e) => {
                        e.dataTransfer.setData(
                            BIN_TYPE,
                            JSON.stringify([entry.id]),
                        );
                        e.dataTransfer.effectAllowed = "copy";
                    }}
                    badge={traySet.has(entry.id) ? "in queue" : null}
                    deleteDisabled={traySet.has(entry.id)}
                    deleteTitle={traySet.has(entry.id)
                        ? "in use — remove from editing images first"
                        : undefined}
                    ondelete={() => remove(entry.id)}
                />
            {/each}
        </div>
    {:else if initError}
        <div class="bin-error">
            <p class="mono error-note">{initError}</p>
            <button
                type="button"
                class="ghost mono"
                onclick={() => retryInit()}
            >
                retry
            </button>
        </div>
    {:else}
        <button
            type="button"
            class="dropzone"
            onclick={() => fileInput.click()}
        >
            <span class="mono">drop or paste media, or browse</span>
        </button>
    {/if}

    <input
        bind:this={fileInput}
        type="file"
        accept="image/*"
        multiple
        hidden
        onchange={handleBrowse}
    />
</section>
