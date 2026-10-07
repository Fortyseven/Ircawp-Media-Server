<script>
    import ImageSizeCaption from "./ImageSizeCaption.svelte";
    import { getEntry } from "../lib/media-bin.svelte.js";

    const BIN_TYPE = "application/x-ircawp-bin";

    // `images` holds pointers into the media bin ({ binId }), never copies.
    let { images = $bindable([]), onbinids } = $props();

    let dragging = $state(false);
    let draggedIndex = $state(null);

    function isBinDrag(event) {
        return event.dataTransfer?.types?.includes(BIN_TYPE);
    }

    function readBinIds(dataTransfer) {
        try {
            const ids = JSON.parse(dataTransfer.getData(BIN_TYPE));
            return Array.isArray(ids) ? ids : [];
        } catch {
            return [];
        }
    }

    // A bin frame dropped onto this section adds its entries to the tray.
    function handleBinDrop(event) {
        event.preventDefault();
        event.stopPropagation();
        dragging = false;
        const ids = readBinIds(event.dataTransfer);
        if (ids.length) onbinids(ids);
    }

    function removeAt(i) {
        images = images.filter((_, j) => j !== i);
    }

    function moveImage(from, to) {
        if (from === null || from === to) return;
        const reordered = [...images];
        const [image] = reordered.splice(from, 1);
        reordered.splice(to, 0, image);
        images = reordered;
    }
</script>

<div
    class="upload"
    class:dragging
    role="group"
    aria-label="Editing images"
    ondragover={(e) => {
        if (isBinDrag(e)) {
            e.preventDefault();
            dragging = true;
        }
    }}
    ondragleave={(e) => {
        // dragleave also fires when moving between children; only clear
        // when the pointer actually leaves the section.
        if (!e.currentTarget.contains(e.relatedTarget)) dragging = false;
    }}
    ondrop={(e) => {
        if (isBinDrag(e)) handleBinDrop(e);
    }}
>
    <div
        class="dropzone dropzone-hint"
        aria-hidden="true"
    >
        <span class="mono"
            >{images.length
                ? "drop more media here from the media bin"
                : "drop media here from the media bin"}</span
        >
    </div>

    {#if images.length}
        <div
            class="thumbs"
            role="list"
        >
            {#each images as img, i}
                {@const src = getEntry(img.binId)?.dataUrl ?? ""}
                <div
                    class="thumb"
                    role="listitem"
                    draggable={true}
                    ondragstart={() => (draggedIndex = i)}
                    ondragend={() => (draggedIndex = null)}
                    ondragover={(event) => {
                        if (draggedIndex !== null) event.preventDefault();
                    }}
                    ondrop={(event) => {
                        if (isBinDrag(event)) {
                            handleBinDrop(event);
                            return;
                        }
                        event.preventDefault();
                        event.stopPropagation();
                        moveImage(draggedIndex, i);
                    }}
                >
                    <img
                        src={src}
                        alt="editing image {i + 1}"
                    />
                    <ImageSizeCaption src={src} />
                    <button
                        type="button"
                        class="thumb-del"
                        onclick={() => removeAt(i)}
                        aria-label="remove image {i + 1}"
                    >
                        ×
                    </button>
                </div>
            {/each}
        </div>
    {/if}
</div>
