<script>
    // Shared frame chrome for the media bin grid and the history filmstrip:
    // image + label, optional view button, optional "in queue" badge, delete
    // control, and a slot for corner extras (history's prompt-options menu).
    let {
        src,
        label,
        alt = undefined,
        title = undefined,
        active = false,
        draggable = false,
        badge = null,
        onview = null,
        ondragstart = null,
        ondelete,
        deleteDisabled = false,
        deleteTitle = undefined,
    } = $props();
</script>

<div
    class="frame"
    class:active
    draggable={draggable}
    ondragstart={ondragstart}
>
    {#if onview}
        <button
            type="button"
            class="frame-view"
            onclick={onview}
            title={title}
        >
            <img
                src={src}
                alt={alt ?? label}
                loading="lazy"
            />
            <span class="frame-label mono">{label}</span>
        </button>
    {:else}
        <div
            class="frame-view frame-static"
            title={title}
        >
            <img
                src={src}
                alt={alt ?? label}
                loading="lazy"
            />
            <span class="frame-label mono">{label}</span>
        </div>
    {/if}
    {#if badge}
        <span class="frame-badge mono">{badge}</span>
    {/if}
    <button
        type="button"
        class="frame-del"
        aria-label="delete"
        disabled={deleteDisabled}
        title={deleteTitle}
        onclick={(e) => {
            e.stopPropagation();
            ondelete();
        }}
    >
        ×
    </button>
    <slot />
</div>
