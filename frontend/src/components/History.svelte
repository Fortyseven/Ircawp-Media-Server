<script>
    import MediaFrame from "./MediaFrame.svelte";

    const HISTORY_TYPE = "application/x-ircawp-history";

    let {
        items = [],
        activeId = null,
        onview,
        ondelete,
        onclear,
        onuseprompt,
        oncopytobin,
    } = $props();

    let openMenuId = $state(null);

    function itemTooltip(item) {
        const lines = [item.prompt];
        if (item.trueCfgScale != null) lines.push(`CFG: ${item.trueCfgScale}`);
        if (item.seed != null) lines.push(`Seed: ${item.seed}`);
        return lines.join("\n");
    }

    function toggleMenu(id) {
        openMenuId = openMenuId === id ? null : id;
    }

    function closeMenu() {
        openMenuId = null;
    }
</script>

<svelte:window onclick={closeMenu} />

{#if items.length}
    <section class="history">
        <div class="history-head">
            <h2 class="mono">history · {items.length}</h2>
            <button
                class="ghost mono"
                onclick={onclear}>clear all</button
            >
        </div>

        <div class="filmstrip">
            <div class="strip-track">
                {#each items as item (item.id)}
                    <MediaFrame
                        src="data:image/png;base64,{item.images[0]
                            ?.b64_json}"
                        label={item.model +
                            (item.images.length > 1
                                ? ` ×${item.images.length}`
                                : "")}
                        alt={item.prompt}
                        title={itemTooltip(item)}
                        active={item.id === activeId}
                        draggable
                        ondragstart={(e) => {
                            e.dataTransfer.setData(
                                HISTORY_TYPE,
                                String(item.id),
                            );
                            e.dataTransfer.effectAllowed = "copy";
                        }}
                        onview={() => onview(item)}
                        ondelete={() => ondelete(item.id)}
                    >
                        <div class="frame-menu">
                            <button
                                type="button"
                                class="frame-useprompt"
                                aria-label="prompt options"
                                aria-haspopup="true"
                                aria-expanded={openMenuId === item.id}
                                title="prompt options"
                                onclick={(e) => {
                                    e.stopPropagation();
                                    toggleMenu(item.id);
                                }}
                            >
                                ⎘
                            </button>
                            {#if openMenuId === item.id}
                                <div
                                    class="frame-menu-list"
                                    role="menu"
                                >
                                    <button
                                        type="button"
                                        role="menuitem"
                                        onclick={(e) => {
                                            e.stopPropagation();
                                            onuseprompt(item.prompt);
                                            closeMenu();
                                        }}
                                    >
                                        use prompt
                                    </button>
                                    <button
                                        type="button"
                                        role="menuitem"
                                        title="Import a copy of this image into the media bin"
                                        onclick={(e) => {
                                            e.stopPropagation();
                                            oncopytobin(item);
                                            closeMenu();
                                        }}
                                    >
                                        copy to media bin
                                    </button>
                                </div>
                            {/if}
                        </div>
                    </MediaFrame>
                {/each}
            </div>
        </div>
    </section>
{/if}
