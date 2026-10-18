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
        onuseseed,
        oncopytobin,
    } = $props();

    // Menu is fixed-positioned at the section level so the frame's and
    // filmstrip's overflow clipping can't cut it off.
    let openMenuId = $state(null);
    let menuPos = $state({ top: 0, left: 0 });
    const openItem = $derived(items.find((i) => i.id === openMenuId) ?? null);

    function itemTooltip(item) {
        const lines = [item.prompt];
        if (item.trueCfgScale != null) lines.push(`CFG: ${item.trueCfgScale}`);
        if (item.seed != null) lines.push(`Seed: ${item.seed}`);
        if (item.scheduler) lines.push(`Scheduler: ${item.scheduler}`);
        return lines.join("\n");
    }

    function toggleMenu(id, e) {
        if (openMenuId === id) {
            openMenuId = null;
            return;
        }
        const r = e.currentTarget.getBoundingClientRect();
        menuPos = { top: r.bottom + 4, left: r.left };
        openMenuId = id;
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

        <div
            class="filmstrip"
            onscroll={closeMenu}
        >
            <div class="strip-track">
                {#each items as item (item.id)}
                    <MediaFrame
                        src="data:image/png;base64,{item.images[0]?.b64_json}"
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
                                class:open={openMenuId === item.id}
                                title="prompt options"
                                onclick={(e) => {
                                    e.stopPropagation();
                                    toggleMenu(item.id, e);
                                }}
                            >
                                ⎘
                            </button>
                        </div>
                    </MediaFrame>
                {/each}
            </div>
        </div>

        {#if openItem}
            <div
                class="frame-menu-list"
                role="menu"
                style:top="{menuPos.top}px"
                style:left="{menuPos.left}px"
            >
                <button
                    type="button"
                    role="menuitem"
                    onclick={(e) => {
                        e.stopPropagation();
                        onuseprompt(openItem.prompt);
                        closeMenu();
                    }}
                >
                    use prompt
                </button>
                <button
                    type="button"
                    role="menuitem"
                    disabled={openItem.seed == null || openItem.seed === ""}
                    title={openItem.seed == null || openItem.seed === ""
                        ? "This generation used a random seed"
                        : `Seed: ${openItem.seed}`}
                    onclick={(e) => {
                        e.stopPropagation();
                        onuseseed(openItem.seed);
                        closeMenu();
                    }}
                >
                    use seed
                </button>
                <button
                    type="button"
                    role="menuitem"
                    title="Import a copy of this image into the media bin"
                    onclick={(e) => {
                        e.stopPropagation();
                        oncopytobin(openItem);
                        closeMenu();
                    }}
                >
                    copy to media bin
                </button>
            </div>
        {/if}
    </section>
{/if}
