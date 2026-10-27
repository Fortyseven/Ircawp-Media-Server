<script>
    import { filterTemplates } from "../lib/prompt-templates.js";

    // `templates` should already be sorted; `value` is the selected name ("" = none).
    let { templates, value = $bindable(""), disabled = false } = $props();

    let open = $state(false);
    let query = $state("");
    let active = $state(0);
    let inputEl = $state(null);

    const matches = $derived(filterTemplates(templates, query));
    // Entry 0 is always the "none" choice, so indexes into `matches` are shifted by one.
    const showNone = $derived(!query.trim());
    const count = $derived(matches.length + (showNone ? 1 : 0));

    function openList() {
        if (disabled) return;
        query = "";
        active = 0;
        open = true;
    }

    function close() {
        open = false;
        query = "";
    }

    function choose(name) {
        value = name;
        close();
    }

    function chooseIndex(i) {
        if (showNone) {
            choose(i === 0 ? "" : matches[i - 1].name);
        } else if (matches[i]) {
            choose(matches[i].name);
        }
    }

    function handleInput() {
        open = true;
        active = 0;
    }

    function handleKeydown(event) {
        if (event.key === "ArrowDown") {
            event.preventDefault();
            if (!open) return openList();
            active = Math.min(active + 1, count - 1);
        } else if (event.key === "ArrowUp") {
            event.preventDefault();
            active = Math.max(active - 1, 0);
        } else if (event.key === "Enter" && open) {
            event.preventDefault();
            if (count) chooseIndex(active);
        } else if (event.key === "Escape" && open) {
            // Don't let an enclosing dialog treat this as "close".
            event.stopPropagation();
            close();
        }
    }

    function scrollActive(node, isActive) {
        if (isActive) node.scrollIntoView({ block: "nearest" });
        return {
            update(next) {
                if (next) node.scrollIntoView({ block: "nearest" });
            },
        };
    }
</script>

<div class="combo">
    <input
        bind:this={inputEl}
        type="text"
        role="combobox"
        aria-expanded={open}
        aria-controls="template-listbox"
        aria-autocomplete="list"
        autocomplete="off"
        spellcheck="false"
        placeholder={open ? "type to filter…" : "none"}
        {disabled}
        value={open ? query : value}
        oninput={(e) => {
            query = e.currentTarget.value;
            handleInput();
        }}
        onfocus={openList}
        onclick={() => !open && openList()}
        onblur={close}
        onkeydown={handleKeydown}
    />
    {#if open}
        <ul
            id="template-listbox"
            role="listbox"
        >
            {#if showNone}
                <li
                    role="option"
                    aria-selected={value === ""}
                    class:active={active === 0}
                    class="none"
                    use:scrollActive={active === 0}
                    onmousedown={(e) => {
                        e.preventDefault();
                        choose("");
                    }}
                >
                    none
                </li>
            {/if}
            {#each matches as t, i (t.name)}
                {@const idx = i + (showNone ? 1 : 0)}
                <li
                    role="option"
                    aria-selected={value === t.name}
                    class:active={active === idx}
                    class:selected={value === t.name}
                    use:scrollActive={active === idx}
                    onmousedown={(e) => {
                        e.preventDefault();
                        choose(t.name);
                    }}
                    onmousemove={() => (active = idx)}
                >
                    {t.name}
                </li>
            {/each}
            {#if !count}
                <li
                    class="empty"
                    role="presentation"
                >
                    no matches
                </li>
            {/if}
        </ul>
    {/if}
</div>

<style lang="scss">
    .combo {
        position: relative;
        width: 100%;

        input {
            width: 100%;
            font-family: var(--font-display);
            font-size: 0.92rem;
            color: var(--text);
            background: var(--bg-deep);
            border: 1px solid var(--border);
            border-radius: 7px;
            padding: 0.55rem 0.7rem;
            transition: border-color 0.15s;

            &:focus {
                outline: none;
                border-color: var(--amber);
            }

            &:disabled {
                opacity: 0.5;
            }
        }

        ul {
            position: absolute;
            z-index: 20;
            top: calc(100% + 4px);
            left: 0;
            right: 0;
            max-height: 14rem;
            overflow-y: auto;
            margin: 0;
            padding: 0.25rem;
            list-style: none;
            background: var(--panel);
            border: 1px solid var(--border-bright);
            border-radius: 7px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        }

        li {
            padding: 0.4rem 0.55rem;
            border-radius: 5px;
            font-size: 0.88rem;
            color: var(--text);
            cursor: pointer;

            &.none {
                color: var(--muted);
            }

            &.selected {
                color: var(--amber);
            }

            &.active {
                background: var(--amber-dim);
            }

            &.empty {
                color: var(--muted);
                cursor: default;
            }
        }
    }
</style>
