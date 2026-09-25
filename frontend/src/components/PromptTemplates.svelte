<script>
    import {
        PROMPT_PLACEHOLDER,
        exportTemplatesCsv,
        importTemplatesCsv,
        loadTemplates,
        resetTemplates,
        saveTemplates,
        sortedTemplates,
    } from "../lib/prompt-templates.js";

    let { onclose, onchange } = $props();

    // Dialog is conditionally mounted, so loading once here is always fresh.
    let templates = $state(loadTemplates());
    // Display order: name-sorted, independent of storage order.
    const sorted = $derived(sortedTemplates(templates));

    // Editor: add a new template, or edit one (editingName = original name).
    let editingName = $state("");
    let isNew = $state(true);
    let name = $state("");
    let body = $state("");
    let formError = $state("");
    let importError = $state("");
    let importInput = $state(null);

    function closeFromBackdrop(event) {
        if (event.currentTarget === event.target) onclose?.();
    }

    function handleKeydown(event) {
        if (event.key === "Escape") onclose?.();
    }

    function persist(next) {
        templates = next;
        saveTemplates(templates);
        onchange?.(templates);
    }

    function resetEditor() {
        editingName = "";
        isNew = true;
        name = "";
        body = "";
        formError = "";
    }

    function startEdit(template) {
        editingName = template.name;
        isNew = false;
        name = template.name;
        body = template.body;
        formError = "";
    }

    function submit(event) {
        event.preventDefault();
        const nextName = name.trim();
        const nextBody = body.trim();
        if (!nextName || !nextBody) {
            formError = "name and body are both required";
            return;
        }
        const clash = templates.find(
            (t) => t.name === nextName && t.name !== editingName,
        );
        if (clash) {
            formError = `a template named "${nextName}" already exists`;
            return;
        }
        if (isNew) {
            persist([...templates, { name: nextName, body: nextBody }]);
        } else {
            persist(
                templates.map((t) =>
                    t.name === editingName
                        ? { name: nextName, body: nextBody }
                        : t,
                ),
            );
        }
        resetEditor();
    }

    function remove(template) {
        if (!confirm(`Delete template "${template.name}"?`)) return;
        if (!isNew && editingName === template.name) resetEditor();
        persist(templates.filter((t) => t.name !== template.name));
    }

    function handleReset() {
        if (
            !confirm(
                "Reset deletes ALL current templates and restores the built-in defaults. Continue?",
            )
        ) {
            return;
        }
        resetEditor();
        persist(resetTemplates());
    }

    function handleExport() {
        const blob = new Blob([exportTemplatesCsv(templates)], {
            type: "text/csv;charset=utf-8",
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "prompt-templates.csv";
        a.click();
        URL.revokeObjectURL(url);
    }

    async function handleImport(event) {
        const file = event.target.files?.[0];
        event.target.value = "";
        if (!file) return;
        importError = "";
        if (
            !confirm(
                "Import replaces ALL current templates with the file's contents. Continue?",
            )
        ) {
            return;
        }
        try {
            const imported = importTemplatesCsv(await file.text());
            resetEditor();
            persist(imported);
        } catch (e) {
            importError = e.message;
        }
    }
</script>

<svelte:window onkeydown={handleKeydown} />

<div
    class="modal-backdrop"
    role="presentation"
    onclick={closeFromBackdrop}
>
    <div
        class="templates-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="templates-title"
    >
        <div class="dialog-head">
            <h2 id="templates-title">prompt templates</h2>
            <button
                type="button"
                class="dialog-close"
                aria-label="Close templates"
                onclick={() => onclose?.()}>close</button
            >
        </div>

        <form
            class="template-editor"
            onsubmit={submit}
        >
            <div class="row">
                <label class="field grow">
                    <span class="label mono">name</span>
                    <input
                        type="text"
                        bind:value={name}
                        placeholder="template name"
                    />
                </label>
            </div>
            <label class="field">
                <span class="label mono">body</span>
                <textarea
                    bind:value={body}
                    rows="3"
                    placeholder={"Describe the transformation. Include " +
                        PROMPT_PLACEHOLDER +
                        " where the prompt box text should go."}
                ></textarea>
            </label>
            {#if formError}
                <p
                    class="form-error mono"
                    role="alert"
                >
                    {formError}
                </p>
            {/if}
            <div class="template-editor-actions">
                {#if isNew}
                    <button
                        type="submit"
                        class="cta">add</button
                    >
                {:else}
                    <button
                        type="button"
                        class="ghost"
                        onclick={resetEditor}>cancel</button
                    >
                    <button
                        type="submit"
                        class="cta">save</button
                    >
                {/if}
            </div>
        </form>

        <div class="template-list">
            {#each sorted as t (t.name)}
                <div class="template-row">
                    <div class="template-info">
                        <span class="name">{t.name}</span>
                        <span class="preview">{t.body}</span>
                    </div>
                    <div class="template-row-actions">
                        <button
                            type="button"
                            class="ghost"
                            onclick={() => startEdit(t)}>edit</button
                        >
                        <button
                            type="button"
                            class="ghost"
                            onclick={() => remove(t)}>delete</button
                        >
                    </div>
                </div>
            {:else}
                <p class="empty-note mono">no templates — add one above</p>
            {/each}
        </div>

        <div class="template-io">
            <button
                type="button"
                class="ghost"
                onclick={handleExport}>export</button
            >
            <button
                type="button"
                class="ghost"
                onclick={() => importInput?.click()}>import</button
            >
            <button
                type="button"
                class="ghost danger"
                onclick={handleReset}>reset</button
            >
            <input
                bind:this={importInput}
                type="file"
                accept=".csv,text/csv"
                class="hidden-file"
                onchange={handleImport}
                aria-label="Import templates backup file"
            />
            {#if importError}
                <span
                    class="io-error mono"
                    role="alert">{importError}</span
                >
            {/if}
        </div>
    </div>
</div>
