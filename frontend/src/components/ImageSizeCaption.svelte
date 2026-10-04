<script>
    // Probes a data URL once per src and reports its native pixel size.
    // The browser's image cache already holds the decoded thumbnail, so the
    // probe is a cache hit, not a re-decode.
    let { src } = $props();
    let size = $state("");

    $effect(() => {
        size = "";
        if (!src) return;
        const probe = new Image();
        probe.onload = () =>
            (size = `${probe.naturalWidth}×${probe.naturalHeight}`);
        probe.onerror = () => (size = "");
        probe.src = src;
    });
</script>

{#if size}
    <span class="thumb-caption mono">{size}</span>
{/if}
