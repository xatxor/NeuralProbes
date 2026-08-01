"use strict";

const state = {
    meta: null,
    concepts: null,
    translations: null,
    picked: null,
    chosen: new Set(),
    order: [],
    pane: null,
    busy: false,
};

const $ = (id) => document.getElementById(id);
const LISTED = 150;

// A comma is a decimal point in most of the world and never in a model parameter, so it is accepted
// and normalised rather than silently turning 0,95 into NaN.
const numeric = (id) => Number(String($(id).value).replace(",", ".").trim());

async function api(route, body) {
    const response = await fetch(route, body === undefined
        ? {}
        : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `${route}: ${response.status}`);
    return payload;
}

function unpack(readout) {
    const layers = {};
    for (const [layer, plane] of Object.entries(readout.layers)) {
        const binary = atob(plane.data);
        const raw = new Int8Array(binary.length);
        for (let i = 0; i < binary.length; i += 1) raw[i] = (binary.charCodeAt(i) << 24) >> 24;
        if (raw.length !== readout.tokens * state.meta.pairs) {
            throw new Error(`L${layer}: expected ${readout.tokens * state.meta.pairs} bytes, got ${raw.length}`);
        }
        layers[layer] = { raw, scale: plane.scale };
    }
    return layers;
}

const plane = () => state.pane.layers[$("layer").value];
const zed = (token, concept) => {
    const got = plane();
    return got.raw[token * state.meta.pairs + concept] * got.scale;
};

const BYTES = (() => {
    const map = new Map();
    const direct = new Set();
    for (let value = 33; value <= 126; value += 1) direct.add(value);
    for (let value = 161; value <= 172; value += 1) direct.add(value);
    for (let value = 174; value <= 255; value += 1) direct.add(value);
    for (const value of direct) map.set(String.fromCharCode(value), value);
    let spare = 0;
    for (let value = 0; value < 256; value += 1) {
        if (!direct.has(value)) {
            map.set(String.fromCharCode(256 + spare), value);
            spare += 1;
        }
    }
    return map;
})();

function decode(pieces) {
    const decoder = new TextDecoder("utf-8");
    return pieces.map((piece) => {
        const bytes = new Uint8Array([...piece].map((character) => BYTES.get(character) ?? 63));
        return decoder.decode(bytes, { stream: true });
    });
}

function named(concept) {
    const meta = state.concepts.pairs[concept];
    const language = $("language").value;
    const table = state.translations?.[language];
    if (!table) return meta;
    const pair = table.concepts?.[String(meta.pair)];
    return {
        pair: meta.pair,
        concept: pair?.concept || meta.concept,
        antagonist: pair?.antagonist || meta.antagonist,
        class: table.classes?.[meta.class] || meta.class,
    };
}

function shade(value) {
    const strength = Math.min(Math.abs(value) / 4, 1);
    if (strength < 0.03) return { background: "transparent", strong: false };
    const alpha = 0.18 + 0.72 * strength;
    const colour = value > 0 ? "56, 139, 253" : "248, 81, 73";
    return { background: `rgba(${colour}, ${alpha.toFixed(3)})`, strong: alpha > 0.55 };
}

function selectedZed(index) {
    let total = 0;
    for (const concept of state.chosen) total += zed(index, concept);
    return total / state.chosen.size;
}

function stale() {
    return state.pane !== null && $("box").value !== state.pane.prompt;
}

function draw() {
    const host = $("stream");
    host.textContent = "";
    if (!state.pane) {
        $("streamnote").textContent = "nothing read yet — type a prompt and press generate";
        return;
    }
    const pane = state.pane;
    const text = decode(pane.pieces);
    pane.pieces.forEach((piece, index) => {
        const role = pane.role[index];
        // Only genuine special tokens get the marker form. Keying this off the role was wrong: the
        // newline and the word "user" inside a header are scaffolding but ordinary tokens, and
        // showing their raw BPE spelling put literal Ċ and Ġ on screen and ate every line break.
        const special = pane.special[index];
        const span = document.createElement("span");
        span.className = `tok${special ? " special" : ""}${role === "response" ? " response" : ""}`;
        span.title = `${index} · ${role}`;
        span.textContent = special ? `⟨${piece.replace(/[<>|]/g, "")}⟩` : text[index];
        if (state.chosen.size) {
            const paint = shade(selectedZed(index));
            span.style.background = paint.background;
            if (paint.strong) span.classList.add("strong");
        }
        if (index === state.picked) span.classList.add("picked");
        span.addEventListener("click", () => {
            state.picked = index;
            describe(index);
            if ($("order").value === "token") listing();
            draw();
        });
        host.appendChild(span);
    });
    host.classList.toggle("stale", stale());
    const counts = {};
    for (const role of pane.role) counts[role] = (counts[role] || 0) + 1;
    $("streamnote").textContent = (stale() ? "STALE — the prompt was edited, press generate · " : "") +
        `${pane.tokens} tokens (${pane.content} content) · ` +
        Object.entries(counts).map(([role, n]) => `${n} ${role}`).join(", ") +
        (pane.seed === null || pane.seed === undefined ? "" : ` · seed ${pane.seed}`);
}

function toggle(concept, on) {
    if (on) state.chosen.add(concept);
    else state.chosen.delete(concept);
    chips();
    listing();
    if (state.picked !== null) describe(state.picked);
    draw();
}

function chips() {
    const host = $("chips");
    host.textContent = "";
    $("chosencount").textContent = `${state.chosen.size} selected`;
    $("chosen").classList.toggle("hidden", state.chosen.size === 0);
    for (const concept of state.chosen) {
        const meta = named(concept);
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.append(meta.concept);
        const drop = document.createElement("button");
        drop.textContent = "×";
        drop.title = "stop tracking";
        drop.addEventListener("click", () => toggle(concept, false));
        chip.appendChild(drop);
        host.appendChild(chip);
    }
    if (!state.busy) {
        $("status").textContent = state.chosen.size
            ? `heat map is the mean of ${state.chosen.size} tracked concept${state.chosen.size > 1 ? "s" : ""}`
            : "no concepts tracked — the text is unshaded";
    }
}

function listing() {
    const query = $("conceptsearch").value.trim().toLowerCase();
    const list = $("concepts");
    list.textContent = "";

    // Ordering by strength needs a token to be strong at, so it falls back to the ontology until one
    // is picked rather than silently presenting an arbitrary order as a ranking.
    const byToken = $("order").value === "token" && state.pane && state.picked !== null;
    const ranked = byToken
        ? [...state.order].sort((left, right) =>
            Math.abs(zed(state.picked, right)) - Math.abs(zed(state.picked, left)))
        : state.order;
    // A query of digits alone is a pair id and matches exactly, so 1023 does not also return 102.
    const numeric = /^\d+$/.test(query);
    const hits = ranked.filter((concept) => {
        const plain = state.concepts.pairs[concept];
        const shown = named(concept);
        if (!query) return true;
        if (numeric) return String(plain.pair) === query;
        return [plain.concept, plain.antagonist, plain.class, shown.concept, shown.antagonist,
        shown.class].join(" ").toLowerCase().includes(query);
    });
    const pinned = hits.filter((concept) => state.chosen.has(concept));
    const rest = hits.filter((concept) => !state.chosen.has(concept));
    const shown = pinned.concat(rest.slice(0, LISTED));

    for (const concept of shown) {
        const meta = named(concept);
        const item = document.createElement("li");
        if (state.chosen.has(concept)) item.classList.add("active");

        const box = document.createElement("input");
        box.type = "checkbox";
        box.checked = state.chosen.has(concept);
        box.addEventListener("change", () => toggle(concept, box.checked));

        const body = document.createElement("div");
        body.className = "body";
        const value = byToken ? zed(state.picked, concept) : null;
        const badge = value === null ? "" :
            `<span class="did">${value >= 0 ? "+" : ""}${value.toFixed(2)}σ</span>`;
        body.innerHTML = `${badge}<strong>${meta.concept}</strong><div class="pole">vs ${meta.antagonist}</div>` +
            `<div class="klass">${meta.class} · pair ${meta.pair}</div>`;
        body.addEventListener("click", () => toggle(concept, !state.chosen.has(concept)));

        item.append(box, body);
        list.appendChild(item);
    }
    $("listnote").textContent = rest.length > LISTED
        ? `showing ${shown.length} of ${hits.length} matches — narrow the filter to see the rest`
        : `${hits.length} of ${state.meta.pairs} concepts match`;
}

function describe(index) {
    if (!state.pane) return;
    const limit = Number($("dedupe").value);

    const values = [];
    for (let concept = 0; concept < state.meta.pairs; concept += 1) {
        values.push([concept, zed(index, concept)]);
    }
    values.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));

    const piece = decode([state.pane.pieces[index]])[0];
    $("tokentitle").textContent = `token ${index}: ${JSON.stringify(piece)}`;
    $("tokenmeta").textContent =
        `${state.pane.role[index]} · click any concept to track it`;
    $("tokencount").textContent = "";

    const host = $("tokenlist");
    host.textContent = "";
    const suppressed = new Set();
    // Every concept, not the first 40. A cap here silently hid 996 of 1036, so a concept could be
    // ranked 41st on a token and be invisible with no indication anything had been dropped.
    const needle = ($("tokensearch")?.value || "").trim().toLowerCase();
    const digits = /^\d+$/.test(needle);
    let shown = 0;
    for (const [concept, value] of values) {
        if (needle) {
            const plain = state.concepts.pairs[concept];
            const also = named(concept);
            if (digits) {
                if (String(plain.pair) !== needle) continue;
            } else if (![plain.concept, plain.antagonist, plain.class, also.concept, also.antagonist,
                         also.class].join(" ").toLowerCase().includes(needle)) {
                continue;
            }
        }
        const dupe = limit < 1 && suppressed.has(concept);
        if (limit < 1) {
            for (const other of state.concepts.neighbours[concept] || []) suppressed.add(other);
        }
        shown += 1;
        const meta = named(concept);
        const card = document.createElement("button");
        card.className = `card${dupe ? " dupe" : ""}${state.chosen.has(concept) ? " active" : ""}`;
        card.innerHTML =
            `<span class="cardz ${value >= 0 ? "up" : "down"}">${value >= 0 ? "+" : ""}${value.toFixed(2)}σ</span>` +
            `<span class="cardname">${meta.concept}</span>` +
            `<span class="cardpole">vs ${meta.antagonist}</span>`;
        card.addEventListener("click", () => toggle(concept, !state.chosen.has(concept)));
        host.appendChild(card);
    }
    $("tokencount").textContent = shown === state.meta.pairs
        ? `all ${shown} concepts`
        : `${shown} of ${state.meta.pairs} concepts`;
}

function adopt(got, prompt) {
    const readout = got.readout;
    state.pane = {
        pieces: readout.pieces, role: readout.role, offset: readout.offset,
        special: readout.special, tokens: readout.tokens, content: readout.content,
        layers: unpack(readout),
        prompt, grown: got.grown, seed: got.seed, reply: got.reply,
    };
    if (state.picked !== null && state.picked >= readout.tokens) state.picked = null;
    draw();
    if (state.picked !== null) describe(state.picked);
    listing();
}

function working(on, message) {
    state.busy = on;
    $("generate").disabled = on;
    if (message) $("status").textContent = message;
    else chips();
}

async function run() {
    if (state.busy) return;
    const prompt = $("box").value;
    if (!prompt.trim()) {
        $("status").textContent = "the box is empty — type a prompt first";
        return;
    }
    const started = performance.now();
    working(true, "generating…");
    try {
        const got = await api("/generate", {
            prompt,
            budget: numeric("budget"),
            sample: $("sample").checked,
            temperature: numeric("temperature"),
            top_p: numeric("top_p"),
            top_k: numeric("top_k"),
            seed: $("seed").value.trim(),
        });
        adopt(got, prompt);
        working(false, `generated ${got.new} tokens in ` +
            `${((performance.now() - started) / 1000).toFixed(1)}s` +
            (got.seed === null ? " (greedy)" : ` (seed ${got.seed})`));
    } catch (error) {
        working(false, `failed: ${error.message}`);
        console.error(error);
    }
}

async function boot() {
    state.meta = await api("/meta");
    state.concepts = await api("/concepts.json");
    try {
        state.translations = await api("/translations.json");
    } catch {
        $("language").disabled = true;
    }

    $("rig").textContent = `${state.meta.model} · ${state.meta.device} ${state.meta.dtype} · ` +
        `${state.meta.depth} layers × ${state.meta.hidden}` +
        (state.meta.synthetic ? " · synthetic directions" : "");

    state.order = state.concepts.pairs.map((_, index) => index).sort((left, right) => {
        const a = state.concepts.pairs[left];
        const b = state.concepts.pairs[right];
        return a.class.localeCompare(b.class) || a.concept.localeCompare(b.concept);
    });

    const layers = $("layer");
    for (const layer of state.meta.layers) {
        const option = document.createElement("option");
        option.value = String(layer);
        option.textContent = `L${layer}`;
        layers.appendChild(option);
    }

    listing();
    chips();
    draw();

    $("generate").addEventListener("click", () => run());
    $("box").addEventListener("input", draw);
    $("box").addEventListener("keydown", (event) => {
        if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
            event.preventDefault();
            run();
        }
    });
    $("layer").addEventListener("change", () => {
        draw();
        if (state.picked !== null) describe(state.picked);
        if ($("order").value === "token") listing();
    });
    $("dedupe").addEventListener("input", (event) => {
        const value = Number(event.target.value);
        $("dedupevalue").textContent = value >= 1 ? "off" : value.toFixed(2);
        if (state.picked !== null) describe(state.picked);
    });
    $("tokensearch").addEventListener("input", () => {
        if (state.picked !== null) describe(state.picked);
    });
    $("conceptsearch").addEventListener("input", listing);
    $("order").addEventListener("change", listing);
    $("language").addEventListener("change", () => {
        chips();
        listing();
        if (state.picked !== null) describe(state.picked);
    });
    $("clearall").addEventListener("click", () => {
        state.chosen.clear();
        chips();
        listing();
        if (state.picked !== null) describe(state.picked);
        draw();
    });
    $("status").textContent = "ready — type a prompt and press generate";
}

boot().catch((error) => {
    $("status").textContent = `failed to load: ${error.message}`;
    console.error(error);
});
