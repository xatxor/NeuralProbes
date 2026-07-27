"use strict";

const state = {
    manifest: null,
    concepts: null,
    translations: null,
    picked: null,
    chosen: new Set(),
    order: [],
    panes: {},
};

const $ = (id) => document.getElementById(id);
const DATA = "data";
const LISTED = 150;

async function json(path) {
    const response = await fetch(`${DATA}/${path}`);
    if (!response.ok) throw new Error(`${path}: ${response.status}`);
    return response.json();
}

async function blob(stem, method, layer, scale, tokens, pairs) {
    const response = await fetch(`${DATA}/${stem}.m${method}.L${layer}.bin`);
    if (!response.ok) throw new Error(`${stem}.m${method}.L${layer}: ${response.status}`);
    const raw = new Int8Array(await response.arrayBuffer());
    if (raw.length !== tokens * pairs) {
        throw new Error(`${stem}: expected ${tokens * pairs} bytes, got ${raw.length}`);
    }
    return { raw, scale, pairs };
}

const zed = (scores, token, concept) => scores.raw[token * scores.pairs + concept] * scores.scale;

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

function selectedZed(pane, index) {
    let total = 0;
    for (const concept of state.chosen) total += zed(pane.scores, index, concept);
    return total / state.chosen.size;
}

function draw(slot) {
    const pane = state.panes[slot];
    if (!pane) return;
    const host = $(`${slot}stream`);
    host.textContent = "";
    const text = decode(pane.tokens.pieces);
    pane.tokens.pieces.forEach((piece, index) => {
        const role = pane.tokens.role[index];
        const span = document.createElement("span");
        const special = /^<\|.*\|>$/.test(piece) || /^<\/?think>$/.test(piece);
        span.className = `tok${special ? " special" : ""}${role === "response" ? " response" : ""}`;
        span.textContent = special ? `⟨${piece.replace(/[<>|]/g, "")}⟩` : text[index];
        if (state.chosen.size && pane.scores) {
            const paint = shade(selectedZed(pane, index));
            span.style.background = paint.background;
            if (paint.strong) span.classList.add("strong");
        }
        if (slot === "primary" && index === state.picked) span.classList.add("picked");
        span.addEventListener("click", () => {
            state.picked = index;
            describe(slot, index);
            draw("primary");
            draw("secondary");
        });
        host.appendChild(span);
    });
}

function toggle(concept, on) {
    if (on) state.chosen.add(concept);
    else state.chosen.delete(concept);
    chips();
    listing();
    if (state.picked !== null) describe("primary", state.picked);
    draw("primary");
    draw("secondary");
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
    $("status").textContent = state.chosen.size
        ? `heat map is the mean of ${state.chosen.size} tracked concept${state.chosen.size > 1 ? "s" : ""}`
        : "no concepts tracked — the text is unshaded";
}

function listing() {
    const query = $("conceptsearch").value.trim().toLowerCase();
    const list = $("concepts");
    list.textContent = "";

    const moved = $("moved").checked && state.effects;
    const ranked = $("order").value === "did" && state.effects
        ? [...state.order].sort((left, right) =>
            Math.abs(state.effects.did[right] ?? 0) - Math.abs(state.effects.did[left] ?? 0))
        : state.order;
    // A query of digits alone is a pair id and matches exactly, so 1023 does not also return 102.
    const numeric = /^\d+$/.test(query);
    const hits = ranked.filter((concept) => {
        const plain = state.concepts.pairs[concept];
        const shown = named(concept);
        // This checkbox was computed but never applied, so it did nothing. It now filters to the
        // concepts whose behaviour actually differs between reward-hacking episodes and the rest.
        if (moved && !state.effects.hit.includes(plain.pair)) return false;
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
        const effect = state.effects?.did[meta.pair];
        const badge = effect === undefined ? "" :
            `<span class="did${state.effects.hit.includes(meta.pair) ? " strong" : ""}">` +
            `${effect >= 0 ? "+" : ""}${effect.toFixed(2)}σ</span>`;
        body.innerHTML = `${badge}<strong>${meta.concept}</strong><div class="pole">vs ${meta.antagonist}</div>` +
            `<div class="klass">${meta.class} · pair ${meta.pair}</div>`;
        body.addEventListener("click", () => toggle(concept, !state.chosen.has(concept)));

        item.append(box, body);
        list.appendChild(item);
    }
    $("listnote").textContent = rest.length > LISTED
        ? `showing ${shown.length} of ${hits.length} matches — narrow the filter to see the rest`
        : `${hits.length} of ${state.manifest.pairs} concepts match`;
}

function describe(slot, index) {
    const pane = state.panes[slot];
    if (!pane || !pane.scores) return;
    const limit = Number($("dedupe").value);

    const values = [];
    for (let concept = 0; concept < pane.scores.pairs; concept += 1) {
        values.push([concept, zed(pane.scores, index, concept)]);
    }
    values.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));

    const piece = decode([pane.tokens.pieces[index]])[0];
    $("tokentitle").textContent = `token ${index}: ${JSON.stringify(piece)}`;
    $("tokenmeta").textContent =
        `${pane.tokens.role[index]} · norm ${(pane.tokens.norms?.[`L${$("layer").value}`]?.[index]) ?? "—"}` +
        (pane.tokens.logprob[index] === null ? "" : ` · logprob ${pane.tokens.logprob[index]}`) +
        " · click any concept to track it";
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
    $("tokencount").textContent = shown === pane.scores.pairs
        ? `all ${shown} concepts`
        : `${shown} of ${pane.scores.pairs} concepts`;
}

async function open(slot, stem) {
    if (!stem) {
        delete state.panes[slot];
        document.querySelector(`[data-slot="${slot}"]`).classList.add("hidden");
        return;
    }
    const record = state.manifest.generations.find((row) => row.stem === stem);
    const method = Number($("method").value);
    const layer = Number($("layer").value);
    const tokens = await json(`${stem}.tokens.json`);
    let scores = null;
    try {
        scores = await blob(stem, method, layer, record.scales[`m${method}.L${layer}`],
            tokens.pieces.length, state.manifest.pairs);
    } catch (error) {
        $("status").textContent = `no blob for ${stem} m${method} L${layer}: ${error.message}`;
    }
    state.panes[slot] = { stem, tokens, scores, record };
    document.querySelector(`[data-slot="${slot}"]`).classList.remove("hidden");
    $(`${slot}title`).textContent = `${record.behaviour} · cell ${record.cell} · sample ${record.sample}` +
        (record.steer ? ` · steered ${record.steer}` : "");
    const verdict = record.verdict ? ` · judge: ${record.verdict.refusal ?? "?"}` : "";
    $(`${slot}meta`).textContent =
        `${record.topic} · ${record.tactic} · ${record.half} · ${record.tokens} tokens ` +
        `(${record.width} prompt)${verdict}`;
    draw(slot);
    if (slot === "primary" && state.picked !== null && state.picked < tokens.pieces.length) {
        describe("primary", state.picked);
    }
}

function options(select, values, labeller) {
    select.textContent = "";
    for (const value of values) {
        const option = document.createElement("option");
        option.value = String(value);
        option.textContent = labeller ? labeller(value) : String(value);
        select.appendChild(option);
    }
}

function repopulate() {
    const cell = $("cell").value;
    const query = $("search").value.trim().toLowerCase();
    const matches = (row) =>
        !query ||
        [row.stem, row.outcome, row.ending, row.variant, String(row.seed)]
            .join(" ").toLowerCase().includes(query);
    // A trajectory is described by how it ended, so that is the only filter worth having.
    const rows = state.manifest.generations.filter((row) =>
        (!cell || row.ending === cell) && matches(row));

    const label = (row) => `${row.outcome} · seed ${row.seed} · ${row.turns} turns · ` +
        `${row.distinct} impl · ${row.tokens} tok`;
    options($("generation"), rows.map((row) => row.stem),
        (stem) => label(rows.find((row) => row.stem === stem)));
    options($("compare"), [""].concat(state.manifest.generations.map((row) => row.stem)),
        (stem) => (stem ? label(state.manifest.generations.find((row) => row.stem === stem)) : "off"));
    $("matchcount").textContent = `matches (${rows.length})`;
    return rows;
}

async function boot() {
    state.manifest = await json("manifest.json");
    state.concepts = await json("concepts.json");

    try {
        state.translations = await json("translations.json");
    } catch {
        $("language").disabled = true;
    }
    try {
        state.effects = await json("effects.json");
        $("movedwrap").hidden = false;
    } catch {
        $("order").querySelector('[value="did"]').disabled = true;
    }

    state.order = state.concepts.pairs.map((_, index) => index).sort((left, right) => {
        const a = state.concepts.pairs[left];
        const b = state.concepts.pairs[right];
        return a.class.localeCompare(b.class) || a.concept.localeCompare(b.concept);
    });

    options($("method"), state.manifest.methods.map((_, index) => index),
        (index) => state.manifest.methods[index]);
    options($("layer"), state.manifest.layers, (layer) => `L${layer}`);
    $("layer").value = String(state.manifest.layers.includes(18) ? 18 : state.manifest.layers[0]);

    // One filter: the outcome. Options are built from what is actually present, and each carries its
    // count so an empty class is visible rather than silently missing.
    const endings = [...new Set(state.manifest.generations.map((row) => row.ending))];
    const outcomeOf = (ending) =>
        (state.manifest.generations.find((row) => row.ending === ending) || {}).outcome || ending;
    const cellSelect = $("cell");
    for (const value of endings) {
        const option = document.createElement("option");
        option.value = value;
        const n = state.manifest.generations.filter((row) => row.ending === value).length;
        option.textContent = `${outcomeOf(value)} (${n})`;
        cellSelect.appendChild(option);
    }
    const suggestions = $("suggestions");
    for (const value of endings.map(outcomeOf)) {
        const option = document.createElement("option");
        option.value = value;
        suggestions.appendChild(option);
    }

    listing();
    chips();
    const rows = repopulate();
    if (rows.length) await open("primary", rows[0].stem);

    $("generation").addEventListener("change", (event) => open("primary", event.target.value));
    $("compare").addEventListener("change", (event) => open("secondary", event.target.value));
    for (const id of ["method", "layer"]) {
        $(id).addEventListener("change", async () => {
            for (const slot of Object.keys(state.panes)) await open(slot, state.panes[slot].stem);
        });
    }
    for (const id of ["cell", "search"]) {
        $(id).addEventListener(id === "search" ? "input" : "change", async () => {
            const filtered = repopulate();
            if (filtered.length && (id !== "search" || filtered.length <= 3)) {
                await open("primary", filtered[0].stem);
            }
        });
    }
    $("dedupe").addEventListener("input", (event) => {
        const value = Number(event.target.value);
        $("dedupevalue").textContent = value >= 1 ? "off" : value.toFixed(2);
        if (state.picked !== null) describe("primary", state.picked);
    });
    $("tokensearch").addEventListener("input", () => {
        if (state.picked !== null) describe("primary", state.picked);
    });
    $("conceptsearch").addEventListener("input", listing);
    $("order").addEventListener("change", listing);
    $("moved").addEventListener("change", listing);
    $("language").addEventListener("change", () => {
        chips();
        listing();
        if (state.picked !== null) describe("primary", state.picked);
    });
    $("clearall").addEventListener("click", () => {
        state.chosen.clear();
        chips();
        listing();
        if (state.picked !== null) describe("primary", state.picked);
        draw("primary");
        draw("secondary");
    });
}

boot().catch((error) => {
    $("status").textContent = `failed to load: ${error.message}`;
    console.error(error);
});
