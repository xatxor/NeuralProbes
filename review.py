#! /usr/bin/env python

"""Render the curation shortlist as a single page for review.

Curation proposes eight prompts per class out of twenty-four candidates, but a high label score means
the prompt is *about* the class, not that it makes a good test of it. Only a person can tell those
apart, so the proposal has to be reviewable quickly: 148 classes is too many to read as raw JSON.

The page is self-contained -- data inlined, no network -- because it is served from the box over an
SSH tunnel, where anything fetching a second file is one more thing to go wrong at four in the
morning. Edits export as a fresh `prompts.json`.
"""

import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

log = logging.getLogger("review")

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>curation review</title>
<style>
:root {{ color-scheme: dark; --bg:#14161a; --panel:#1c1f26; --line:#2c313b; --ink:#dfe3ea;
         --dim:#8b94a3; --pick:#3ba55d; --warn:#d99a2b; --accent:#5b8dd9; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); display:flex; height:100vh;
        font:14px/1.5 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif; }}
#side {{ width:340px; flex:none; overflow-y:auto; border-right:1px solid var(--line);
         background:var(--panel); }}
#side .row {{ padding:7px 12px; border-bottom:1px solid var(--line); cursor:pointer;
              display:flex; gap:8px; align-items:baseline; }}
#side .row:hover {{ background:#232833; }}
#side .row.on {{ background:#2b3242; border-left:3px solid var(--accent); padding-left:9px; }}
#side .n {{ color:var(--dim); font-variant-numeric:tabular-nums; font-size:12px; min-width:26px; }}
#side .c {{ margin-left:auto; font-variant-numeric:tabular-nums; font-size:12px; }}
.ok {{ color:var(--pick); }} .short {{ color:var(--warn); }}
#main {{ flex:1; overflow-y:auto; padding:22px 28px 90px; }}
h1 {{ margin:0 0 4px; font-size:19px; }}
.meta {{ color:var(--dim); font-size:13px; margin-bottom:6px; }}
.ex {{ color:var(--dim); font-size:13px; font-style:italic; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-left:3px solid var(--line);
         border-radius:6px; padding:10px 13px; margin:9px 0; cursor:pointer; }}
.card:hover {{ border-color:#3a4150; }}
.card.on {{ border-left-color:var(--pick); background:#1b2620; }}
.card .top {{ display:flex; gap:10px; color:var(--dim); font-size:12px; margin-bottom:5px;
              font-variant-numeric:tabular-nums; }}
.card .why {{ color:var(--pick); }}
.card pre {{ margin:0; white-space:pre-wrap; word-break:break-word; font:13px/1.45 ui-monospace,
             SFMono-Regular,Menlo,monospace; max-height:15em; overflow:auto; }}
#bar {{ position:fixed; bottom:0; left:340px; right:0; background:#11141a;
        border-top:1px solid var(--line); padding:10px 28px; display:flex; gap:16px;
        align-items:center; }}
button {{ background:var(--accent); color:#fff; border:0; border-radius:5px; padding:7px 15px;
          font-size:13px; cursor:pointer; }}
button.ghost {{ background:#2c313b; }}
kbd {{ background:#2c313b; border-radius:3px; padding:1px 5px; font-size:11px; }}
</style>
<div id="side"></div>
<div id="main"></div>
<div id="bar">
  <button onclick="save()">Download prompts.json</button>
  <button class="ghost" onclick="reset()">Reset this class</button>
  <span id="tally" class="meta"></span>
  <span class="meta" style="margin-left:auto">
    <kbd>j</kbd>/<kbd>k</kbd> class &nbsp; <kbd>1</kbd>-<kbd>9</kbd> toggle &nbsp; <kbd>/</kbd> find
  </span>
</div>
<script>
const DATA = {data};
const KEEP = {keep};
let cur = Object.keys(DATA)[0];
const sel = {{}};
for (const [id, c] of Object.entries(DATA))
  sel[id] = new Set(c.picked.map(p => p.conversation_id));

function tally() {{
  const short = Object.values(sel).filter(s => s.size < KEEP).length;
  const total = Object.values(sel).reduce((a, s) => a + s.size, 0);
  document.getElementById('tally').textContent =
    `${{total}} prompts \\u00b7 ${{Object.keys(DATA).length - short}}/${{Object.keys(DATA).length}} classes at ${{KEEP}}`;
}}

function side() {{
  document.getElementById('side').innerHTML = Object.entries(DATA).map(([id, c]) => {{
    const n = sel[id].size, cls = n >= KEEP ? 'ok' : 'short';
    return `<div class="row ${{id === cur ? 'on' : ''}}" onclick="go('${{id}}')">
      <span class="n">${{id}}</span><span>${{c.class}}</span>
      <span class="c ${{cls}}">${{n}}/${{c.candidates.length}}</span></div>`;
  }}).join('');
  const on = document.querySelector('#side .row.on');
  if (on) on.scrollIntoView({{block: 'nearest'}});
}}

function main() {{
  const c = DATA[cur];
  document.getElementById('main').innerHTML =
    `<h1>${{c.class}}</h1>
     <div class="meta">class ${{cur}} \\u00b7 ${{c.candidates.length}} candidates \\u00b7
       ${{sel[cur].size}} selected</div>
     <div class="ex">${{(c.examples || []).join(' &nbsp;\\u00b7&nbsp; ')}}</div>` +
    c.candidates.map((p, i) => {{
      const on = sel[cur].has(p.conversation_id);
      const why = c.picked.find(q => q.conversation_id === p.conversation_id);
      return `<div class="card ${{on ? 'on' : ''}}" onclick="tog('${{p.conversation_id}}')">
        <div class="top"><span>${{i + 1}}</span><span>score ${{p.score}}</span>
          <span>${{p.chars}} chars</span>
          ${{why ? `<span class="why">${{why.why}}</span>` : ''}}</div>
        <pre>${{p.text.replace(/[&<>]/g, m => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}})[m])}}</pre>
      </div>`;
    }}).join('');
}}

function go(id) {{ cur = id; side(); main(); window.scrollTo(0, 0);
                   document.getElementById('main').scrollTop = 0; }}
function tog(cid) {{ sel[cur].has(cid) ? sel[cur].delete(cid) : sel[cur].add(cid);
                     side(); main(); tally(); }}
function reset() {{ sel[cur] = new Set(DATA[cur].picked.map(p => p.conversation_id));
                    side(); main(); tally(); }}

function save() {{
  const out = {{}};
  for (const [id, c] of Object.entries(DATA))
    out[id] = {{class: c.class,
                prompts: c.candidates.filter(p => sel[id].has(p.conversation_id))}};
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 2)],
                                        {{type: 'application/json'}}));
  a.download = 'prompts.json';
  a.click();
}}

addEventListener('keydown', e => {{
  if (e.target.tagName === 'INPUT') return;
  const ids = Object.keys(DATA), at = ids.indexOf(cur);
  if (e.key === 'j' && at < ids.length - 1) go(ids[at + 1]);
  else if (e.key === 'k' && at > 0) go(ids[at - 1]);
  else if (e.key >= '1' && e.key <= '9') {{
    const p = DATA[cur].candidates[+e.key - 1];
    if (p) tog(p.conversation_id);
  }} else if (e.key === '/') {{
    e.preventDefault();
    const q = prompt('find class:');
    if (q) {{
      const hit = Object.entries(DATA)
        .find(([, c]) => c.class.toLowerCase().includes(q.toLowerCase()));
      if (hit) go(hit[0]);
    }}
  }}
}});

side(); main(); tally();
</script>
"""


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    review = json.loads(args.review.read_text())
    built = json.loads(args.classes.read_text())
    examples = built.get("examples", {})

    # The two illustrating contrasts travel with each class: judging whether a prompt suits
    # "Principal Hierarchy & Trust Dynamics" is impossible without seeing what that name covers.
    for entry in review.values():
        entry["examples"] = [item["contrast"] for item in examples.get(entry["class"], [])]

    args.out.write_text(
        PAGE.format(data=json.dumps(review, ensure_ascii=False), keep=args.keep)
    )
    total = sum(len(entry["picked"]) for entry in review.values())
    short = sum(1 for entry in review.values() if len(entry["picked"]) < args.keep)
    log.info(f"wrote {args.out}: {len(review)} classes, {total} proposed prompts, {short} below {args.keep}")
    log.info(f"{args.out.stat().st_size / 1e6:.1f} MB, self-contained")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--review", type=Path, default=Path("candidates.json"))
    parser.add_argument("--classes", type=Path, default=Path("classes.json"))
    parser.add_argument("--out", type=Path, default=Path("review/index.html"))
    parser.add_argument("--keep", type=int, default=8)
    main(parser.parse_args())
