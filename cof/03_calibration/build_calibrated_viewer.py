"""Build a static, per-response viewer for calibrated AIME concept scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


HTML = """<!doctype html><meta charset=utf-8><title>Calibrated AIME concepts</title>
<style>body{font:15px system-ui,sans-serif;margin:24px;max-width:1200px;color:#1f2937}select,input{margin:4px;padding:5px}.meta{color:#4b5563}table{border-collapse:collapse;width:100%;margin-top:12px}th,td{padding:6px;text-align:left;border-bottom:1px solid #ddd}th{position:sticky;top:0;background:white}td.num{text-align:right;font-variant-numeric:tabular-nums}</style>
<h1>Calibrated AIME-2024 concept activations</h1>
<p class=meta>__DESCRIPTION__</p>
<label>Response <select id=response></select></label><label>Layer <select id=layer></select></label><label>Search <input id=search placeholder="concept name"></label><p id=meta class=meta></p><table><thead><tr><th>Rank</th><th>Concept</th><th>Calibrated z</th><th>Raw cosine</th><th>Baseline mean</th><th>Baseline std</th></tr></thead><tbody id=rows></tbody></table>
<script>let index,data;const $=x=>document.getElementById(x),esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));function options(el,rows,v,l){el.innerHTML=rows.map(x=>`<option value="${v(x)}">${esc(l(x))}</option>`).join('')}async function load(){index=await(await fetch('index.json')).json();options($('response'),index.responses,x=>x.key,x=>x.label);options($('layer'),index.layers,x=>x,x=>`L${x}`);for(let el of[$('response'),$('layer')])el.onchange=refresh;$('search').oninput=render;refresh()}async function refresh(){data=await(await fetch(`${$('response').value}-L${$('layer').value}.json`)).json();let r=index.responses.find(x=>x.key==$('response').value);$('meta').textContent=`${r.label} · ${data.length} concepts · sorted by calibrated z-score`;render()}function render(){let q=$('search').value.toLowerCase(),rows=data.filter(x=>x.concept.toLowerCase().includes(q));$('rows').innerHTML=rows.map((x,i)=>`<tr><td>${i+1}</td><td>${esc(x.concept)}</td><td class=num>${x.z_mean_cosine.toFixed(4)}</td><td class=num>${x.mean_cosine.toFixed(4)}</td><td class=num>${x.calibration_mean.toFixed(4)}</td><td class=num>${x.calibration_std.toFixed(4)}</td></tr>`).join('')}load()</script>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aime-results", type=Path, required=True)
    parser.add_argument("--calibrated-results", type=Path, required=True)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--description", default="Response-mean <code>diff</code> cosine activations. This viewer has no token heatmap because token-level calibrated values were not saved.")
    args = parser.parse_args()
    scores = pd.read_parquet(args.scores or args.calibrated_results / "concept_scores-aime_2024-calibrated.parquet")
    pairs = pd.read_parquet(args.aime_results / "correlations.parquet")[["pair", "concept"]].drop_duplicates()
    scores = scores.merge(pairs, on="pair", validate="many_to_one")
    responses = {str(row["id"]): row for row in (json.loads(line) for line in (args.aime_results / "aime_2024.jsonl").read_text().splitlines())}
    if set(scores["id"].astype(str)) != set(responses):
        raise RuntimeError("Calibrated score IDs do not match AIME responses")
    viewer = args.calibrated_results / "calibrated_viewer"
    viewer.mkdir(exist_ok=True)
    (viewer / "index.html").write_text(HTML.replace("__DESCRIPTION__", args.description))
    columns = ["pair", "concept", "z_mean_cosine", "mean_cosine", "calibration_mean", "calibration_std"]
    all_rows = scores.groupby(["layer", "pair", "concept"], as_index=False)[columns[2:]].mean()
    for key, frame in [("all", all_rows), *[(str(response_id), scores[scores.id.astype(str) == response_id]) for response_id in responses]]:
        for layer, group in frame.groupby("layer"):
            (viewer / f"{key}-L{layer}.json").write_text(json.dumps(group.sort_values("z_mean_cosine", ascending=False)[columns].to_dict("records")))
    index = {
        "layers": sorted(scores.layer.unique().tolist()),
        "responses": [{"key": "all", "label": "All AIME responses"}] + [
            {"key": key, "label": f"AIME {key} · {'correct' if row.get('correct') else 'incorrect'}"}
            for key, row in responses.items()
        ],
    }
    (viewer / "index.json").write_text(json.dumps(index))
    print(f"Wrote {viewer / 'index.html'}")


if __name__ == "__main__":
    main()
