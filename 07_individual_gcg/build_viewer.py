#!/usr/bin/env python3
"""Build a single-file viewer for the individual-GCG summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


HTML = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Individual GCG results</title><style>
body{font:15px/1.45 system-ui,sans-serif;color:#1f2937;margin:0}main{max-width:1200px;margin:auto;padding:28px}h1{margin:0}.muted{color:#64748b}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:22px 0}label{display:grid;gap:4px;color:#475569}select{padding:7px;border:1px solid #cbd5e1;border-radius:6px;background:white}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px;border-bottom:1px solid #e2e8f0;text-align:left}th{color:#64748b;font-size:12px}td.num{text-align:right}.card{border:1px solid #dbe2ea;border-radius:10px;padding:16px;margin:18px 0}.metric{display:inline-block;min-width:180px;margin:8px 26px 8px 0}.metric b{display:block;font-size:27px}
</style><main><h1>Individual GCG experiment</h1><p class=muted>100 AdvBench + 100 Alpaca prompts · Qwen3-8B · 500-step individual suffixes · max 128 tokens. StrongREJECT ASR threshold: 0.65.</p><div id=metrics class=card></div><div class=controls><label>Dataset<select id=dataset></select></label><label>Prompt condition<select id=condition></select></label><label>Layer<select id=layer></select></label><label>α<select id=alpha></select></label><label>Intervention<select id=mode></select></label></div><table><thead><tr><th>Dataset</th><th>Prompt</th><th>Layer</th><th>α</th><th>Intervention</th><th>n</th><th>ASR</th><th>Mean StrongREJECT</th><th>Mean tokens</th></tr></thead><tbody id=rows></tbody></table></main><script>
const data=__DATA__,fields=['dataset','condition','layer','alpha','mode'];const $=id=>document.getElementById(id);
function value(x){return x==null||x===''?'unsteered':String(x)}
function options(f){let a=[...new Set(data.map(x=>value(x[f])))].sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}));$(f).innerHTML='<option value="">All</option>'+a.map(x=>'<option>'+x+'</option>').join('')}
function n(x,d=3){return Number.isFinite(x)?x.toFixed(d):'—'}
function render(){let r=data.filter(x=>fields.every(f=>!$(f).value||value(x[f])===$(f).value));$('rows').innerHTML=r.map(x=>'<tr><td>'+x.dataset+'</td><td>'+x.condition+'</td><td>'+value(x.layer)+'</td><td>'+value(x.alpha)+'</td><td>'+value(x.mode)+'</td><td class=num>'+x.samples+'</td><td class=num>'+(Number.isFinite(x.asr)?(100*x.asr).toFixed(1)+'%':'—')+'</td><td class=num>'+n(x.mean_strongreject)+'</td><td class=num>'+n(x.mean_tokens,1)+'</td></tr>').join('');let b=r.find(x=>x.dataset==='advbench'&&x.condition==='baseline'&&value(x.layer)==='unsteered'),g=r.find(x=>x.dataset==='advbench'&&x.condition==='gcg'&&value(x.layer)==='unsteered');$('metrics').innerHTML=b&&g?'<div class=metric>Baseline ASR<b>'+(100*b.asr).toFixed(1)+'%</b></div><div class=metric>GCG ASR<b>'+(100*g.asr).toFixed(1)+'%</b></div><div class=metric>GCG − baseline<b>'+(100*(g.asr-b.asr)).toFixed(1)+' pp</b></div>':'<span class=muted>Choose “All” filters to show the unsteered baseline/GCG comparison.</span>'}
for(const f of fields){options(f);$(f).onchange=render}render();</script>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, default=Path("viewer.html"))
    args = parser.parse_args()
    with args.summary.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("layer", "alpha"):
            row[key] = float(row[key]) if row[key] else None
        for key in ("samples", "mean_tokens", "mean_strongreject", "asr"):
            row[key] = float(row[key]) if row[key] else None
    assert rows, "summary.csv has no rows"
    args.output.write_text(HTML.replace("__DATA__", json.dumps(rows)), encoding="utf-8")


if __name__ == "__main__":
    main()
