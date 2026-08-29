"""Small viewer for the faithful Faster-GCG rerun."""
from __future__ import annotations
import argparse, csv, json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

CONDITIONS = (("alpaca", "baseline"), ("advbench", "baseline"), ("advbench", "gcg"), ("advbench", "random"))

def pca2(matrix):
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    variance = singular**2
    return centered @ components[:2].T, variance[:2] / variance.sum()

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        query = urlparse(self.path)
        if query.path == "/":
            self.send_response(302); self.send_header("Location", "/faster_viewer.html"); self.end_headers(); return
        if query.path == "/api/index": return self.send_json(self.server.index)
        if query.path == "/api/pca":
            layer = int(parse_qs(query.query).get("layer", ["25"])[0])
            if layer not in self.server.pca:
                matrix, points = [], []
                for dataset, condition in CONDITIONS:
                    paths = sorted((self.server.full_results/"traces"/dataset/condition).iterdir(), key=lambda path: int(path.name))
                    if len(paths) != 100: raise ValueError(f"Expected 100 {dataset}/{condition} traces, found {len(paths)}")
                    for path in paths:
                        meta = json.loads((path/"meta.json").read_text())
                        start, end = meta["regions"]["prompt"]
                        if meta["regions"]["suffix"][1] > meta["regions"]["suffix"][0]: end = meta["regions"]["suffix"][1]
                        matrix.append(np.load(path/f"raw-L{layer}.npy", mmap_mode="r")[start:end].mean(axis=0, dtype=np.float32))
                        points.append({"dataset":dataset,"condition":condition,"id":path.name})
                coordinates, explained = pca2(np.asarray(matrix, dtype=np.float32))
                self.server.pca[layer] = {"shape":[len(matrix),len(matrix[0])],"explained":explained.tolist(),"points":[point|{"pc1":float(x),"pc2":float(y)} for point,(x,y) in zip(points,coordinates,strict=True)]}
            return self.send_json(self.server.pca[layer])
        if query.path == "/api/sample":
            args = parse_qs(query.query); return self.send_json([r for r in self.server.rows if r["dataset"] == "advbench" and r["id"] == args.get("id", [""])[0]])
        return super().do_GET()
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()
    def send_json(self, value):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--results", type=Path, required=True); p.add_argument("--full-results", type=Path, required=True); p.add_argument("--steering-results", type=Path); p.add_argument("--port", type=int, default=18772); a=p.parse_args()
    rows=[json.loads(line) for line in (a.results/"selected.jsonl").read_text().splitlines() if line]
    summary=list(csv.DictReader((a.results/"summary.csv").open()))
    steering=list(csv.DictReader((a.steering_results/"summary.csv").open())) if a.steering_results and (a.steering_results/"summary.csv").exists() else []
    index={"summary":summary,"steering":steering,"samples":sorted({r["id"] for r in rows if r["dataset"] == "advbench"},key=int),"rows":len(rows),"gcg":[r for r in rows if r["dataset"] == "advbench" and r["condition"] == "gcg"]}
    server=ThreadingHTTPServer(("127.0.0.1",a.port),lambda *x,**k:Handler(*x,directory=str(Path(__file__).parent),**k)); server.index=index; server.rows=rows; server.full_results=a.full_results; server.pca={}
    print(f"http://localhost:{a.port}/faster_viewer.html",flush=True); server.serve_forever()
if __name__ == "__main__": main()
