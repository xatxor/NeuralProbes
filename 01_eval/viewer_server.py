"""Serve the concept viewer and one compact trace at a time."""
import argparse, json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import numpy as np

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        query = parse_qs(urlparse(self.path).query)
        if urlparse(self.path).path == "/api/trace":
            root = Path(self.directory).parent / "traces" / query["benchmark"][0] / query["id"][0]
            meta = json.loads((root / "meta.json").read_text())
            pair = int(query["pair"][0]); index = meta["pair_ids"].index(pair)
            values = np.load(root / f"{query['method'][0]}-L{query['layer'][0]}.npy", mmap_mode="r")[:, index]
            body = json.dumps({"tokens": meta["tokens"], "values": values.astype(float).tolist()}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        super().do_GET()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--results", type=Path, default=Path("01_eval/results")); p.add_argument("--port", type=int, default=8000); a=p.parse_args()
    handler=lambda *args, **kwargs: Handler(*args, directory=str(a.results / "concept_viewer"), **kwargs)
    ThreadingHTTPServer(("127.0.0.1", a.port), handler).serve_forever()
if __name__ == "__main__": main()
