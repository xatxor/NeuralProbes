"""Serve the projection viewer and one memory-mapped concept trace at a time."""

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/trace":
            return super().do_GET()
        try:
            query = parse_qs(parsed.query)
            root = (
                Path(self.directory).parent
                / "traces"
                / "aime_2024"
                / query["id"][0]
            )
            layer, pair = int(query["layer"][0]), int(query["pair"][0])
            if not query["id"][0].isdigit() or layer not in (11, 14, 18, 22, 25) or not 0 <= pair < 1036:
                raise ValueError("Invalid response, layer, or concept")
            meta = json.loads((root / "meta.json").read_text())
            values = np.load(root / f"L{layer}.npy", mmap_mode="r")[:, pair]
            body = json.dumps(
                {
                    "tokens": meta["tokens"],
                    "values": values.astype(float).tolist(),
                    "thinking_start": meta["thinking_start"],
                    "thinking_end": meta["thinking_end"],
                    "color_scale": meta["color_scales"][str(layer)],
                },
                ensure_ascii=False,
            ).encode()
        except (KeyError, ValueError, FileNotFoundError, IndexError) as error:
            self.send_error(400, str(error))
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18768)
    args = parser.parse_args()
    handler = lambda *items, **kwargs: Handler(
        *items, directory=str(args.results / "viewer"), **kwargs
    )
    print(f"Open http://127.0.0.1:{args.port}/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
