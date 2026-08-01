#! /usr/bin/env python

"""Type a conversation by hand, generate into it, and read every concept off every token.

The `scope/` viewer replays generations produced hours earlier on a cluster and packed into int8
blobs. This does the same thing with the model in the loop: the conversation is a plain textarea, the
chat scaffolding is typed by hand, and the readout is a forward pass over whatever is in the box at
the moment you ask for it.

You type a user prompt and nothing else. The scaffolding around it is built here, by hand rather than
by `apply_chat_template`, for one reason: Qwen2.5's template silently injects "You are Qwen, created
by Alibaba Cloud" as a system turn, and there is meant to be no system prompt at all. Building the
string explicitly is the only way to promise that for every model.

The assistant turn opens with an empty `<think>\n\n</think>` block where the model has one, which is
the form a *filled* Qwen3 turn takes and what every earlier run in this project used. Qwen3's
generation prompt does not include it, so prefilling it is a deliberate choice, not the default.
Models without a `<think>` token get no block, since there it would only be six junk tokens.

Self-contained by decision. It duplicates `capture` from `evals.py` and `quantise` from
`evalscope.py` rather than importing them, because this directory is copied to a server on its own
and an import from the project root would break the moment it lands there.
"""

import base64
import json
import logging
import random
from argparse import ArgumentParser, Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch

log = logging.getLogger("genlive")

# Clipping bound for the int8 packing, in z units, matching `evalscope.py` so the front end's shading
# means the same thing here as it does there.
SPAN = 8.0
# Tokens that are scaffolding rather than content. They are displayed, but they are excluded from the
# mean and standard deviation the z-scores are built on: they sit far off the manifold the ordinary
# tokens occupy and would drag both statistics badly.
SCAFFOLD = ("<think>", "</think>", "<|endoftext|>")
HERE = Path(__file__).resolve().parent


def hardware(device: str, dtype: str) -> tuple[str, torch.dtype]:
    """Pick where the model runs and in what precision.

    :param device: `auto`, or a torch device string.
    :param dtype: `auto`, or one of `bfloat16`, `float16`, `float32`.

    :return: the device string and the torch dtype.
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if dtype == "auto":
        if device == "cuda":
            chosen = (torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False)
                      else torch.float16)
        elif device == "mps":
            chosen = torch.float16
        else:
            chosen = torch.float32
    else:
        chosen = getattr(torch, dtype)
    return device, chosen


def roles(ids: list[int], tokenizer: Any) -> list[str]:
    """Label every token with the turn it belongs to, by walking the chat markers.

    The text is raw, so there is no role mask to inherit -- it has to be recovered from the token
    stream. `<|im_start|>` opens a turn and is followed by the role name up to the first newline;
    `<|im_end|>` closes it. Everything outside a turn is `loose`, which is a legitimate state here:
    typing bare text with no markers at all is a thing you may well want to do.

    :param ids: token ids of the whole box.
    :param tokenizer: the model's tokenizer.

    :return: one role per token: `template`, `system`, `user`, `response`, `loose`, or whatever
        other role name was typed.
    """
    vocab = tokenizer.get_vocab()
    start, end = vocab.get("<|im_start|>"), vocab.get("<|im_end|>")
    marks = {vocab[name] for name in SCAFFOLD if name in vocab}
    pieces = tokenizer.convert_ids_to_tokens(ids)
    text = [tokenizer.convert_tokens_to_string([piece]) for piece in pieces]

    out: list[str] = []
    current, index = "loose", 0
    while index < len(ids):
        if start is not None and ids[index] == start:
            out.append("template")
            index += 1
            name: list[str] = []
            # The role name runs to the first newline. Guard on `<|im_end|>` too, so a malformed
            # header cannot swallow the rest of the conversation.
            while index < len(ids) and ids[index] != end:
                out.append("template")
                name.append(text[index])
                index += 1
                if "\n" in text[index - 1]:
                    break
            label = "".join(name).strip()
            current = "response" if label == "assistant" else (label or "loose")
        elif end is not None and ids[index] == end:
            out.append("template")
            current = "loose"
            index += 1
        elif ids[index] in marks:
            out.append("template")
            index += 1
        else:
            out.append(current)
            index += 1
    return out


def quantise(block: np.ndarray) -> tuple[str, float]:
    """Pack a `[token, pair]` plane of z-scores into base64 int8 with a shared scale.

    :param block: z-scores for one layer.

    :return: base64 of the packed bytes, and the scale that turns them back into z.
    """
    scale = SPAN / 127.0
    packed = np.clip(np.round(block / scale), -127, 127).astype(np.int8).tobytes()
    return base64.b64encode(packed).decode(), scale


class Live:
    """Model, directions and hooks, held open across requests."""

    def __init__(self, args: Namespace) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.lock = Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.device, self.dtype = hardware(args.device, args.dtype)
        self.model = AutoModelForCausalLM.from_pretrained(args.model, dtype=self.dtype)
        self.model.to(self.device)
        self.model.eval()
        self.layers = [int(v) for v in args.layers.split(",")]
        hidden = self.model.config.hidden_size
        depth = self.model.config.num_hidden_layers
        log.info(f"{args.model} on {self.device} in {self.dtype}: {depth} layers, hidden {hidden}")
        for layer in self.layers:
            if not 1 <= layer <= depth:
                raise SystemExit(f"layer {layer} is outside this model's 1..{depth}")

        rows = self._catalogue(args.pairs)
        self.synthetic = args.synthetic
        if self.synthetic:
            # Random directions of the right shape. Every stage downstream runs for real -- hook,
            # cosine, z-score, packing, transport, shading -- and the only thing that is fake is what
            # the directions mean. The front end says so in a banner it cannot be dismissed from.
            log.warning(f"SYNTHETIC directions: {len(rows)} random unit vectors of width {hidden}")
            raw = np.random.default_rng(args.seed).normal(size=(len(self.layers), len(rows), hidden))
        else:
            from safetensors.numpy import load_file

            block = load_file(args.vectors)["diff"]
            if block.shape[-1] != hidden:
                raise SystemExit(
                    f"{args.vectors} is width {block.shape[-1]} but {args.model} is width {hidden}; "
                    f"these vectors do not belong to this model. Pass --synthetic to run the "
                    f"pipeline on random directions instead.")
            raw = np.stack([block[i] for i in [args.rows.index(v) for v in self.layers]])
        unit = raw / np.linalg.norm(raw, axis=-1, keepdims=True)
        self.catalogue = {
            "pairs": rows,
            "neighbours": self._neighbours(unit[-1].astype(np.float32), args.neighbours),
        }
        self.vectors = torch.tensor(unit, dtype=torch.float32, device=self.device)

        self.armed = False
        self.captured: dict[int, torch.Tensor] = {}
        for position, layer in enumerate(self.layers):
            self.model.model.layers[layer - 1].register_forward_hook(self._hook(position))

        # Which ids are real special tokens, as against ordinary ones that merely happen to be
        # scaffolding. The distinction is what the stream renders on: `<|im_start|>` is shown as a
        # marker, but the newline and the word "user" after it are shown as the text they are.
        self.added = set(self.tokenizer.get_added_vocab().values())
        self.thinks = "<think>" in self.tokenizer.get_vocab()
        log.info(f"assistant turn opens with {self.wrap('')[-40:]!r}"
                 + ("" if self.thinks else " (no <think> token in this vocabulary)"))

    def wrap(self, prompt: str) -> str:
        """Put one user prompt into a bare chat scaffold, with no system turn.

        :param prompt: what you typed.

        :return: the exact string the model is fed.
        """
        think = "<think>\n\n</think>\n\n" if self.thinks else ""
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{think}"

    @staticmethod
    def _catalogue(path: Path) -> list[dict]:
        """Concept names, in pair order."""
        import pyarrow.parquet as pq

        return [{"pair": row["pair"], "concept": row["concept"], "antagonist": row["antagonist"],
                 "class": row["class_name"]} for row in pq.read_table(path).to_pylist()]

    @staticmethod
    def _neighbours(unit: np.ndarray, count: int) -> list[list[int]]:
        """Nearest concepts by cosine, for the viewer's near-duplicate suppression."""
        similarity = unit @ unit.T
        np.fill_diagonal(similarity, -2.0)
        return np.argsort(-similarity, axis=1)[:, :count].tolist()

    def _hook(self, position: int) -> Any:
        """Project one block's residual stream onto every direction.

        A cosine, not a dot product: token norms vary by two orders of magnitude within a single
        sequence, so raw dot products would mostly measure norm. Disarmed during generation, where it
        would fire once per sampled token and capture nothing worth keeping.
        """
        def fire(module: Any, inputs: Any, output: Any) -> None:
            if not self.armed:
                return
            state = (output[0] if isinstance(output, tuple) else output).float()
            unit = state / torch.linalg.vector_norm(state, dim=-1, keepdim=True).clamp_min(1e-6)
            self.captured[position] = (unit @ self.vectors[position].T)[0]

        return fire

    def measure(self, ids: list[int]) -> dict:
        """One forward pass over exactly these ids, reduced to per-token z-scores per layer.

        :param ids: the token stream to read.

        :return: pieces, roles, and one packed plane per layer.
        """
        role = roles(ids, self.tokenizer)
        self.captured.clear()
        self.armed = True
        try:
            with torch.inference_mode():
                self.model.model(input_ids=torch.tensor([ids], device=self.device), use_cache=False)
        finally:
            self.armed = False

        # Statistics come from content tokens only; the scores of every token are then expressed
        # against them, scaffolding included. If the box holds nothing but scaffolding there is no
        # content to compare against, so everything counts and the shading is honest about it.
        keep = np.array([r != "template" for r in role])
        if not keep.any():
            keep = np.ones(len(role), dtype=bool)

        planes = {}
        for position, layer in enumerate(self.layers):
            plane = self.captured[position].float().cpu().numpy()
            centre = plane[keep].mean(axis=0, keepdims=True)
            spread = np.maximum(plane[keep].std(axis=0, keepdims=True), 1e-6)
            data, scale = quantise((plane - centre) / spread)
            planes[str(layer)] = {"data": data, "scale": scale}
        return {"pieces": self.tokenizer.convert_ids_to_tokens(ids), "role": role,
                "special": [i in self.added for i in ids],
                "tokens": len(ids), "content": int(keep.sum()), "layers": planes}

    def offsets(self, text: str) -> tuple[list[int], list[int]]:
        """Tokenize the box and report where each token starts in the text.

        The character offsets are what lets the front end mark which tokens the model wrote: the
        client remembers where the last generation began and everything at or beyond that point is
        the model's.

        :param text: the raw contents of the box.

        :return: token ids and the start offset of each.
        """
        got = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return got["input_ids"], [int(a) for a, _ in got["offset_mapping"]]

    def grow(self, prompt: str, params: dict) -> dict:
        """Answer one user prompt and read every token of the result.

        The readout runs on `ids + fresh` rather than on a re-tokenization of the joined text, so the
        measured stream is exactly the one the model saw, with no chance of a boundary shifting where
        the two halves meet.

        :param prompt: the user prompt, unscaffolded.
        :param params: generation settings from the front end.

        :return: the reply, the full fed string, the seed used, and the readout.
        """
        if not prompt.strip():
            raise ValueError("nothing to answer: the prompt is empty")
        opening = self.wrap(prompt)
        ids, starts = self.offsets(opening)

        seed = params.get("seed")
        if params.get("sample", True):
            seed = random.randrange(2 ** 31) if seed in (None, "") else int(seed)
            torch.manual_seed(seed)
            extra: dict[str, Any] = {"do_sample": True, "temperature": float(params["temperature"]),
                                     "top_p": float(params["top_p"]), "top_k": int(params["top_k"])}
        else:
            seed = None
            extra = {"do_sample": False}

        # Stop at the end of the turn. `<|im_end|>` is kept, so what is measured is a complete turn.
        stop = {self.tokenizer.eos_token_id}
        for name in ("<|im_end|>", "<|endoftext|>"):
            if (found := self.tokenizer.get_vocab().get(name)) is not None:
                stop.add(found)

        prompt = torch.tensor([ids], device=self.device)
        with torch.inference_mode():
            grown = self.model.generate(
                prompt, attention_mask=torch.ones_like(prompt),
                max_new_tokens=int(params.get("budget", 256)),
                eos_token_id=sorted(v for v in stop if v is not None),
                pad_token_id=self.model.generation_config.pad_token_id
                or self.tokenizer.eos_token_id, **extra)
        fresh = grown[0].tolist()[len(ids):]
        reply = self.tokenizer.decode(fresh)

        readout = self.measure(ids + fresh)
        # Offsets for the scaffolded prompt are known; every generated token is stamped with the
        # join, which is where the front end draws the line between prompt and reply.
        readout["offset"] = starts + [len(opening)] * len(fresh)
        return {"reply": reply, "text": opening + reply, "grown": len(opening), "seed": seed,
                "new": len(fresh), "readout": readout}


class Handler(BaseHTTPRequestHandler):
    """Static files plus two endpoints. One model, so requests are serialized on a lock."""

    live: Live
    settings: dict

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *rest: Any) -> None:
        log.info(f"{self.address_string()} {fmt % rest}")

    def _send(self, code: int, body: bytes, kind: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    def do_GET(self) -> None:
        route = self.path.split("?")[0]
        if route in ("/", "/index.html"):
            return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if route == "/app.js":
            return self._send(200, (HERE / "app.js").read_bytes(), "text/javascript; charset=utf-8")
        if route == "/style.css":
            return self._send(200, (HERE / "style.css").read_bytes(), "text/css; charset=utf-8")
        if route == "/meta":
            return self._json(self.settings)
        if route == "/concepts.json":
            return self._json(self.live.catalogue)
        if route == "/translations.json":
            path = self.settings.get("translations")
            if path and Path(path).exists():
                return self._send(200, Path(path).read_bytes(), "application/json")
            return self._json({"error": "no translations"}, 404)
        self._json({"error": f"no route {route}"}, 404)

    def do_POST(self) -> None:
        route = self.path.split("?")[0]
        if route != "/generate":
            return self._json({"error": f"no route {route}"}, 404)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        try:
            with self.live.lock:
                return self._json(self.live.grow(body.get("prompt", ""), body))
        except ValueError as error:
            return self._json({"error": str(error)}, 400)
        except Exception as error:                                  # noqa: BLE001
            log.exception("request failed")
            return self._json({"error": f"{type(error).__name__}: {error}"}, 500)


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                        datefmt="%H:%M:%S")
    args.rows = [int(v) for v in args.rows.split(",")]
    live = Live(args)
    Handler.live = live
    Handler.settings = {
        "model": args.model, "device": live.device, "dtype": str(live.dtype).replace("torch.", ""),
        "layers": live.layers, "pairs": len(live.catalogue["pairs"]), "span": SPAN,
        "synthetic": live.synthetic, "translations": str(args.translations) if args.translations else None,
        "depth": live.model.config.num_hidden_layers, "hidden": live.model.config.hidden_size,
        # Shown in the UI so the scaffolding is never a mystery: this is exactly what wraps a prompt.
        "scaffold": live.wrap("…"), "thinks": live.thinks,
    }
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    log.info(f"http://{args.host}:{args.port}/ — layers {live.layers}, "
             f"{len(live.catalogue['pairs'])} concepts"
             + (" — SYNTHETIC DIRECTIONS" if live.synthetic else ""))
    server.serve_forever()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--vectors", type=Path, default=Path("probes/diff.safetensors"))
    parser.add_argument("--pairs", type=Path, default=Path("probes/pairs.parquet"))
    parser.add_argument("--translations", type=Path, default=None)
    parser.add_argument("--layers", default="18,25", help="blocks to read, 1-based")
    parser.add_argument("--rows", default="11,14,18,22,25",
                        help="layers the vector file's first axis holds, in order")
    parser.add_argument("--synthetic", action="store_true",
                        help="random unit directions instead of the real ones, for testing the "
                             "pipeline against a model the vectors do not fit")
    parser.add_argument("--seed", type=int, default=0, help="seed for the synthetic directions")
    parser.add_argument("--neighbours", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    main(parser.parse_args())
