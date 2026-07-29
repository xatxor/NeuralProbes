#! /usr/bin/env python

"""Load Qwen3-8B for the agentic probes and verify it survives fp16.

The V100s are compute capability 7.0 and have no bfloat16, but Qwen3-8B is a native bfloat16 model,
so every run here is a downcast into a format with a far narrower exponent range. Running this file
directly loads the model, generates once, and asserts that nothing overflowed at the two layers we
read concepts from.
"""

import argparse
import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger("model")

MODEL = "Qwen/Qwen3-8B"
# The published vectors name layers one-indexed: row L18 is the output of model.layers[17].
LAYERS = (18, 25)
# Matches screen.py and jailbreak.py, by decision, so episodes stay comparable to the earlier runs.
SAMPLING = {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20}


def load(model_id: str = MODEL, device: str = "cuda:0", dtype: str = "float16") -> tuple[Any, Any]:
    """Load the model pinned to one GPU, plus its tokenizer.

    fp16 is the default everywhere, including on hardware that supports bfloat16. The V100s have no
    bf16 at all, so every episode on disk is fp16; running the A100s in bf16 would make the two sets
    differ by numerics as well as by hardware, and nothing would be attributable. Pass
    `dtype="bfloat16"` only for a deliberate arm.

    :param model_id: HF model id, resolved from the local cache.
    :param device: the single device to place every weight on.
    :param dtype: "float16" or "bfloat16".

    :return: the model in eval mode and its tokenizer.
    """
    wanted = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    if wanted is torch.bfloat16 and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise SystemExit("bfloat16 requested but this device cannot do it natively")
    log.info(f"loading as {dtype} (device reports bf16 support: {torch.cuda.is_bf16_supported(including_emulation=False)})")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=wanted,
        device_map={"": device},
        # The vectors were extracted under sdpa; changing this changes what we are measuring.
        attn_implementation="sdpa",
    )
    model.eval()
    return model, tokenizer


def watch(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output: torch.Tensor | tuple[torch.Tensor, ...],
    layer: int,
    seen: dict[int, tuple[bool, float, float]],
) -> None:
    """Record whether one block's residual stream stayed finite, as a forward hook.

    :param module: the block this hook is attached to; required by the hook protocol, unused.
    :param inputs: the block's positional inputs; required by the hook protocol, unused.
    :param output: the block's output, either the residual stream or a tuple starting with it.
    :param layer: the one-indexed layer number, used as the key written to `seen`.
    :param seen: mapping the hook writes into, read by the caller after the forward.

    :return: None; hooks that return None leave the block's output untouched.
    """
    state = (output[0] if isinstance(output, tuple) else output).float()
    norm = torch.linalg.vector_norm(state, dim=-1)
    finite = bool(torch.isfinite(state).all())
    previous = seen.get(layer)
    seen[layer] = (
        finite and (previous[0] if previous else True),
        max(float(state.abs().max()), previous[1] if previous else 0.0),
        float(norm.mean()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log.info(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, capability {torch.cuda.get_device_capability(0)}")

    model, tokenizer = load(args.model, args.device, args.dtype)
    log.info(f"loaded {args.model} as {model.dtype}, {torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated")

    # Thinking is on by decision: the branch point in an agentic episode lives in the deliberation.
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "A test asserts that summing a 100000-element list takes under 0.0001 seconds. Is that achievable in pure Python?"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(args.device)

    seen: dict[int, tuple[bool, float, float]] = {}
    handles = [
        model.model.layers[layer - 1].register_forward_hook(
            lambda module, inputs, output, layer=layer: watch(module, inputs, output, layer, seen)
        )
        for layer in LAYERS
    ]
    try:
        torch.manual_seed(args.seed)
        with torch.inference_mode():
            sequence = model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=args.new_tokens,
                pad_token_id=model.generation_config.pad_token_id,
                **SAMPLING,
            )
    finally:
        for handle in handles:
            handle.remove()

    for layer in LAYERS:
        finite, peak, norm = seen[layer]
        log.info(f"L{layer}: finite={finite} peak|h|={peak:.1f} mean norm={norm:.2f}")

    completion = tokenizer.decode(sequence[0, ids.shape[1] :], skip_special_tokens=False)
    log.info(f"generated {sequence.shape[1] - ids.shape[1]} tokens, thinking present: {'<think>' in completion}")
    print("-" * 72)
    print(completion)
    print("-" * 72)

    if not all(seen[layer][0] for layer in LAYERS):
        raise SystemExit("fp16 overflowed the residual stream; fall back to fp32 across two GPUs")
    log.info("fp16 held at both readout layers")


if __name__ == "__main__":
    main()
