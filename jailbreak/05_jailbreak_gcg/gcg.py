"""Qwen-chat-template-native Faster-GCG attack."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass(frozen=True)
class GCGConfig:
    steps: int = 500
    suffix_tokens: int = 40
    batch_size: int = 64
    topk: int = 32
    candidate_chunk_size: int = 32
    seed: int = 42
    distance_penalty: float = 10.0
    temperature: float = 0.1


@dataclass
class PromptState:
    prompt: str
    input_ids: torch.Tensor
    control_slice: slice
    target_slice: slice


def rendered_prompt(tokenizer: Any, prompt: str, suffix: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt + suffix}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_state(tokenizer: Any, prompt: str, control: torch.Tensor, target: str) -> PromptState | None:
    """Render a candidate and find its exact token and target boundaries.

    Candidates that do not survive decode/render/tokenize as the same token IDs
    return ``None``. This is the Qwen-safe equivalent of llm-attacks' filtering.
    """
    suffix = tokenizer.decode(control.tolist(), skip_special_tokens=False)
    rendered = rendered_prompt(tokenizer, prompt, suffix)
    suffix_start = rendered.rfind(suffix)
    if suffix_start < 0:
        return None
    suffix_end = suffix_start + len(suffix)
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors=None,
    )
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    positions = [i for i, (start, end) in enumerate(offsets) if start >= suffix_start and end <= suffix_end and end > start]
    if not positions or positions != list(range(positions[0], positions[-1] + 1)):
        return None
    candidate = torch.tensor(ids[positions[0] : positions[-1] + 1], dtype=torch.long)
    if not torch.equal(candidate.cpu(), control.cpu()):
        return None
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if not target_ids:
        raise ValueError("GCG target tokenized to an empty sequence")
    all_ids = torch.tensor(ids + target_ids, dtype=torch.long)
    return PromptState(prompt, all_ids, slice(positions[0], positions[-1] + 1), slice(len(ids), len(ids) + len(target_ids)))


def allowed_token_ids(tokenizer: Any) -> torch.Tensor:
    """ASCII, non-special tokens only, following the original attack's filter."""
    special = set(tokenizer.all_special_ids)
    allowed = []
    for token_id in range(tokenizer.vocab_size):
        if token_id in special:
            continue
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        if text and text.isascii() and text.isprintable():
            allowed.append(token_id)
    if not allowed:
        raise ValueError("Tokenizer has no printable non-special tokens")
    return torch.tensor(allowed, dtype=torch.long)


def initial_control(tokenizer: Any, prompt: str, target: str, length: int) -> torch.Tensor:
    """Choose a repeated printable token that survives Qwen's re-tokenization."""
    for token_id in tokenizer.encode("!", add_special_tokens=False) + allowed_token_ids(tokenizer).tolist():
        control = torch.full((length,), token_id, dtype=torch.long)
        if build_state(tokenizer, prompt, control, target) is not None:
            return control
    raise RuntimeError("Could not find a stable printable GCG initializer")


def token_grad(model: Any, state: PromptState, device: torch.device) -> torch.Tensor:
    ids = state.input_ids.to(device)
    control = ids[state.control_slice]
    embed = model.get_input_embeddings()
    one_hot = F.one_hot(control, num_classes=embed.num_embeddings).to(dtype=embed.weight.dtype, device=device).detach()
    one_hot.requires_grad_()
    inputs_embeds = embed(ids).detach()
    inputs_embeds[state.control_slice] = one_hot @ embed.weight
    logits = model(inputs_embeds=inputs_embeds.unsqueeze(0), use_cache=False).logits[0]
    target = ids[state.target_slice]
    loss = F.cross_entropy(logits[state.target_slice.start - 1 : state.target_slice.stop - 1], target)
    gradient = torch.autograd.grad(loss, one_hot, only_inputs=True)[0].detach()
    return gradient


@torch.inference_mode()
def loss_sums(model: Any, states: list[PromptState], candidates: list[torch.Tensor], device: torch.device, chunk_size: int) -> torch.Tensor:
    total = torch.zeros(len(candidates), device=device)
    for state in states:
        for start in range(0, len(candidates), chunk_size):
            chunk = candidates[start : start + chunk_size]
            batch = torch.stack([state.input_ids.clone() for _ in chunk]).to(device)
            for row, candidate in enumerate(chunk):
                batch[row, state.control_slice] = candidate.to(device)
            logits = model(input_ids=batch, use_cache=False).logits
            target = batch[:, state.target_slice]
            value = F.cross_entropy(
                logits[:, state.target_slice.start - 1 : state.target_slice.stop - 1].transpose(1, 2),
                target,
                reduction="none",
            ).mean(dim=1)
            total[start : start + len(chunk)] += value
    return total


def sample_candidates(
    control: torch.Tensor,
    gradient: torch.Tensor,
    distances: torch.Tensor,
    allowed: torch.Tensor,
    batch_size: int,
    topk: int,
    temperature: float,
    distance_penalty: float,
    seed: int,
    evaluated: set[tuple[int, ...]],
) -> list[torch.Tensor]:
    """Faster-GCG distance ranking, temperature sampling, and loop avoidance."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    masked = torch.full_like(gradient, torch.inf)
    masked[:, allowed] = gradient[:, allowed]
    topk_values, choices = torch.topk(-masked, k=min(topk, len(allowed)), dim=1)
    topk_distances = torch.gather(distances, 1, choices)
    order = torch.argsort(topk_values + distance_penalty * topk_distances, dim=1)
    choices = torch.gather(choices, 1, order).cpu()
    weights = torch.pow(torch.arange(choices.shape[1], 0, -1, dtype=torch.float32) / choices.shape[1], 1 / temperature)
    positions = torch.randint(len(control), (batch_size,), generator=generator)
    sampled_ranks = torch.empty(batch_size, dtype=torch.long)
    for position in positions.unique().tolist():
        mask = positions == position
        sampled_ranks[mask] = torch.multinomial(weights, int(mask.sum()), replacement=False, generator=generator)
    rows = []
    seen: set[tuple[int, ...]] = set()
    for position, rank in zip(positions.tolist(), sampled_ranks.tolist(), strict=True):
        candidate = control.clone()
        candidate[position] = choices[position, rank]
        key = tuple(candidate.tolist())
        if key in evaluated or key in seen:
            continue
        seen.add(key)
        rows.append(candidate)
    return rows


def _checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def distributed_context() -> tuple[int, int]:
    return (dist.get_rank(), dist.get_world_size()) if dist.is_available() and dist.is_initialized() else (0, 1)


def optimize(model: Any, tokenizer: Any, examples: list[dict[str, str]], output: Path, config: GCGConfig, device: torch.device) -> dict[str, Any]:
    """Optimize one suffix against all prompts; safely resumes from attack.json."""
    if not examples:
        raise ValueError("GCG needs at least one training prompt")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    rank, world_size = distributed_context()
    local_examples = examples[rank::world_size]
    if not local_examples:
        raise ValueError("Each GCG worker needs at least one training prompt")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "attack.json"
    saved = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
    control = torch.tensor(saved["control_token_ids"], dtype=torch.long) if saved else initial_control(tokenizer, examples[0]["prompt"], examples[0]["target"], config.suffix_tokens)
    start_step = int(saved.get("completed_steps", 0)) if saved else 0
    history = list(saved.get("history", [])) if saved else []
    allowed = allowed_token_ids(tokenizer)
    evaluated: set[tuple[int, ...]] = {tuple(control.tolist())}
    embedding = model.get_input_embeddings().weight.detach()
    embedding_norm = (embedding.float() ** 2).sum(dim=1)
    started = time.perf_counter()
    for step in range(start_step, config.steps):
        states = [build_state(tokenizer, row["prompt"], control, row["target"]) for row in local_examples]
        if any(state is None for state in states):
            raise RuntimeError("Current suffix no longer has stable Qwen token boundaries")
        states = [state for state in states if state is not None]
        gradient = torch.stack([token_grad(model, state, device) for state in states]).sum(dim=0)
        if world_size > 1:
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        gradient = (gradient / len(examples)).cpu()
        current = embedding[control.to(device)]
        distances = torch.sqrt(torch.clamp(
            embedding_norm[None, :] + (current.float() ** 2).sum(dim=1)[:, None] - 2 * (current @ embedding.T).float(),
            min=0,
        )).cpu()
        distances /= distances.norm(dim=1, keepdim=True).clamp_min(1e-6)
        gradient /= gradient.norm(dim=1, keepdim=True).clamp_min(1e-6)
        raw = sample_candidates(
            control, gradient, distances, allowed, config.batch_size, config.topk,
            config.temperature, config.distance_penalty, config.seed + step, evaluated,
        )
        candidates = [
            candidate for candidate in raw
            if all(build_state(tokenizer, row["prompt"], candidate, row["target"]) is not None for row in examples)
        ]
        if not candidates:
            raise RuntimeError("No GCG candidates retained after Qwen token-boundary filtering")
        candidate_states = [build_state(tokenizer, row["prompt"], control, row["target"]) for row in local_examples]
        value = loss_sums(model, [state for state in candidate_states if state is not None], candidates, device, config.candidate_chunk_size)
        if world_size > 1:
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / len(examples)
        evaluated.update(tuple(candidate.tolist()) for candidate in candidates)
        best = int(value.argmin())
        control = candidates[best]
        record = {"step": step + 1, "loss": float(value[best]), "suffix": tokenizer.decode(control.tolist(), skip_special_tokens=False)}
        history.append(record)
        payload = {
            "target_mode": "AdvBench instruction-specific targets",
            "config": asdict(config),
            "completed_steps": step + 1,
            "control_token_ids": control.tolist(),
            "suffix": record["suffix"],
            "history": history,
            "elapsed_seconds": (saved or {}).get("elapsed_seconds", 0.0) + time.perf_counter() - started,
        }
        if rank == 0:
            _checkpoint(checkpoint, payload)
            print(f"GCG step {step + 1}/{config.steps}: loss={record['loss']:.4f}", flush=True)
        if world_size > 1:
            dist.barrier()
    if world_size > 1:
        dist.barrier()
    return json.loads(checkpoint.read_text())
