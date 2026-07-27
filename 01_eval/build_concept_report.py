"""Turn concept-analysis evaluator outputs into tables and a no-dependency local viewer."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from concept_analysis import VECTOR_REPO, VECTOR_REVISION

ROOT = Path(__file__).resolve().parent
BENCHMARKS = ("aime_2024", "math_500", "gpqa_diamond")


def records(path: Path, benchmark: str) -> list[dict]:
    file = path / f"{benchmark}.jsonl"
    if not file.exists():
        return []
    return [json.loads(line) | {"benchmark": benchmark} for line in file.read_text().splitlines() if line]


def pearson_binary(frame: pd.DataFrame) -> float | None:
    if len(frame) < 3 or frame.correct.nunique() != 2:
        return None
    return float(frame["correct"].astype(float).corr(frame["mean_cosine"]))


def correlations(scores: pd.DataFrame) -> pd.DataFrame:
    def one(group: pd.DataFrame) -> pd.Series:
        yes = group.loc[group.correct, "mean_cosine"]
        no = group.loc[~group.correct, "mean_cosine"]
        return pd.Series({"n": len(group), "accuracy": float(group.correct.mean()), "correlation": pearson_binary(group), "correct_mean": yes.mean(), "incorrect_mean": no.mean(), "mean_difference": yes.mean() - no.mean()})

    pieces = []
    for scope, frame in [("pooled", scores), *[(name, scores[scores.benchmark == name]) for name in BENCHMARKS]]:
        part = frame.groupby(["method", "layer", "pair"], sort=False).apply(one, include_groups=False).reset_index()
        part.insert(0, "scope", scope)
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def write_viewer(results: Path, response_rows: list[dict], scores: pd.DataFrame, selected: pd.DataFrame, pairs: pd.DataFrame, correlations_table: pd.DataFrame) -> None:
    viewer = results / "concept_viewer"
    viewer.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "concept_viewer.html", viewer / "index.html")
    index = {
        "pairs": pairs[["pair", "concept", "antagonist", "class_name"]].to_dict("records"),
        "responses": [],
        "correlations": json.loads(correlations_table.to_json(orient="records")),
    }
    for row in response_rows:
        if "reasoning_tokens" not in row:
            continue
        key = f"{row['benchmark']}-{row['id']}"
        subset = selected[(selected.benchmark == row["benchmark"]) & (selected.id.astype(str) == str(row["id"]))]
        response_scores = scores[(scores.benchmark == row["benchmark"]) & (scores.id.astype(str) == str(row["id"]))]
        rankings = {}
        for (method, layer), group in response_scores.groupby(["method", "layer"]):
            ranked = group.sort_values("mean_cosine", ascending=False)[["pair", "mean_cosine"]].rename(columns={"mean_cosine": "activation"})
            ranked["direction"] = ranked["activation"].map(lambda value: "positive" if value >= 0 else "negative")
            rankings[f"{method}:{layer}"] = ranked.to_dict("records")
        (viewer / f"{key}.json").write_text(json.dumps({"tokens": row["reasoning_tokens"], "color_scales": row.get("activation_color_scales", {}), "rankings": rankings, "activations": subset.to_dict("records")}, ensure_ascii=False))
        index["responses"].append({"key": key, "benchmark": row["benchmark"], "id": str(row["id"]), "correct": row.get("correct"), "reasoning_status": row.get("reasoning_status")})
    (viewer / "index.json").write_text(json.dumps(index, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    response_rows = [row for benchmark in BENCHMARKS for row in records(args.results, benchmark)]
    response = pd.DataFrame(response_rows)
    score_files = [args.results / f"concept_scores-{benchmark}.parquet" for benchmark in BENCHMARKS]
    score_files = [path for path in score_files if path.exists()]
    if response.empty or not score_files:
        raise SystemExit("No concept-analysis results found. Run evaluate.py with --concept-analysis first.")
    scores = pd.concat([pd.read_parquet(path) for path in score_files], ignore_index=True)
    scores = scores.merge(response[["benchmark", "id", "correct"]], on=["benchmark", "id"], how="left", validate="many_to_one")
    pairs = pd.read_parquet(hf_hub_download(VECTOR_REPO, "pairs.parquet", revision=VECTOR_REVISION))
    correlations_table = correlations(scores).merge(pairs[["pair", "concept", "antagonist", "class_name"]], on="pair", how="left")
    scores.to_parquet(args.results / "concept_scores.parquet", index=False)
    correlations_table.to_parquet(args.results / "correlations.parquet", index=False)
    correlations_table.to_csv(args.results / "correlations.csv", index=False)
    highlight_files = [args.results / f"token_highlights-{benchmark}.parquet" for benchmark in BENCHMARKS]
    highlights = pd.concat([pd.read_parquet(path) for path in highlight_files if path.exists()], ignore_index=True)
    highlights.to_parquet(args.results / "token_highlights.parquet", index=False)
    selected_files = [args.results / f"token_cosines-{benchmark}.parquet" for benchmark in BENCHMARKS]
    selected = pd.concat([pd.read_parquet(path) for path in selected_files if path.exists()], ignore_index=True)
    selected.to_parquet(args.results / "token_cosines.parquet", index=False)
    write_viewer(args.results, response_rows, scores, selected, pairs, correlations_table)
    print(f"Wrote {args.results / 'correlations.parquet'} and {args.results / 'concept_viewer' / 'index.html'}")


if __name__ == "__main__":
    main()
