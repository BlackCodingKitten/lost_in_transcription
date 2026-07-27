#!/usr/bin/env python3
"""Paired analysis of MAIA 32-frame predictions with vs. without transcription."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def load_items(video_only_path: Path, transcript_path: Path) -> pd.DataFrame:
    with video_only_path.open(encoding="utf-8") as f:
        video_only = json.load(f)
    with transcript_path.open(encoding="utf-8") as f:
        with_transcript = json.load(f)

    rows: list[dict] = []
    for video_id, video_data in video_only.items():
        if video_id not in with_transcript:
            raise ValueError(f"Missing video in transcription file: {video_id}")
        trans_video = with_transcript[video_id]
        for question_id, pools in video_data["questions"].items():
            for pool_id, item_without in pools.items():
                try:
                    item_with = trans_video["questions"][question_id][pool_id]
                except KeyError as exc:
                    raise ValueError(
                        f"Missing paired item: {video_id}/{question_id}/{pool_id}"
                    ) from exc
                for key in ("target", "0", "1"):
                    if item_without.get(key) != item_with.get(key):
                        raise ValueError(
                            f"Item mismatch for {video_id}/{question_id}/{pool_id}: {key}"
                        )
                rows.append(
                    {
                        "video": video_id,
                        "audio_category": trans_video.get("classification", "unknown"),
                        "question_group": question_id,
                        "pool": pool_id,
                        "correct_without": bool(item_without["is_correct"]),
                        "correct_with": bool(item_with["is_correct"]),
                    }
                )
    return pd.DataFrame(rows)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (m - rank) * p_values[index])
        adjusted[index] = min(running_max, 1.0)
    return adjusted


def mcnemar_summary(group: pd.DataFrame) -> dict:
    both_correct = int((group.correct_without & group.correct_with).sum())
    correct_to_wrong = int((group.correct_without & ~group.correct_with).sum())
    wrong_to_correct = int((~group.correct_without & group.correct_with).sum())
    both_wrong = int((~group.correct_without & ~group.correct_with).sum())
    discordant = correct_to_wrong + wrong_to_correct
    p_value = (
        binomtest(
            min(correct_to_wrong, wrong_to_correct),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
        if discordant
        else 1.0
    )
    odds_ratio = correct_to_wrong / wrong_to_correct
    se = math.sqrt(1 / correct_to_wrong + 1 / wrong_to_correct)
    odds_low = math.exp(math.log(odds_ratio) - 1.96 * se)
    odds_high = math.exp(math.log(odds_ratio) + 1.96 * se)
    return {
        "N": len(group),
        "accuracy_without": group.correct_without.mean(),
        "accuracy_with": group.correct_with.mean(),
        "delta_pp": 100 * (group.correct_with.mean() - group.correct_without.mean()),
        "both_correct": both_correct,
        "correct_to_wrong": correct_to_wrong,
        "wrong_to_correct": wrong_to_correct,
        "both_wrong": both_wrong,
        "mcnemar_exact_p": p_value,
        "harm_to_benefit_odds_ratio": odds_ratio,
        "odds_ratio_95ci_low": odds_low,
        "odds_ratio_95ci_high": odds_high,
    }


def cluster_bootstrap_ci(values: np.ndarray, iterations: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    means = np.empty(iterations, dtype=float)
    chunk = 10_000
    for start in range(0, iterations, chunk):
        size = min(chunk, iterations - start)
        indices = rng.integers(0, n, size=(size, n))
        means[start : start + size] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def sign_flip_p(values: np.ndarray, iterations: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())
    exceedances = 0
    chunk = 10_000
    completed = 0
    while completed < iterations:
        size = min(chunk, iterations - completed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, len(values)))
        statistics = np.abs((signs * values).mean(axis=1))
        exceedances += int(np.sum(statistics >= observed - 1e-15))
        completed += size
    return (exceedances + 1) / (iterations + 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_only", type=Path)
    parser.add_argument("with_transcription", type=Path)
    parser.add_argument("--output", type=Path, default=Path("mcnemar_results.csv"))
    parser.add_argument("--bootstrap", type=int, default=100_000)
    parser.add_argument("--permutations", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    items = load_items(args.video_only, args.with_transcription)
    records: list[dict] = []

    overall = mcnemar_summary(items)
    overall["category"] = "Overall"
    records.append(overall)

    category_order = ["dialogue", "unknown_dialogue", "music", "silence/noise"]
    for category in category_order:
        summary = mcnemar_summary(items[items.audio_category == category])
        summary["category"] = category
        records.append(summary)

    results = pd.DataFrame(records)
    results["holm_p"] = np.nan
    results.loc[1:, "holm_p"] = holm_adjust(results.loc[1:, "mcnemar_exact_p"].to_numpy())

    per_video = (
        items.groupby(["video", "audio_category"])
        .agg(
            accuracy_without=("correct_without", "mean"),
            accuracy_with=("correct_with", "mean"),
        )
        .reset_index()
    )
    per_video["delta_pp"] = 100 * (per_video.accuracy_with - per_video.accuracy_without)

    for index, row in results.iterrows():
        values = (
            per_video.delta_pp.to_numpy()
            if row.category == "Overall"
            else per_video.loc[per_video.audio_category == row.category, "delta_pp"].to_numpy()
        )
        low, high = cluster_bootstrap_ci(values, args.bootstrap, args.seed + index)
        results.loc[index, "cluster_ci_low"] = low
        results.loc[index, "cluster_ci_high"] = high
        results.loc[index, "sign_flip_p"] = sign_flip_p(
            values, args.permutations, args.seed + 100 + index
        )

    results.loc[1:, "sign_flip_holm_p"] = holm_adjust(
        results.loc[1:, "sign_flip_p"].to_numpy()
    )
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False))
    print(f"\nMatched items: {len(items):,}; videos: {items.video.nunique()}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
