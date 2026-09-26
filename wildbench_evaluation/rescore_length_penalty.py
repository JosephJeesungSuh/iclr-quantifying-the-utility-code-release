"""
Re-score a wildbench_pairwise details JSONL with a different length_penalty_K,
without re-running any API calls.

Usage:
    python rescore_length_penalty.py \
        --input results/foo_vs_bar_details.jsonl \
        --length_penalty_K 1000
"""

import argparse
import json
from pathlib import Path


def apply_length_penalty(choice: str, response_A: str, response_B: str, K: int) -> str:
    if K <= 0:
        return choice
    len_A, len_B = len(response_A), len(response_B)
    if choice == "A+" and (len_A - len_B) > K:
        return "A=B"
    if choice == "B+" and (len_B - len_A) > K:
        return "A=B"
    return choice


def choice_to_score(choice: str) -> float:
    return {"A++": 1.0, "A+": 0.5, "A=B": 0.0, "B+": -0.5, "B++": -1.0}.get(choice, 0.0)


def rescore(input_path: str, K: int) -> None:
    records = []
    with open(input_path) as f:
        for line in f:
            records.append(json.loads(line))

    scores = []
    for r in records:
        if r["judge_failed"]:
            r["final_score_model1"] = None
            continue

        r1 = r["model1_output"]
        r2 = r["model2_output"]

        j1 = r["judge_order1"]
        j1["penalized_choice"] = apply_length_penalty(j1["raw_choice"], r1, r2, K)
        j1["score"] = choice_to_score(j1["penalized_choice"])

        j2 = r["judge_order2"]
        if j2 is not None:
            # order2 has A=model2, B=model1, so responses are swapped
            j2["penalized_choice"] = apply_length_penalty(j2["raw_choice"], r2, r1, K)
            j2["score"] = choice_to_score(j2["penalized_choice"])
            score_m1 = (j1["score"] + (-j2["score"])) / 2.0
        else:
            score_m1 = j1["score"]

        r["final_score_model1"] = score_m1
        scores.append(score_m1)

    n = len(scores)
    avg_reward = sum(scores) / n if n else 0.0
    wins = sum(1 for s in scores if s > 0)
    ties = sum(1 for s in scores if s == 0)
    losses = sum(1 for s in scores if s < 0)
    win_rate = wins / n if n else 0.0

    print(f"length_penalty_K : {K}")
    print(f"Tasks evaluated  : {n}")
    print(f"WB-Reward        : {avg_reward * 100:+.2f}")
    print(f"Win rate (model1): {win_rate * 100:.1f}%")
    print(f"Wins / Ties / Losses: {wins} / {ties} / {losses}")

    # Write updated details
    out_path = Path(input_path).with_name(
        Path(input_path).stem + f"_K{K}.jsonl"
    )
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Updated details written to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to _details.jsonl file")
    parser.add_argument("--length_penalty_K", type=int, default=1000)
    args = parser.parse_args()
    rescore(args.input, args.length_penalty_K)


if __name__ == "__main__":
    main()
