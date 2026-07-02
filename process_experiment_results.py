from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Experiment output folder.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for aggregate tables and figures.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def read_eval_rows(experiment_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for eval_path in sorted(experiment_dir.glob("*/eval_results.jsonl")):
        run_dir = eval_path.parent
        run_config = load_json(run_dir / "config.json")
        algorithm = run_config.get("algorithm", run_dir.name)
        run_name = run_config.get("name", run_dir.name)
        train_seed = run_config.get("args", {}).get("seed")

        for line in eval_path.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            eval_seeds = event.get("eval_seeds", [])
            episodic_returns = event.get("episodic_returns", [])
            episodic_lengths = event.get("episodic_lengths", [])
            average_overestimations = event.get("average_overestimations", [])
            start_overestimations = event.get("start_overestimations", [])
            for index, episodic_return in enumerate(episodic_returns):
                if index < len(episodic_lengths):
                    episodic_length = float(episodic_lengths[index])
                elif event.get("env_id", "") == "CartPole-v1":
                    episodic_length = float(episodic_return)
                else:
                    episodic_length = ""
                rows.append(
                    {
                        "algorithm": algorithm,
                        "run_name": run_name,
                        "train_seed": event.get("train_seed", train_seed),
                        "global_step": int(event["global_step"]),
                        "env_id": event.get("env_id", ""),
                        "eval_seed": eval_seeds[index] if index < len(eval_seeds) else index,
                        "episodic_return": float(episodic_return),
                        "episodic_length": episodic_length,
                        "average_overestimation": (
                            float(average_overestimations[index])
                            if index < len(average_overestimations)
                            else ""
                        ),
                        "start_overestimation": (
                            float(start_overestimations[index])
                            if index < len(start_overestimations)
                            else ""
                        ),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return_groups = defaultdict(list)
    length_groups = defaultdict(list)
    average_overestimation_groups = defaultdict(list)
    start_overestimation_groups = defaultdict(list)
    for row in rows:
        key = (row["algorithm"], row["global_step"])
        return_groups[key].append(row["episodic_return"])
        if row["episodic_length"] != "":
            length_groups[key].append(row["episodic_length"])
        if row["average_overestimation"] != "":
            average_overestimation_groups[key].append(row["average_overestimation"])
        if row["start_overestimation"] != "":
            start_overestimation_groups[key].append(row["start_overestimation"])

    aggregates = []
    for (algorithm, global_step), values in sorted(return_groups.items()):
        value_std = stdev(values) if len(values) > 1 else 0.0
        lengths = length_groups[(algorithm, global_step)]
        average_overestimations = average_overestimation_groups[(algorithm, global_step)]
        start_overestimations = start_overestimation_groups[(algorithm, global_step)]
        length_std = stdev(lengths) if len(lengths) > 1 else (0.0 if lengths else "")
        average_overestimation_std = (
            stdev(average_overestimations) if len(average_overestimations) > 1 else (0.0 if average_overestimations else "")
        )
        start_overestimation_std = (
            stdev(start_overestimations) if len(start_overestimations) > 1 else (0.0 if start_overestimations else "")
        )
        aggregates.append(
            {
                "algorithm": algorithm,
                "global_step": global_step,
                "mean_return": mean(values),
                "std_return": value_std,
                "sem_return": value_std / math.sqrt(len(values)) if values else 0.0,
                "mean_length": mean(lengths) if lengths else "",
                "std_length": length_std,
                "sem_length": length_std / math.sqrt(len(lengths)) if lengths else "",
                "mean_average_overestimation": (
                    mean(average_overestimations) if average_overestimations else ""
                ),
                "std_average_overestimation": average_overestimation_std,
                "sem_average_overestimation": (
                    average_overestimation_std / math.sqrt(len(average_overestimations))
                    if average_overestimations
                    else ""
                ),
                "min_average_overestimation": (
                    min(average_overestimations) if average_overestimations else ""
                ),
                "max_average_overestimation": (
                    max(average_overestimations) if average_overestimations else ""
                ),
                "mean_start_overestimation": (
                    mean(start_overestimations) if start_overestimations else ""
                ),
                "std_start_overestimation": start_overestimation_std,
                "sem_start_overestimation": (
                    start_overestimation_std / math.sqrt(len(start_overestimations))
                    if start_overestimations
                    else ""
                ),
                "min_start_overestimation": (
                    min(start_overestimations) if start_overestimations else ""
                ),
                "max_start_overestimation": (
                    max(start_overestimations) if start_overestimations else ""
                ),
                "num_returns": len(values),
            }
        )
    return aggregates


def final_summary_rows(aggregate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_by_algorithm = {}
    for row in aggregate_rows:
        algorithm = row["algorithm"]
        if algorithm not in latest_by_algorithm or row["global_step"] > latest_by_algorithm[algorithm]["global_step"]:
            latest_by_algorithm[algorithm] = row
    return [latest_by_algorithm[algorithm] for algorithm in sorted(latest_by_algorithm)]


def default_output_dir(input_dir: Path) -> Path:
    if input_dir.name == "runs":
        return input_dir.parent / "results"
    return input_dir / "results"


def setup_matplotlib(output_dir: Path) -> None:
    matplotlib_config_dir = output_dir / ".matplotlib"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))


def plot_metric(
    rows: list[dict[str, Any]],
    output_path: Path,
    mean_key: str,
    spread_key: str,
    ylabel: str,
) -> None:
    matplotlib_config_dir = output_path.parent / ".matplotlib"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_algorithm = defaultdict(list)
    for row in rows:
        by_algorithm[row["algorithm"]].append(row)

    fig, ax = plt.subplots(figsize=(8, 5))
    for algorithm, algorithm_rows in sorted(by_algorithm.items()):
        algorithm_rows = sorted(algorithm_rows, key=lambda row: row["global_step"])
        algorithm_rows = [row for row in algorithm_rows if row[mean_key] != ""]
        if not algorithm_rows:
            continue
        steps = [row["global_step"] for row in algorithm_rows]
        means = [row[mean_key] for row in algorithm_rows]
        sems = [row[spread_key] for row in algorithm_rows]
        lower = [value - sem for value, sem in zip(means, sems)]
        upper = [value + sem for value, sem in zip(means, sems)]
        ax.plot(steps, means, label=algorithm)
        ax.fill_between(steps, lower, upper, alpha=0.2)

    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_all_metrics(rows: list[dict[str, Any]], output_dir: Path) -> None:
    plot_metric(
        rows,
        output_dir / "evaluation_returns.png",
        "mean_return",
        "sem_return",
        "Evaluation return",
    )
    plot_metric(
        rows,
        output_dir / "evaluation_lengths.png",
        "mean_length",
        "sem_length",
        "Episode length",
    )
    plot_metric(
        rows,
        output_dir / "average_overestimation.png",
        "mean_average_overestimation",
        "sem_average_overestimation",
        "Average overestimation",
    )
    plot_metric(
        rows,
        output_dir / "start_overestimation.png",
        "mean_start_overestimation",
        "sem_start_overestimation",
        "Start-state overestimation",
    )


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.folder).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(experiment_dir)

    eval_rows = read_eval_rows(experiment_dir)
    if not eval_rows:
        raise SystemExit(f"No eval_results.jsonl files found under {experiment_dir}.")

    aggregate = aggregate_rows(eval_rows)
    write_csv(
        output_dir / "evaluation_returns.csv",
        eval_rows,
        [
            "algorithm",
            "run_name",
            "train_seed",
            "global_step",
            "env_id",
            "eval_seed",
            "episodic_return",
            "episodic_length",
            "average_overestimation",
            "start_overestimation",
        ],
    )
    write_csv(
        output_dir / "evaluation_summary.csv",
        aggregate,
        [
            "algorithm",
            "global_step",
            "mean_return",
            "std_return",
            "sem_return",
            "mean_length",
            "std_length",
            "sem_length",
            "mean_average_overestimation",
            "std_average_overestimation",
            "sem_average_overestimation",
            "min_average_overestimation",
            "max_average_overestimation",
            "mean_start_overestimation",
            "std_start_overestimation",
            "sem_start_overestimation",
            "min_start_overestimation",
            "max_start_overestimation",
            "num_returns",
        ],
    )
    write_csv(
        output_dir / "final_summary.csv",
        final_summary_rows(aggregate),
        [
            "algorithm",
            "global_step",
            "mean_return",
            "std_return",
            "sem_return",
            "mean_length",
            "std_length",
            "sem_length",
            "mean_average_overestimation",
            "std_average_overestimation",
            "sem_average_overestimation",
            "min_average_overestimation",
            "max_average_overestimation",
            "mean_start_overestimation",
            "std_start_overestimation",
            "sem_start_overestimation",
            "min_start_overestimation",
            "max_start_overestimation",
            "num_returns",
        ],
    )
    plot_all_metrics(aggregate, output_dir)
    print(f"wrote results to {output_dir}")


if __name__ == "__main__":
    main()
