from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ALGORITHM_SCRIPTS = {
    "avl": "cleanrl/avl.py",
    "ddqn": "cleanrl/ddqn.py",
    "dqn": "cleanrl/dqn.py",
    "duelql": "cleanrl/duelql.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Experiment folder or config JSON path.")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Run only algorithm run names matching this shell-style pattern. Repeatable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write commands and summaries without launching training processes.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print live progress updates from algorithm eval events.",
    )
    return parser.parse_args()


def resolve_config_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate

    json_files = sorted(
        path
        for path in candidate.glob("*.json")
        if path.name not in {"batch_summary.json", "experiment_config.json"}
    )
    if len(json_files) != 1:
        raise ValueError(f"Expected exactly one JSON config in {candidate}, found {len(json_files)}.")
    return json_files[0]


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = resolve_config_path(path)
    return json.loads(config_path.read_text()), config_path


def make_command(script: Path, args: dict[str, Any]) -> list[str]:
    command = [sys.executable, str(script)]
    for key, value in args.items():
        flag = f"--{key.replace('_', '-')}"
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                command.append(flag)
            else:
                command.append(f"--no-{key.replace('_', '-')}")
        elif key == "eval_seeds" and isinstance(value, (list, tuple)):
            command.extend([flag, ",".join(str(item) for item in value)])
        elif isinstance(value, (list, tuple)):
            command.append(flag)
            command.extend(str(item) for item in value)
        else:
            command.extend([flag, str(value)])
    return command


def resolve_script(algorithm_run: dict[str, Any], root: Path) -> Path:
    if algorithm_run.get("script"):
        script = Path(algorithm_run["script"])
        return script if script.is_absolute() else root / script

    algorithm = algorithm_run.get("algorithm")
    if algorithm not in ALGORITHM_SCRIPTS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Add script=... for custom runs.")
    return root / ALGORITHM_SCRIPTS[algorithm]


def merged_args(config: dict[str, Any], algorithm_config: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    environment = dict(config.get("environment", {}))
    shared_args = dict(config.get("shared_args", {}))
    evaluation = dict(config.get("evaluation", {}))
    algorithm_args = dict(algorithm_config.get("args", {}))
    variant_args = dict(variant.get("args", {}))

    args = {}
    args.update(environment)
    args.update(shared_args)
    args.update(evaluation)
    args.update(algorithm_args)
    args.update(variant_args)
    return args


def expand_algorithm_runs(config: dict[str, Any]) -> list[dict[str, Any]]:
    runs = []
    for algorithm_config in config["algorithms"]:
        algorithm = algorithm_config.get("algorithm")
        variants = algorithm_config.get("variants") or [{}]
        copies = int(algorithm_config.get("copies", 1))

        for variant_index, variant in enumerate(variants):
            args = merged_args(config, algorithm_config, variant)
            seeds = algorithm_config.get("seeds")
            if seeds is None:
                seeds = [args.get("seed", copy_index + 1) for copy_index in range(copies)]

            variant_name = variant.get("name", f"variant{variant_index}")
            for copy_index, seed in enumerate(seeds):
                run_args = dict(args)
                if seed is not None:
                    run_args["seed"] = seed

                name_template = algorithm_config.get("name", "{algorithm}_{variant}_seed{seed}")
                name = name_template.format(
                    algorithm=algorithm,
                    variant=variant_name,
                    seed=seed,
                    copy=copy_index,
                    **variant,
                )
                if len(seeds) > 1 and "{seed" not in name_template and "{copy" not in name_template:
                    name = f"{name}_seed{seed}"
                if algorithm and not name.startswith(f"{algorithm}_"):
                    name = f"{algorithm}_{name}"

                runs.append(
                    {
                        "name": name,
                        "algorithm": algorithm,
                        "script": algorithm_config.get("script"),
                        "resource_type": algorithm_config.get("resource_type", config.get("resource_type", "gpu")),
                        "args": run_args,
                    }
                )
    return runs


def prepare_run(
    algorithm_run: dict[str, Any],
    root: Path,
    output_dir: Path,
    progress: bool = False,
) -> tuple[Path, list[str]]:
    run_dir = output_dir / algorithm_run["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    args = dict(algorithm_run.get("args", {}))
    args.setdefault("exp_name", algorithm_run["name"])
    if args.get("eval_frequency", 0) and not args.get("eval_results_path"):
        args["eval_results_path"] = str(run_dir / "eval_results.jsonl")
    if progress and not args.get("progress_file"):
        args["progress_file"] = str(run_dir / "progress.jsonl")
    script = resolve_script(algorithm_run, root)
    command = make_command(script, args)

    (run_dir / "config.json").write_text(json.dumps(algorithm_run, indent=2))
    (run_dir / "command.json").write_text(json.dumps(command, indent=2))
    return run_dir, command


def launch_run(
    algorithm_run: dict[str, Any],
    root: Path,
    output_dir: Path,
    cuda_device: int | None = None,
    progress: bool = False,
) -> subprocess.Popen[Any]:
    run_dir, command = prepare_run(algorithm_run, root, output_dir, progress=progress)
    stdout = open(run_dir / "stdout.log", "w")
    stderr = open(run_dir / "stderr.log", "w")
    env = os.environ.copy()
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    process = subprocess.Popen(command, cwd=root, stdout=stdout, stderr=stderr, env=env)
    process._cleanrl_av_stdout = stdout  # type: ignore[attr-defined]
    process._cleanrl_av_stderr = stderr  # type: ignore[attr-defined]
    process._cleanrl_av_name = algorithm_run["name"]  # type: ignore[attr-defined]
    process._cleanrl_av_run_dir = run_dir  # type: ignore[attr-defined]
    process._cleanrl_av_started_at = time.time()  # type: ignore[attr-defined]
    process._cleanrl_av_cuda_device = cuda_device  # type: ignore[attr-defined]
    process._cleanrl_av_progress_file = run_dir / "progress.jsonl"  # type: ignore[attr-defined]
    process._cleanrl_av_seen_progress = 0  # type: ignore[attr-defined]
    process._cleanrl_av_print_progress = progress  # type: ignore[attr-defined]
    return process


def close_process_logs(process: subprocess.Popen[Any]) -> None:
    for attr in ("_cleanrl_av_stdout", "_cleanrl_av_stderr"):
        handle = getattr(process, attr, None)
        if handle is not None:
            handle.close()


def write_run_summary(process: subprocess.Popen[Any], return_code: int) -> None:
    run_dir = getattr(process, "_cleanrl_av_run_dir")
    started_at = getattr(process, "_cleanrl_av_started_at")
    finished_at = time.time()
    summary = {
        "name": getattr(process, "_cleanrl_av_name"),
        "return_code": return_code,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": finished_at - started_at,
        "cuda_device": getattr(process, "_cleanrl_av_cuda_device", None),
    }
    (run_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2))


def read_new_progress_events(process: subprocess.Popen[Any]) -> list[dict[str, Any]]:
    progress_file = getattr(process, "_cleanrl_av_progress_file", None)
    if progress_file is None or not progress_file.exists():
        return []

    seen = getattr(process, "_cleanrl_av_seen_progress", 0)
    lines = progress_file.read_text().splitlines()
    new_lines = lines[seen:]
    process._cleanrl_av_seen_progress = len(lines)  # type: ignore[attr-defined]

    events = []
    for line in new_lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def print_progress_events(process: subprocess.Popen[Any]) -> None:
    if not getattr(process, "_cleanrl_av_print_progress", False):
        return

    name = getattr(process, "_cleanrl_av_name")
    for event in read_new_progress_events(process):
        event_type = event.get("event")
        step = event.get("global_step", 0)
        total = event.get("total_timesteps", 0)
        if event_type == "started":
            print(f"[{name}] started 0/{total}", flush=True)
        elif event_type == "eval":
            pct = 100.0 * step / total if total else 0.0
            mean_return = event.get("mean_return")
            avg_over = event.get("mean_average_overestimation")
            start_over = event.get("mean_start_overestimation")
            print(
                f"[{name}] eval step={step}/{total} ({pct:.1f}%) "
                f"return={mean_return:.3f} avg_over={avg_over:.3f} start_over={start_over:.3f}",
                flush=True,
            )
        elif event_type == "finished":
            print(f"[{name}] finished {step}/{total}", flush=True)


def is_completed_successfully(summary_path: Path) -> bool:
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return summary.get("return_code") == 0


def run_batch(
    config: dict[str, Any],
    config_path: Path,
    root: Path,
    include_patterns: list[str] | None = None,
    dry_run: bool = False,
    progress: bool = False,
) -> None:
    if "output_dir" in config:
        output_dir = Path(config["output_dir"])
    else:
        output_dir = config_path.parent
    if not output_dir.is_absolute() and "output_dir" in config:
        output_dir = config_path.parent / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_config.json").write_text(json.dumps(config, indent=2))

    algorithm_runs = expand_algorithm_runs(config)
    include_patterns = include_patterns or []
    if include_patterns:
        algorithm_runs = [
            run
            for run in algorithm_runs
            if any(fnmatch.fnmatch(run["name"], pattern) for pattern in include_patterns)
        ]
        if not algorithm_runs:
            raise ValueError(f"No algorithm runs matched include patterns: {include_patterns}")

    max_parallel = int(config.get("max_parallel", 1))
    max_parallel_cpu = int(config.get("max_parallel_cpu", max_parallel))
    gpu_devices = list(config.get("gpu_devices", []))
    max_parallel_gpu = int(config.get("max_parallel_gpu", len(gpu_devices) or max_parallel))
    if not gpu_devices and max_parallel_gpu > 0:
        gpu_devices = list(range(max_parallel_gpu))
    rerun_completed = bool(config.get("rerun_completed", False))
    progress = progress or bool(config.get("progress", False))

    pending = []
    skipped = []
    for algorithm_run in algorithm_runs:
        summary_path = output_dir / algorithm_run["name"] / "experiment_summary.json"
        if summary_path.exists() and is_completed_successfully(summary_path) and not rerun_completed:
            skipped.append(algorithm_run["name"])
            print(f"algorithm run {algorithm_run['name']} already completed; skipping", flush=True)
        else:
            pending.append(algorithm_run)

    needs_cpu = any(run.get("resource_type", "gpu") == "cpu" for run in pending)
    needs_gpu = any(run.get("resource_type", "gpu") == "gpu" for run in pending)
    if needs_cpu and max_parallel_cpu <= 0:
        raise ValueError("At least one CPU run is pending, but max_parallel_cpu is 0.")
    if needs_gpu and max_parallel_gpu <= 0:
        raise ValueError("At least one GPU run is pending, but max_parallel_gpu is 0.")

    if dry_run:
        for algorithm_run in pending:
            prepare_run(algorithm_run, root, output_dir, progress=progress)
            print(f"prepared command for {algorithm_run['name']}", flush=True)
        pending = []

    running_cpu: list[subprocess.Popen[Any]] = []
    running_gpu: list[subprocess.Popen[Any]] = []
    failures: list[tuple[str, int]] = []

    def handle_finished(processes: list[subprocess.Popen[Any]]) -> list[subprocess.Popen[Any]]:
        still_running = []
        for process in processes:
            print_progress_events(process)
            return_code = process.poll()
            if return_code is None:
                still_running.append(process)
                continue

            name = getattr(process, "_cleanrl_av_name")
            close_process_logs(process)
            write_run_summary(process, return_code)
            if return_code != 0:
                failures.append((name, return_code))
            print(f"process for algorithm run {name} exited with return_code={return_code}", flush=True)
        return still_running

    while pending or running_cpu or running_gpu:
        running_cpu = handle_finished(running_cpu)
        running_gpu = handle_finished(running_gpu)
        free_gpus = list(gpu_devices)
        for process in running_gpu:
            cuda_device = getattr(process, "_cleanrl_av_cuda_device", None)
            if cuda_device in free_gpus:
                free_gpus.remove(cuda_device)

        launched_any = True
        while pending and launched_any:
            launched_any = False
            for index, algorithm_run in list(enumerate(pending)):
                resource_type = algorithm_run.get("resource_type", "gpu")
                if resource_type == "cpu":
                    if len(running_cpu) >= max_parallel_cpu:
                        continue
                    pending.pop(index)
                    process = launch_run(algorithm_run, root, output_dir, progress=progress)
                    running_cpu.append(process)
                    print(f"algorithm run {algorithm_run['name']} launched", flush=True)
                    launched_any = True
                    break
                if resource_type == "gpu":
                    if len(running_gpu) >= max_parallel_gpu or not free_gpus:
                        continue
                    pending.pop(index)
                    cuda_device = free_gpus.pop(0)
                    process = launch_run(algorithm_run, root, output_dir, cuda_device=cuda_device, progress=progress)
                    running_gpu.append(process)
                    print(
                        f"algorithm run {algorithm_run['name']} launched on CUDA_VISIBLE_DEVICES={cuda_device}",
                        flush=True,
                    )
                    launched_any = True
                    break
                raise ValueError(f"Unknown resource_type '{resource_type}' for {algorithm_run['name']}")

        if pending or running_cpu or running_gpu:
            time.sleep(1.0)

    summary = {
        "description": config.get("description", ""),
        "environment": config.get("environment", {}),
        "evaluation": config.get("evaluation", {}),
        "include_patterns": include_patterns,
        "num_algorithm_runs": len(algorithm_runs),
        "algorithm_runs": [run["name"] for run in algorithm_runs],
        "num_skipped_completed": len(skipped),
        "skipped_completed": skipped,
        "num_failures": len(failures),
        "failures": failures,
        "dry_run": dry_run,
    }
    (output_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(f"Batch finished with failures: {failures}")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config, config_path = load_config(args.folder)
    run_batch(config, config_path, root, include_patterns=args.include, dry_run=args.dry_run, progress=args.progress)


if __name__ == "__main__":
    main()
