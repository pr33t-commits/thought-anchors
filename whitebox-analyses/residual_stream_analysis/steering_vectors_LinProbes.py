import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRESETS = {
    "local": {
        "residual_input_dir": SCRIPT_DIR / "Input",
        "chunk_input_dir": PROJECT_ROOT / "math_rollouts",
        "output_dir": SCRIPT_DIR / "Results" / "linear_probes",
        "model": "deepseek-r1-distill-Qwen-1.5b",
    },
    "vm": {
        "residual_input_dir": Path("/home/lodaya_dimpal/storage/residual_stream_analysis/Input"),
        "chunk_input_dir": Path("/home/lodaya_dimpal/storage/math_rollouts"),
        "output_dir": Path("/home/lodaya_dimpal/storage/Results/linear_probes"),
        "model": "deepseek-r1-distill-llama-8b",
    },
}

DEFAULT_RESIDUAL_FILE = "residual_stream_extracts.json"


Example = Tuple[str, str, int, str, np.ndarray]


def parse_problem_ids(problem_ids: Optional[str]) -> Optional[List[str]]:
    if not problem_ids:
        return None
    return [problem.strip().removeprefix("problem_") for problem in problem_ids.split(",") if problem.strip()]

def build_residual_path(args: argparse.Namespace) -> Path:
    if args.residual_path:
        return args.residual_path
    residual_input_dir = args.residual_input_dir or PRESETS[args.preset]["residual_input_dir"]
    return Path(residual_input_dir) / args.residual_file


def build_labels_root(args: argparse.Namespace) -> Path:
    if args.input_dir:
        return args.input_dir
    if args.labels_root:
        return args.labels_root

    chunk_input_dir = args.chunk_input_dir or PRESETS[args.preset]["chunk_input_dir"]
    model = args.model or PRESETS[args.preset]["model"]
    labels_root = (
        Path(chunk_input_dir)
        / model.split(":")[0]
        / f"temperature_{args.temperature}_top_p_{args.top_p}"
    )

    if args.rollout_type == "forced_answer":
        dir_name = f"{args.base_solution_type}_base_solution_forced_answer"
    else:
        dir_name = f"{args.base_solution_type}_base_solution"

    if args.input_exp_suffix:
        dir_name = f"{dir_name}_{args.input_exp_suffix}"

    return labels_root / dir_name


def build_output_dir(args: argparse.Namespace) -> Path:
    return args.output_dir or PRESETS[args.preset]["output_dir"]


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_problem_id(problem_id: Any) -> str:
    if "problem_" in str(problem_id):
        return str(problem_id).removeprefix("problem_")
    return str(problem_id)

def normalize_chunk_id(chunk_id: Any) -> int:
    if isinstance(chunk_id, int):
        return chunk_id
    return int(str(chunk_id).removeprefix("chunk_"))


def normalize_layer(layer: Any) -> str:
    layer_text = str(layer)
    return layer_text.removeprefix("layer_")


def iter_residual_records(residual_data: Any) -> Iterable[Tuple[str, str, int, str, Any]]:
    """
    Yield (exp_id, problem_id, chunk_id, layer, vector) from the expected format:
    [{exp_id: {problem_id: {chunk_id: {layer: residual_stream_vector}}}}, ...]

    A single top-level dict is also accepted for convenience.
    """
    entries = residual_data if isinstance(residual_data, list) else [residual_data]

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for exp_id, problems in entry.items():
            if not isinstance(problems, dict):
                continue
            for problem_id, chunks in problems.items():
                if not isinstance(chunks, dict):
                    continue
                normalized_problem = normalize_problem_id(problem_id)
                for chunk_id, layers in chunks.items():
                    if not isinstance(layers, dict):
                        continue
                    normalized_chunk = normalize_chunk_id(chunk_id)
                    for layer, vector in layers.items():
                        yield (
                            str(exp_id),
                            normalized_problem,
                            normalized_chunk,
                            normalize_layer(layer),
                            vector,
                        )


def vector_to_1d(vector: Any, pool: str) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim == 0:
        raise ValueError("Residual stream vector must have at least one dimension.")
    if arr.ndim > 1:
        if pool == "mean":
            arr = arr.mean(axis=tuple(range(arr.ndim - 1)))
        elif pool == "flatten":
            arr = arr.reshape(-1)
        else:
            raise ValueError(f"Unsupported vector pooling strategy: {pool}")
    if not np.isfinite(arr).all():
        raise ValueError("Residual stream vector contains NaN or infinite values.")
    return arr


def load_problem_labels(labels_root: Path, problem_id: str) -> Dict[int, List[str]]:
    labels_path = labels_root / f"problem_{problem_id}" / "chunks_labeled.json"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    labels = {}
    for chunk in load_json(labels_path):
        chunk_id = normalize_chunk_id(chunk["chunk_idx"])
        tags = chunk.get("function_tags", [])
        if isinstance(tags, str):
            tags = [tags]
        labels[chunk_id] = [str(tag) for tag in tags if str(tag)]
    return labels


def build_examples(
    residual_path: Path,
    labels_root: Path,
    tag_mode: str,
    vector_pool: str,
    problem_ids: Optional[List[str]] = None,
) -> Tuple[Dict[Tuple[str, str], List[Example]], Dict[str, Any]]:
    residual_data = load_json(residual_path)
    labels_cache: Dict[str, Dict[int, List[str]]] = {}
    examples_by_probe: Dict[Tuple[str, str], List[Example]] = defaultdict(list)
    skipped = Counter()
    allowed_problems = set(problem_ids) if problem_ids is not None else None

    expected_dim: Dict[Tuple[str, str], int] = {}
    for exp_id, problem_id, chunk_id, layer, vector in iter_residual_records(residual_data):
        if allowed_problems is not None and problem_id not in allowed_problems:
            skipped["filtered_problem"] += 1
            continue

        if problem_id not in labels_cache:
            try:
                labels_cache[problem_id] = load_problem_labels(labels_root, problem_id)
            except FileNotFoundError:
                skipped["missing_problem_labels"] += 1
                continue

        tags = labels_cache[problem_id].get(chunk_id, [])
        if not tags:
            skipped["missing_chunk_labels"] += 1
            continue

        try:
            arr = vector_to_1d(vector, vector_pool)
        except ValueError:
            skipped["invalid_vector"] += 1
            continue

        probe_key = (exp_id, layer)
        if probe_key in expected_dim and arr.shape[0] != expected_dim[probe_key]:
            skipped["inconsistent_vector_dim"] += 1
            continue
        expected_dim.setdefault(probe_key, arr.shape[0])

        selected_tags = tags if tag_mode == "explode" else tags[:1]
        for tag in selected_tags:
            examples_by_probe[probe_key].append((exp_id, problem_id, chunk_id, tag, arr))

    metadata = {
        "residual_path": str(residual_path),
        "labels_root": str(labels_root),
        "tag_mode": tag_mode,
        "vector_pool": vector_pool,
        "problem_ids": problem_ids,
        "skipped": dict(skipped),
        "num_probes": len(examples_by_probe),
        "problems_loaded": sorted(labels_cache.keys(), key=lambda item: int(item) if item.isdigit() else item),
    }
    return examples_by_probe, metadata

def make_probe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        steps=[
            ("standardize", StandardScaler()),
            (
                "probe",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                    penalty="l2",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def evaluate_probe(
    probe_key: Tuple[str, str],
    examples: List[Example],
    requested_splits: int,
) -> Dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    from sklearn.model_selection import GroupKFold

    exp_id, layer = probe_key
    X = np.vstack([example[4] for example in examples])
    y = np.asarray([example[3] for example in examples])
    groups = np.asarray([example[1] for example in examples])

    classes = sorted(set(y))
    unique_groups = sorted(set(groups))
    result: Dict[str, Any] = {
        "exp_id": exp_id,
        "layer": layer,
        "num_examples": int(len(examples)),
        "num_features": int(X.shape[1]),
        "num_classes": int(len(classes)),
        "classes": classes,
        "class_counts": dict(Counter(y)),
        "num_groups": int(len(unique_groups)),
        "groups": unique_groups,
    }

    if len(classes) < 2:
        result["status"] = "skipped"
        result["reason"] = "Need at least two function_tag classes."
        return result
    if len(unique_groups) < 2:
        result["status"] = "skipped"
        result["reason"] = "Need at least two problem groups for grouped cross-validation."
        return result

    n_splits = min(requested_splits, len(unique_groups))
    splitter = GroupKFold(n_splits=n_splits)
    y_true_all: List[str] = []
    y_pred_all: List[str] = []
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        train_classes = set(y[train_idx])
        test_classes = set(y[test_idx])
        if len(train_classes) < 2:
            fold_results.append(
                {
                    "fold": fold_idx,
                    "status": "skipped",
                    "reason": "Training split has fewer than two classes.",
                    "test_groups": sorted(set(groups[test_idx])),
                }
            )
            continue

        model = make_probe()
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])

        y_true_all.extend(y[test_idx].tolist())
        y_pred_all.extend(y_pred.tolist())
        fold_results.append(
            {
                "fold": fold_idx,
                "status": "ok",
                "train_examples": int(len(train_idx)),
                "test_examples": int(len(test_idx)),
                "train_groups": sorted(set(groups[train_idx])),
                "test_groups": sorted(set(groups[test_idx])),
                "train_classes": sorted(train_classes),
                "test_classes": sorted(test_classes),
                "accuracy": float(accuracy_score(y[test_idx], y_pred)),
                "f1_macro": float(f1_score(y[test_idx], y_pred, average="macro", zero_division=0)),
                "f1_weighted": float(
                    f1_score(y[test_idx], y_pred, average="weighted", zero_division=0)
                ),
            }
        )

    if not y_true_all:
        result["status"] = "skipped"
        result["reason"] = "No fold had at least two classes in the training data."
        result["folds"] = fold_results
        return result

    labels = sorted(set(y_true_all).union(y_pred_all))
    result.update(
        {
            "status": "ok",
            "n_splits": int(n_splits),
            "folds": fold_results,
            "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
            "f1_macro": float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)),
            "f1_weighted": float(
                f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
            ),
            "confusion_matrix_labels": labels,
            "confusion_matrix": confusion_matrix(y_true_all, y_pred_all, labels=labels).tolist(),
            "classification_report": classification_report(
                y_true_all,
                y_pred_all,
                labels=labels,
                output_dict=True,
                zero_division=0,
            ),
        }
    )
    return result


def write_summary_csv(results: List[Dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "exp_id",
        "layer",
        "status",
        "num_examples",
        "num_features",
        "num_classes",
        "num_groups",
        "n_splits",
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "reason",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in fieldnames})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train multinomial logistic-regression linear probes over residual stream "
            "vectors to classify chunk function_tags."
        )
    )
    parser.add_argument("--preset", choices=PRESETS.keys(), default="local", help="Use local or VM path defaults.")
    parser.add_argument(
        "--residual-path",
        type=Path,
        default=None,
        help="Full path to residual stream JSON. Overrides --residual-input-dir/--residual-file.",
    )
    parser.add_argument(
        "--residual-input-dir",
        type=Path,
        default=None,
        help="Directory containing the residual stream JSON.",
    )
    parser.add_argument(
        "--residual-file",
        type=str,
        default=DEFAULT_RESIDUAL_FILE,
        help="Residual stream JSON filename inside --residual-input-dir.",
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=None,
        help="Full directory containing problem_*/chunks_labeled.json. Overrides rollout path args.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help="Full resolved chunk-label input directory. Alias-style override matching deep_thinking_tokens.py.",
    )
    parser.add_argument("-cid", "--chunk_input_dir", type=Path, default=None, help="Root directory to read chunk labels from.")
    parser.add_argument("-m", "--model", type=str, default=None, help="Rollout model/deployment path in the data directory.")
    parser.add_argument("-b", "--base_solution_type", type=str, default="correct", choices=["correct", "incorrect"])
    parser.add_argument("-r", "--rollout_type", type=str, default="default", choices=["default", "forced_answer"])
    parser.add_argument("-t", "--temperature", type=float, default=0.6)
    parser.add_argument("-tp", "--top_p", type=float, default=0.95)
    parser.add_argument("-is", "--input_exp_suffix", type=str, default=None)
    parser.add_argument("-sp", "--specific_problems", type=str, default=None, help="Comma-separated problem IDs to include.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for linear probe results. Defaults come from --preset.",
    )
    parser.add_argument("--splits", type=int, default=5, help="Maximum GroupKFold split count.")
    parser.add_argument(
        "--tag-mode",
        choices=["primary", "explode"],
        default="primary",
        help="Use first function_tag per chunk, or duplicate examples across all tags.",
    )
    parser.add_argument(
        "--vector-pool",
        choices=["mean", "flatten"],
        default="mean",
        help="How to convert non-1D residual arrays into one feature vector.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.splits < 2:
        raise ValueError("--splits must be at least 2.")

    residual_path = build_residual_path(args)
    labels_root = build_labels_root(args)
    output_dir = build_output_dir(args)
    specific_problems = parse_problem_ids(args.specific_problems)

    examples_by_probe, metadata = build_examples(
        residual_path=residual_path,
        labels_root=labels_root,
        tag_mode=args.tag_mode,
        vector_pool=args.vector_pool,
        problem_ids=specific_problems,
    )

    results = []
    for probe_key in sorted(
        examples_by_probe,
        key=lambda item: (item[0], int(item[1]) if item[1].isdigit() else item[1]),
    ):
        results.append(evaluate_probe(probe_key, examples_by_probe[probe_key], args.splits))

    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = output_dir / "linear_probe_results.json"
    summary_path = output_dir / "linear_probe_summary.csv"

    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "results": results}, f, indent=2)
    write_summary_csv(results, summary_path)

    ok_count = sum(result["status"] == "ok" for result in results)
    skipped_count = len(results) - ok_count
    print(f"Evaluated {ok_count} probes; skipped {skipped_count}.")
    print(f"Saved detailed results to {detailed_path}")
    print(f"Saved summary CSV to {summary_path}")


if __name__ == "__main__":
    main()
