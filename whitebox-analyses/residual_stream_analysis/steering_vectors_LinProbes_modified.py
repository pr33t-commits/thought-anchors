import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRESETS = {
    "local": {
        "residual_input_dir": SCRIPT_DIR / "Inputs",
        "chunk_input_dir": PROJECT_ROOT / "math_rollouts",
        "output_dir": SCRIPT_DIR / "Results" / "linear_probes",
        "model": "deepseek-r1-distill-qwen-14b",
    },
    "vm": {
        "residual_input_dir": SCRIPT_DIR / "Inputs",
        "chunk_input_dir": Path("../workspace/math_rollouts"),
        "output_dir": SCRIPT_DIR / "Results/linear_probes",
        "model": "deepseek-r1-distill-qwen-14b",
    },
}

DEFAULT_RESIDUAL_FILE = "residual_stream_extracts.json"
DEFAULT_VECTOR_POOLS = ["mean_vector", "last_tokenvector"]
SAMPLE_WEIGHT_FIELDS = (
    "counterfactual_importance_kl",
    "conterfactual_importance_kl",
)

Example = Tuple[str, str, int, str, np.ndarray, float]


def log_problem_timing(problem_id: str, step: str, elapsed: float) -> None:
    print(f"problem_{problem_id} timing: {step}={elapsed:.2f}s", flush=True)


def log_probe_timing(exp_id: str, layer: str, step: str, elapsed: float) -> None:
    print(f"probe {exp_id}/layer_{layer} timing: {step}={elapsed:.2f}s", flush=True)


def parse_problem_ids(problem_ids: Optional[str]) -> Optional[List[str]]:
    if not problem_ids:
        return None
    return [problem.strip().removeprefix("problem_") for problem in problem_ids.split(",") if problem.strip()]


def parse_vector_pools(value: Optional[str]) -> List[str]:
    if not value:
        return list(DEFAULT_VECTOR_POOLS)
    pools = [item.strip() for item in value.split(",") if item.strip()]
    if not pools:
        return list(DEFAULT_VECTOR_POOLS)
    return pools


def build_residual_paths(
    args: argparse.Namespace,
    problem_ids: Optional[List[str]] = None,
) -> List[Path]:
    if args.residual_path:
        path = Path(args.residual_path)
        if path.is_dir():
            return discover_residual_paths(path, args.residual_file, problem_ids)
        return [path]
    residual_input_dir = args.residual_input_dir or PRESETS[args.preset]["residual_input_dir"]
    return discover_residual_paths(Path(residual_input_dir), args.residual_file, problem_ids)


def discover_residual_paths(
    residual_input_dir: Path,
    residual_file: str,
    problem_ids: Optional[List[str]] = None,
) -> List[Path]:
    if problem_ids is not None:
        return [
            residual_input_dir / f"problem_{problem_id}" / residual_file
            for problem_id in problem_ids
        ]

    problem_paths = sorted(residual_input_dir.glob(f"problem_*/{residual_file}"))
    if problem_paths:
        return problem_paths

    legacy_path = residual_input_dir / residual_file
    return [legacy_path]


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


def problem_id_from_residual_path(path: Path) -> Optional[str]:
    parent_name = path.parent.name
    if parent_name.startswith("problem_"):
        return parent_name.removeprefix("problem_")
    return None


def load_residual_inputs(paths: Sequence[Path]) -> List[Tuple[Path, Any]]:
    missing_paths = [path for path in paths if not path.is_file()]
    if missing_paths:
        missing = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing residual stream JSON file(s):\n{missing}")

    residual_inputs = []
    for path in paths:
        step_start = time.perf_counter()
        residual_data = load_json(path)
        elapsed = time.perf_counter() - step_start
        problem_id = problem_id_from_residual_path(path)
        if problem_id is not None:
            log_problem_timing(problem_id, "load_residual_json", elapsed)
        else:
            print(f"residual input timing: load_residual_json {path}={elapsed:.2f}s", flush=True)
        residual_inputs.append((path, residual_data))
    return residual_inputs


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
        if pool == "mean_vector":
            arr = arr.mean(axis=tuple(range(arr.ndim - 1)))
        elif pool == "last_tokenvector":
            arr = arr.reshape(-1, arr.shape[-1])[-1]
        else:
            raise ValueError(f"Unsupported vector pooling strategy: {pool}")
    if not np.isfinite(arr).all():
        raise ValueError("Residual stream vector contains NaN or infinite values.")
    return arr


def _normalize_sample_weight(chunk: Dict[str, Any]) -> float:
    for field in SAMPLE_WEIGHT_FIELDS:
        if field in chunk:
            value = chunk.get(field, 0.0)
            try:
                weight = float(value)
            except (TypeError, ValueError):
                return 0.0
            return max(weight, 0.0)
    return 0.0


def load_problem_labels(labels_root: Path, problem_id: str) -> Dict[int, Dict[str, Any]]:
    labels_path = labels_root / f"problem_{problem_id}" / "chunks_labeled.json"
    if not labels_path.is_file():
        print(f"labels file not found for problem_{problem_id} at expected path: {labels_path}", flush=True)
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    chunks = load_json(labels_path)
    if chunks:
        print(chunks[0], flush=True)

    labels: Dict[int, Dict[str, Any]] = {}
    for chunk in chunks:
        chunk_id = normalize_chunk_id(chunk["chunk_idx"])
        tags = chunk.get("function_tags", [])
        if isinstance(tags, str):
            tags = [tags]
        labels[chunk_id] = {
            "tags": [str(tag) for tag in tags if str(tag)],
            "sample_weight": _normalize_sample_weight(chunk),
        }
    return labels


def build_examples(
    residual_paths: Sequence[Path],
    labels_root: Path,
    tag_mode: str,
    vector_pool: str,
    problem_ids: Optional[List[str]] = None,
) -> Tuple[Dict[Tuple[str, str], List[Example]], Dict[str, Any]]:
    residual_inputs = load_residual_inputs(residual_paths)
    labels_cache: Dict[str, Dict[int, Dict[str, Any]]] = {}
    examples_by_probe: Dict[Tuple[str, str], List[Example]] = defaultdict(list)
    skipped = Counter()
    allowed_problems = set(problem_ids) if problem_ids is not None else None
    problem_timers: Dict[str, Counter] = defaultdict(Counter)
    problem_counts: Dict[str, Counter] = defaultdict(Counter)

    expected_dim: Dict[Tuple[str, str], int] = {}

    for residual_path, residual_data in residual_inputs:
        residual_file_problem_id = problem_id_from_residual_path(residual_path)
        step_start = time.perf_counter()
        for exp_id, problem_id, chunk_id, layer, vector in iter_residual_records(residual_data):
            if residual_file_problem_id is not None and problem_id != residual_file_problem_id:
                print(
                    f"Warning: {residual_path} contains problem_{problem_id}, "
                    f"expected problem_{residual_file_problem_id}",
                    flush=True,
                )
            problem_counts[problem_id]["residual_records"] += 1
            if allowed_problems is not None and problem_id not in allowed_problems:
                skipped["filtered_problem"] += 1
                problem_counts[problem_id]["filtered_records"] += 1
                continue

            if problem_id not in labels_cache:
                label_start = time.perf_counter()
                try:
                    labels_cache[problem_id] = load_problem_labels(labels_root, problem_id)
                except FileNotFoundError:
                    skipped["missing_problem_labels"] += 1
                    problem_counts[problem_id]["missing_problem_labels"] += 1
                    continue
                label_elapsed = time.perf_counter() - label_start
                problem_timers[problem_id]["load_problem_labels"] += label_elapsed
                log_problem_timing(problem_id, "load_problem_labels", label_elapsed)

            lookup_start = time.perf_counter()
            label_info = labels_cache[problem_id].get(chunk_id, {})
            problem_timers[problem_id]["label_lookup"] += time.perf_counter() - lookup_start
            tags = label_info.get("tags", [])
            sample_weight = float(label_info.get("sample_weight", 0.0))

            if not tags:
                print(f"Warning: No labels found for problem_{problem_id} chunk_{chunk_id}", flush=True)
                skipped["missing_chunk_labels"] += 1
                problem_counts[problem_id]["missing_chunk_labels"] += 1
                continue

            vector_start = time.perf_counter()
            try:
                arr = vector_to_1d(vector, vector_pool)
            except ValueError:
                print(f"Warning: Invalid vector for problem_{problem_id} chunk_{chunk_id}", flush=True)
                skipped["invalid_vector"] += 1
                problem_counts[problem_id]["invalid_vector"] += 1
                continue
            problem_timers[problem_id]["vector_to_1d"] += time.perf_counter() - vector_start

            dim_start = time.perf_counter()
            probe_key = (exp_id, layer)
            if probe_key in expected_dim and arr.shape[0] != expected_dim[probe_key]:
                skipped["inconsistent_vector_dim"] += 1
                problem_counts[problem_id]["inconsistent_vector_dim"] += 1
                continue
            expected_dim.setdefault(probe_key, arr.shape[0])
            problem_timers[problem_id]["dimension_check"] += time.perf_counter() - dim_start

            append_start = time.perf_counter()
            selected_tags = tags if tag_mode == "explode" else tags[:1]
            for tag in selected_tags:
                examples_by_probe[probe_key].append((exp_id, problem_id, chunk_id, tag, arr, sample_weight))
                problem_counts[problem_id]["examples"] += 1
            problem_timers[problem_id]["append_examples"] += time.perf_counter() - append_start
        file_elapsed = time.perf_counter() - step_start
        if residual_file_problem_id is not None:
            log_problem_timing(residual_file_problem_id, "process_residual_records", file_elapsed)
        else:
            print(f"residual input timing: process_residual_records {residual_path}={file_elapsed:.2f}s", flush=True)
    for problem_id in sorted(problem_counts, key=lambda item: int(item) if item.isdigit() else item):
        for step, elapsed in problem_timers[problem_id].items():
            if step == "load_problem_labels":
                continue
            log_problem_timing(problem_id, step, elapsed)
        print(
            f"problem_{problem_id} timing: "
            f"residual_records={problem_counts[problem_id]['residual_records']} "
            f"examples={problem_counts[problem_id]['examples']} "
            f"filtered_records={problem_counts[problem_id]['filtered_records']} "
            f"missing_problem_labels={problem_counts[problem_id]['missing_problem_labels']} "
            f"missing_chunk_labels={problem_counts[problem_id]['missing_chunk_labels']} "
            f"invalid_vector={problem_counts[problem_id]['invalid_vector']} "
            f"inconsistent_vector_dim={problem_counts[problem_id]['inconsistent_vector_dim']}",
            flush=True,
        )

    metadata = {
        "residual_paths": [str(path) for path in residual_paths],
        "labels_root": str(labels_root),
        "tag_mode": tag_mode,
        "vector_pool": vector_pool,
        "problem_ids": problem_ids,
        "skipped": dict(skipped),
        "num_probes": len(examples_by_probe),
        "problems_loaded": sorted(labels_cache.keys(), key=lambda item: int(item) if item.isdigit() else item),
        "sample_weight_fields": list(SAMPLE_WEIGHT_FIELDS),
    }
    return examples_by_probe, metadata


def make_probe(probe_type: str):
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    base_lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        penalty="l2",
        solver="lbfgs",
    )

    if probe_type == "multinomial":
        classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            # multi_class="multinomial",
            penalty="l2",
            solver="lbfgs",
        )
    elif probe_type == "one_vs_rest":
        classifier = OneVsRestClassifier(base_lr)
    else:
        raise ValueError(f"Unsupported probe_type: {probe_type}")

    return Pipeline(
        steps=[
            ("standardize", StandardScaler()),
            ("probe", classifier),
        ]
    )


def extract_class_coefficients(model: Any, probe_type: str) -> Dict[str, Dict[str, Any]]:
    probe = model.named_steps["probe"]

    if probe_type == "multinomial":
        classes = [str(label) for label in probe.classes_]
        coef = np.asarray(probe.coef_, dtype=float)
        intercept = np.asarray(probe.intercept_, dtype=float)
        return {
            class_label: {
                "coef": coef[idx].tolist(),
                "intercept": float(intercept[idx]) if intercept.ndim > 0 else float(intercept),
            }
            for idx, class_label in enumerate(classes)
        }

    classes = [str(label) for label in probe.classes_]
    coefficients: Dict[str, Dict[str, Any]] = {}
    for class_label, estimator in zip(classes, probe.estimators_):
        coef = np.asarray(estimator.coef_, dtype=float).reshape(-1)
        intercept = np.asarray(estimator.intercept_, dtype=float).reshape(-1)
        coefficients[class_label] = {
            "coef": coef.tolist(),
            "intercept": float(intercept[0]) if intercept.size else 0.0,
        }
    return coefficients


def evaluate_probe(
    probe_key: Tuple[str, str],
    examples: List[Example],
    requested_splits: int,
    probe_type: str,
) -> Dict[str, Any]:
    eval_start = time.perf_counter()
    import_start = time.perf_counter()
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        recall_score,
    )
    from sklearn.model_selection import GroupKFold
    import_elapsed = time.perf_counter() - import_start

    exp_id, layer = probe_key

    step_start = time.perf_counter()
    X = np.vstack([example[4] for example in examples])
    y = np.asarray([example[3] for example in examples])
    groups = np.asarray([example[1] for example in examples])
    sample_weights = np.asarray([example[5] for example in examples], dtype=float)
    log_probe_timing(exp_id, layer, "build_arrays", time.perf_counter() - step_start)

    print(X.shape, y.shape, groups.shape, sample_weights.shape)
    print(X.mean(), y[:5])

    step_start = time.perf_counter()
    classes = sorted(set(y))
    unique_groups = sorted(set(groups))
    result: Dict[str, Any] = {
        "exp_id": exp_id,
        "layer": layer,
        "probe_type": probe_type,
        "num_examples": int(len(examples)),
        "num_features": int(X.shape[1]),
        "num_classes": int(len(classes)),
        "classes": classes,
        "class_counts": dict(Counter(y)),
        "num_groups": int(len(unique_groups)),
        "groups": unique_groups,
    }
    log_probe_timing(exp_id, layer, "summarize_groups_classes", time.perf_counter() - step_start)

    if len(classes) < 2:
        result["status"] = "skipped"
        result["reason"] = "Need at least two function_tag classes."
        log_probe_timing(exp_id, layer, "evaluate_probe_total", time.perf_counter() - eval_start)
        return result
    if len(unique_groups) < 2:
        result["status"] = "skipped"
        result["reason"] = "Need at least two problem groups for grouped cross-validation."
        log_probe_timing(exp_id, layer, "evaluate_probe_total", time.perf_counter() - eval_start)
        return result

    step_start = time.perf_counter()
    n_splits = min(requested_splits, len(unique_groups))
    splitter = GroupKFold(n_splits=n_splits)
    y_true_all: List[str] = []
    y_pred_all: List[str] = []
    weights_all: List[float] = []
    fold_results = []
    log_probe_timing(exp_id, layer, "setup_group_kfold", time.perf_counter() - step_start)

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        fold_start = time.perf_counter()
        step_start = time.perf_counter()
        train_classes = set(y[train_idx])
        test_classes = set(y[test_idx])
        log_probe_timing(exp_id, layer, f"fold_{fold_idx}_class_check", time.perf_counter() - step_start)
        if len(train_classes) < 2:
            fold_results.append(
                {
                    "fold": fold_idx,
                    "status": "skipped",
                    "reason": "Training split has fewer than two classes.",
                    "test_groups": sorted(set(groups[test_idx])),
                }
            )
            log_probe_timing(exp_id, layer, f"fold_{fold_idx}_total", time.perf_counter() - fold_start)
            continue

        step_start = time.perf_counter()
        model = make_probe(probe_type)
        log_probe_timing(exp_id, layer, f"fold_{fold_idx}_make_probe", time.perf_counter() - step_start)
        step_start = time.perf_counter()
        model.fit(X[train_idx], y[train_idx])
        log_probe_timing(exp_id, layer, f"fold_{fold_idx}_fit", time.perf_counter() - step_start)
        step_start = time.perf_counter()
        y_pred = model.predict(X[test_idx])
        log_probe_timing(exp_id, layer, f"fold_{fold_idx}_predict", time.perf_counter() - step_start)

        fold_sample_weights = sample_weights[test_idx]
        fold_report = classification_report(
            y[test_idx],
            y_pred,
            labels=classes,
            output_dict=True,
            zero_division=0,
            sample_weight=fold_sample_weights,
        )
        fold_coefficients = extract_class_coefficients(model, probe_type)

        step_start = time.perf_counter()
        y_true_all.extend(y[test_idx].tolist())
        y_pred_all.extend(y_pred.tolist())
        weights_all.extend(fold_sample_weights.tolist())
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
                "accuracy": float(accuracy_score(y[test_idx], y_pred, sample_weight=fold_sample_weights)),
                "f1_macro": float(f1_score(y[test_idx], y_pred, average="macro", zero_division=0, sample_weight=fold_sample_weights)),
                "f1_weighted": float(f1_score(y[test_idx], y_pred, average="weighted", zero_division=0, sample_weight=fold_sample_weights)),
                "recall_macro": float(recall_score(y[test_idx], y_pred, average="macro", zero_division=0, sample_weight=fold_sample_weights)),
                "recall_weighted": float(recall_score(y[test_idx], y_pred, average="weighted", zero_division=0, sample_weight=fold_sample_weights)),
                "classification_report": fold_report,
                "class_coefficients": fold_coefficients,
            }
        )
        log_probe_timing(exp_id, layer, f"fold_{fold_idx}_metrics", time.perf_counter() - step_start)
        log_probe_timing(exp_id, layer, f"fold_{fold_idx}_total", time.perf_counter() - fold_start)

    if not y_true_all:
        result["status"] = "skipped"
        result["reason"] = "No fold had at least two classes in the training data."
        result["folds"] = fold_results
        log_probe_timing(exp_id, layer, "evaluate_probe_total", time.perf_counter() - eval_start)
        return result

    step_start = time.perf_counter()
    labels = sorted(set(y_true_all).union(y_pred_all))
    aggregate_report = classification_report(
        y_true_all,
        y_pred_all,
        labels=labels,
        output_dict=True,
        zero_division=0,
        sample_weight=np.asarray(weights_all, dtype=float),
    )
    result.update(
        {
            "status": "ok",
            "n_splits": int(n_splits),
            "folds": fold_results,
            "accuracy": float(accuracy_score(y_true_all, y_pred_all, sample_weight=np.asarray(weights_all, dtype=float))),
            "f1_macro": float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0, sample_weight=np.asarray(weights_all, dtype=float))),
            "f1_weighted": float(f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0, sample_weight=np.asarray(weights_all, dtype=float))),
            "recall_macro": float(recall_score(y_true_all, y_pred_all, average="macro", zero_division=0, sample_weight=np.asarray(weights_all, dtype=float))),
            "recall_weighted": float(recall_score(y_true_all, y_pred_all, average="weighted", zero_division=0, sample_weight=np.asarray(weights_all, dtype=float))),
            "confusion_matrix_labels": labels,
            "confusion_matrix": confusion_matrix(
                y_true_all,
                y_pred_all,
                labels=labels,
                sample_weight=np.asarray(weights_all, dtype=float),
            ).tolist(),
            "classification_report": aggregate_report,
        }
    )
    log_probe_timing(exp_id, layer, "aggregate_final_metrics", time.perf_counter() - step_start)
    log_probe_timing(exp_id, layer, "evaluate_probe_total", time.perf_counter() - eval_start)
    return result


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_selection_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_results = [result for result in results if result.get("status") == "ok"]
    summary: Dict[str, Any] = {
        "overall": {},
        "per_class": {},
    }

    if not ok_results:
        return summary

    probe_types = sorted({result["probe_type"] for result in ok_results})
    class_names = sorted({cls for result in ok_results for cls in result.get("classes", [])})

    def collect_metric(metric_name: str) -> Dict[str, float]:
        values_by_type = {}
        for probe_type in probe_types:
            probe_results = [result for result in ok_results if result["probe_type"] == probe_type]
            values = [_safe_float(result.get(metric_name, 0.0)) for result in probe_results]
            values_by_type[probe_type] = float(np.mean(values)) if values else 0.0
        return values_by_type

    overall_f1 = collect_metric("f1_macro")
    overall_recall = collect_metric("recall_macro")

    summary["overall"]["best_f1"] = {
        "metric": "f1_macro",
        "scores_by_probe_type": overall_f1,
        "selected_probe_type": max(overall_f1, key=overall_f1.get) if overall_f1 else None,
    }
    summary["overall"]["best_recall"] = {
        "metric": "recall_macro",
        "scores_by_probe_type": overall_recall,
        "selected_probe_type": max(overall_recall, key=overall_recall.get) if overall_recall else None,
    }

    for class_name in class_names:
        class_f1_scores: Dict[str, float] = {}
        class_recall_scores: Dict[str, float] = {}
        for probe_type in probe_types:
            probe_results = [result for result in ok_results if result["probe_type"] == probe_type]
            f1_values = []
            recall_values = []
            for result in probe_results:
                report = result.get("classification_report", {})
                class_report = report.get(class_name)
                if isinstance(class_report, dict):
                    f1_values.append(_safe_float(class_report.get("f1-score", 0.0)))
                    recall_values.append(_safe_float(class_report.get("recall", 0.0)))
            class_f1_scores[probe_type] = float(np.mean(f1_values)) if f1_values else 0.0
            class_recall_scores[probe_type] = float(np.mean(recall_values)) if recall_values else 0.0

        summary["per_class"][class_name] = {
            "best_f1": {
                "metric": "f1-score",
                "scores_by_probe_type": class_f1_scores,
                "selected_probe_type": max(class_f1_scores, key=class_f1_scores.get) if class_f1_scores else None,
            },
            "best_recall": {
                "metric": "recall",
                "scores_by_probe_type": class_recall_scores,
                "selected_probe_type": max(class_recall_scores, key=class_recall_scores.get) if class_recall_scores else None,
            },
        }

    return summary


def average_fold_coefficients(result: Dict[str, Any]) -> Dict[str, List[float]]:
    coefficients_by_class: Dict[str, List[np.ndarray]] = defaultdict(list)

    for fold in result.get("folds", []):
        if fold.get("status") != "ok":
            continue
        for class_name, coefficient_info in fold.get("class_coefficients", {}).items():
            coef = coefficient_info.get("coef")
            if coef is None:
                continue
            coefficients_by_class[str(class_name)].append(np.asarray(coef, dtype=float))

    return {
        class_name: np.mean(class_coefficients, axis=0).tolist()
        for class_name, class_coefficients in coefficients_by_class.items()
        if class_coefficients
    }


def _layer_sort_key(layer: Any) -> Tuple[int, Any]:
    layer_text = str(layer)
    return (0, int(layer_text)) if layer_text.isdigit() else (1, layer_text)


def _merge_weight_exports(
    exports: Dict[Tuple[str, str, str], Dict[str, Any]],
    *,
    selection_strategy: str,
    probe_type: str,
    layer: str,
    class_weights: Dict[str, List[float]],
) -> None:
    key = (selection_strategy, probe_type, str(layer))
    export = exports.setdefault(
        key,
        {
            "regressor_type": probe_type,
            "layer": str(layer),
            "selection_strategy": selection_strategy,
            "classes": [],
            "weights_by_class": {},
        },
    )

    weights_by_class = export["weights_by_class"]
    for class_name, weights in class_weights.items():
        existing = weights_by_class.setdefault(class_name, [])
        existing.append(np.asarray(weights, dtype=float))


def _finalize_weight_exports(exports: Dict[Tuple[str, str, str], Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    finalized_by_strategy: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for export in exports.values():
        averaged_weights = {
            class_name: np.mean(weights, axis=0).tolist()
            for class_name, weights in export["weights_by_class"].items()
            if weights
        }
        finalized_by_strategy[export["selection_strategy"]].append(
            {
                "regressor_type": export["regressor_type"],
                "layer": export["layer"],
                "selection_strategy": export["selection_strategy"],
                "classes": sorted(averaged_weights),
                "weights_by_class": averaged_weights,
            }
        )

    return {
        selection_strategy: sorted(
            strategy_exports,
            key=lambda item: (item["regressor_type"], _layer_sort_key(item["layer"])),
        )
        for selection_strategy, strategy_exports in sorted(finalized_by_strategy.items())
    }


def count_weight_export_records(weight_exports: Dict[str, List[Dict[str, Any]]]) -> int:
    return sum(len(strategy_exports) for strategy_exports in weight_exports.values())


def build_weight_export_results(
    results: List[Dict[str, Any]],
    selection_summary: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    ok_results = [result for result in results if result.get("status") == "ok"]
    averaged_by_result = [
        (result, average_fold_coefficients(result))
        for result in ok_results
    ]
    exports: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    overall_strategies = {
        "best_overall_f1": ("overall", "best_f1"),
        "best_overall_recall": ("overall", "best_recall"),
    }
    for export_strategy, (_, summary_key) in overall_strategies.items():
        selected_probe_type = (
            selection_summary.get("overall", {})
            .get(summary_key, {})
            .get("selected_probe_type")
        )
        if not selected_probe_type:
            continue
        for result, class_weights in averaged_by_result:
            if result.get("probe_type") != selected_probe_type:
                continue
            _merge_weight_exports(
                exports,
                selection_strategy=export_strategy,
                probe_type=selected_probe_type,
                layer=str(result["layer"]),
                class_weights=class_weights,
            )

    per_class_strategy_keys = {
        "best_class_f1": "best_f1",
        "best_class_recall": "best_recall",
    }
    per_class_summary = selection_summary.get("per_class", {})
    for export_strategy, summary_key in per_class_strategy_keys.items():
        selected_type_by_class = {
            class_name: class_summary.get(summary_key, {}).get("selected_probe_type")
            for class_name, class_summary in per_class_summary.items()
        }
        for result, class_weights in averaged_by_result:
            selected_class_weights = {
                class_name: weights
                for class_name, weights in class_weights.items()
                if selected_type_by_class.get(class_name) == result.get("probe_type")
            }
            if not selected_class_weights:
                continue
            _merge_weight_exports(
                exports,
                selection_strategy=export_strategy,
                probe_type=str(result["probe_type"]),
                layer=str(result["layer"]),
                class_weights=selected_class_weights,
            )

    return _finalize_weight_exports(exports)


def write_summary_csv(results: List[Dict[str, Any]], output_path: Path) -> None:
    step_start = time.perf_counter()
    fieldnames = [
        "vector_pool",
        "probe_type",
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
        "recall_macro",
        "recall_weighted",
        "reason",
    ]
    fieldnames_elapsed = time.perf_counter() - step_start
    step_start = time.perf_counter()
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in fieldnames})
    write_elapsed = time.perf_counter() - step_start
    print(
        "output timing: "
        f"summary_csv_fieldnames={fieldnames_elapsed:.2f}s "
        f"summary_csv_write={write_elapsed:.2f}s",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train multinomial and one-vs-rest linear probes over residual stream "
            "vectors to classify chunk function_tags."
        )
    )
    parser.add_argument("--preset", choices=PRESETS.keys(), default="local", help="Use local or VM path defaults.")
    parser.add_argument(
        "--residual-path",
        type=Path,
        default=None,
        help="Full path to a residual stream JSON or directory of per-problem JSON files. Overrides --residual-input-dir.",
    )
    parser.add_argument(
        "--residual-input-dir",
        type=Path,
        default=None,
        help="Directory containing problem_*/residual stream JSON files.",
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
        "--vector-pools",
        type=str,
        default="mean_vector,last_tokenvector",
        help="Comma-separated list of vector pooling strategies to run.",
    )
    parser.add_argument(
        "--vector-pool",
        type=str,
        default=None,
        help="Legacy single-pool override. If set, only this pool is used.",
    )
    return parser


def _run_for_vector_pool(
    *,
    args: argparse.Namespace,
    vector_pool: str,
    residual_paths: Sequence[Path],
    labels_root: Path,
    output_dir: Path,
    specific_problems: Optional[List[str]],
) -> Dict[str, Any]:
    step_start = time.perf_counter()
    examples_by_probe, metadata = build_examples(
        residual_paths=residual_paths,
        labels_root=labels_root,
        tag_mode=args.tag_mode,
        vector_pool=vector_pool,
        problem_ids=specific_problems,
    )
    print(f"setup timing: build_examples_total[{vector_pool}]={time.perf_counter() - step_start:.2f}s", flush=True)

    all_results: List[Dict[str, Any]] = []
    step_start = time.perf_counter()
    for probe_key in sorted(
        examples_by_probe,
        key=lambda item: (item[0], int(item[1]) if item[1].isdigit() else item[1]),
    ):
        for probe_type in ("multinomial", "one_vs_rest"):
            result = evaluate_probe(probe_key, examples_by_probe[probe_key], args.splits, probe_type)
            result["vector_pool"] = vector_pool
            all_results.append(result)
    print(f"probe timing: evaluate_all_probes_total[{vector_pool}]={time.perf_counter() - step_start:.2f}s", flush=True)

    selection_summary = build_selection_summary(all_results)
    weight_export_results = build_weight_export_results(all_results, selection_summary)

    step_start = time.perf_counter()
    vector_output_dir = output_dir / vector_pool
    vector_output_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = vector_output_dir / "linear_probe_results_test.json"
    summary_path = vector_output_dir / "linear_probe_summary_test.csv"
    selection_path = vector_output_dir / "linear_probe_selection_summary_test.json"
    print(f"output timing: mkdir_and_paths[{vector_pool}]={time.perf_counter() - step_start:.2f}s", flush=True)

    step_start = time.perf_counter()
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    **metadata,
                    "vector_pool": vector_pool,
                    "probe_types": ["multinomial", "one_vs_rest"],
                },
                "selection_summary": selection_summary,
                "results": weight_export_results,
            },
            f,
            indent=2,
        )
    print(f"output timing: detailed_json_write[{vector_pool}]={time.perf_counter() - step_start:.2f}s", flush=True)

    step_start = time.perf_counter()
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(selection_summary, f, indent=2)
    print(f"output timing: selection_json_write[{vector_pool}]={time.perf_counter() - step_start:.2f}s", flush=True)

    write_summary_csv(all_results, summary_path)

    ok_count = sum(result["status"] == "ok" for result in all_results)
    skipped_count = len(all_results) - ok_count
    print(f"Evaluated {ok_count} probes for {vector_pool}; skipped {skipped_count}.")
    print(f"Saved detailed results to {detailed_path}")
    print(f"Saved summary CSV to {summary_path}")
    print(f"Saved selection summary to {selection_path}")

    return {
        "vector_pool": vector_pool,
        "metadata": metadata,
        "selection_summary": selection_summary,
        "num_weight_export_records": count_weight_export_records(weight_export_results),
        "detailed_path": str(detailed_path),
        "summary_path": str(summary_path),
        "selection_path": str(selection_path),
    }


def main() -> None:
    run_start = time.perf_counter()
    args = build_parser().parse_args()
    if args.splits < 2:
        raise ValueError("--splits must be at least 2.")

    step_start = time.perf_counter()
    specific_problems = parse_problem_ids(args.specific_problems)
    residual_paths = build_residual_paths(args, specific_problems)
    labels_root = build_labels_root(args)
    output_dir = build_output_dir(args)
    print(f"setup timing: resolve_paths={time.perf_counter() - step_start:.2f}s", flush=True)

    vector_pools = [args.vector_pool] if args.vector_pool else parse_vector_pools(args.vector_pools)

    run_summaries = []
    for vector_pool in vector_pools:
        run_summaries.append(
            _run_for_vector_pool(
                args=args,
                vector_pool=vector_pool,
                residual_paths=residual_paths,
                labels_root=labels_root,
                output_dir=output_dir,
                specific_problems=specific_problems,
            )
        )

    print(f"Total time: {time.perf_counter() - run_start:.2f}s", flush=True)
    print(json.dumps({"runs": run_summaries}, indent=2))


if __name__ == "__main__":
    main()
