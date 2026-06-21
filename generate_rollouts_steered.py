import argparse
import csv
import json
import math
import os
import random
from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from utils import check_answer, extract_boxed_answers, load_math_problems


load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
HF_KEY = os.getenv("HF_TOKEN")

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR
    / "math_rollouts_steered"
    / "deepseek-r1-distill-qwen-14b"
    / "temperature_0.6_top_p_0.95"
    / "correct_base_solution"
)
DEFAULT_STEERING_VECTOR_PATH = (
    SCRIPT_DIR
    / "whitebox-analyses"
    / "residual_stream_analysis"
    / "Results"
    / "linear_probes"
    / "last_tokenvector"
    / "linear_probe_results_test.json"
)
DEFAULT_SCALING_CSV = (
    SCRIPT_DIR
    / "whitebox-analyses"
    / "residual_stream_analysis"
    / "Inputs"
    / "tag_wise_scaling.csv"
)
HARDCODED_PROBLEM_IDS = [
    330,
    1591,
    2050,
    2137,
    2189,
    2236,
    2238,
    2870,
    3360,
    3448,
    3550,
    3916,
    3935,
    4019,
    4164,
    4605,
    4682,
    6481,
    6596,
    6998,
]
STEERING_TAGS = ("plan_generation", "uncertainty_management", "fact_retrieval")
DEFAULT_SELECTION_STRATEGY = "best_overall_f1"
SUPPORTED_REGRESSOR_TYPE = "multinomial"


class StreamingChunkContext:
    """Tracks generated chunks from streamed text using lightweight sentence boundaries."""

    def __init__(self, chunk_min: int = 0, chunk_max: int = 250):
        self.chunk_min = chunk_min
        self.chunk_max = chunk_max
        self.generated_text = ""
        self.current_chunk = ""
        self.chunks: List[str] = []

    @property
    def num_chunks_generated(self) -> int:
        return len(self.chunks)

    @property
    def chunk_idx_scaled(self) -> float:
        if self.chunk_max <= self.chunk_min:
            return 0.0
        value = (self.num_chunks_generated - self.chunk_min) / (self.chunk_max - self.chunk_min)
        return float(min(1.0, max(0.0, value)))

    def feed(self, text: str) -> None:
        for char in text:
            self.generated_text += char
            self.current_chunk += char
            self._maybe_close_chunk()

    def _maybe_close_chunk(self) -> None:
        text = self.generated_text
        i = len(text) - 1

        is_paragraph_end = any(text.endswith(pattern) for pattern in ("\n\n", "\r\n\r\n"))

        is_sentence_end = False
        if i < len(text) - 1 and text[i] in {".", "?", "!"}:
            next_char = text[i + 1]
            is_sentence_end = next_char == " " or next_char == "\n"

        # Streaming detection can only know a sentence is complete once the
        # following whitespace token has arrived, so also check the previous char.
        if not is_sentence_end and len(text) >= 2:
            prev_char = text[-2]
            if prev_char in {".", "?", "!"} and char_is_boundary_space(text[-1]):
                is_sentence_end = True

        if is_paragraph_end or is_sentence_end:
            chunk = self.current_chunk.strip()
            if chunk:
                self.chunks.append(chunk)
            self.current_chunk = ""
            self._merge_small_chunks()

    def _merge_small_chunks(self) -> None:
        i = 0
        while i < len(self.chunks):
            if len(self.chunks[i]) < 10:
                if i == len(self.chunks) - 1:
                    if i > 0:
                        self.chunks[i - 1] = self.chunks[i - 1] + " " + self.chunks[i]
                        self.chunks.pop(i)
                else:
                    self.chunks[i + 1] = self.chunks[i] + " " + self.chunks[i + 1]
                    self.chunks.pop(i)
                    continue
                if i == 0 and len(self.chunks) == 1:
                    break
            else:
                i += 1

    def snapshot(self) -> Dict[str, Any]:
        return {
            "num_chunks_generated": self.num_chunks_generated,
            "chunk_idx_scaled": self.chunk_idx_scaled,
            "current_chunk": self.current_chunk,
            "chunks": self.chunks,
        }


def char_is_boundary_space(char: str) -> bool:
    return char == " " or char == "\n"


def parse_layer_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    layers = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        layers.append(int(item))
    return sorted(set(layers))


class TagWiseScaling:
    def __init__(self, csv_path: Path, tags: Iterable[str]):
        self.values: Dict[str, List[Tuple[float, float]]] = {tag: [] for tag in tags}
        tags = tuple(tags)
        rows_by_x: Dict[float, Dict[str, float]] = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tag = row.get("function_tag", "")
                if tag not in self.values:
                    continue
                try:
                    x = float(row["chunk_idx_scaled"])
                    y = float(row["counterfactual_importance_kl"])
                except (KeyError, TypeError, ValueError):
                    continue
                rows_by_x.setdefault(x, {})[tag] = max(y, 0.0)

        for x in sorted(rows_by_x):
            values_at_x = rows_by_x[x]
            total = sum(values_at_x.get(tag, 0.0) for tag in tags)
            if total > 0:
                normalized = {
                    tag: values_at_x.get(tag, 0.0) / total
                    for tag in tags
                }
            else:
                uniform = 1.0 / len(tags)
                normalized = {tag: uniform for tag in tags}

            for tag in tags:
                self.values[tag].append((x, float(normalized[tag])))

        for tag, rows in self.values.items():
            if not rows:
                raise ValueError(f"No scaling rows found for function_tag={tag!r} in {csv_path}")

    def ratio(self, tag: str, chunk_idx_scaled: float) -> float:
        rows = self.values[tag]
        xs = [x for x, _ in rows]
        pos = bisect_left(xs, chunk_idx_scaled)
        if pos <= 0:
            return rows[0][1]
        if pos >= len(rows):
            return rows[-1][1]
        left_x, left_y = rows[pos - 1]
        right_x, right_y = rows[pos]
        if right_x == left_x:
            return right_y
        alpha = (chunk_idx_scaled - left_x) / (right_x - left_x)
        return float(left_y + alpha * (right_y - left_y))


def normalize_layer_key(layer: Any) -> int:
    return int(str(layer).removeprefix("layer_"))


def vector_to_tensor(vector: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.as_tensor(vector, dtype=torch.float32, device=device)
    if tensor.ndim > 1:
        tensor = tensor.reshape(-1, tensor.shape[-1])[-1]
    return tensor.to(dtype=dtype)


class SteeringRule:
    def __init__(
        self,
        steering_vectors_path: Path,
        scaling_csv_path: Path,
        selection_strategy: str,
        steering_layers: Optional[Sequence[int]],
        strength_n: float,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.scaling = TagWiseScaling(scaling_csv_path, STEERING_TAGS)
        self.strength_n = strength_n
        self.requested_layers = None if steering_layers is None else sorted(set(int(layer) for layer in steering_layers))
        self.directions = self._load_directions(
            steering_vectors_path=steering_vectors_path,
            selection_strategy=selection_strategy,
            steering_layers=self.requested_layers,
            device=device,
            dtype=dtype,
        )

    def _load_directions(
        self,
        steering_vectors_path: Path,
        selection_strategy: str,
        steering_layers: Optional[Sequence[int]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Dict[int, torch.Tensor]]:
        with open(steering_vectors_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        results = raw.get("results")
        if not isinstance(results, dict):
            raise ValueError(
                f"{steering_vectors_path} must use the linear-probe result format with "
                "a strategy-keyed 'results' object."
            )
        if selection_strategy not in results:
            raise KeyError(
                f"Selection strategy {selection_strategy!r} not found in {steering_vectors_path}. "
                f"Available: {sorted(results)}"
            )

        strategy_records = results[selection_strategy]
        if not isinstance(strategy_records, list):
            raise ValueError(f"results[{selection_strategy!r}] must be a list of probe records.")

        requested_layer_set = None if steering_layers is None else {int(layer) for layer in steering_layers}
        directions: Dict[str, Dict[int, torch.Tensor]] = {tag: {} for tag in STEERING_TAGS}
        for record in strategy_records:
            if record.get("regressor_type") != SUPPORTED_REGRESSOR_TYPE:
                continue
            layer = normalize_layer_key(record["layer"])
            if requested_layer_set is not None and layer not in requested_layer_set:
                continue
            weights_by_class = record.get("weights_by_class", {})
            if not isinstance(weights_by_class, dict):
                continue
            for tag in STEERING_TAGS:
                if tag not in weights_by_class:
                    continue
                directions[tag][layer] = vector_to_tensor(weights_by_class[tag], device=device, dtype=dtype)

        if requested_layer_set is not None:
            available_layers = set()
            for tag_map in directions.values():
                available_layers.update(tag_map)
            missing_layers = sorted(requested_layer_set - available_layers)
            if missing_layers:
                raise KeyError(
                    f"Requested steering layers {missing_layers} are not available in "
                    f"{steering_vectors_path} for selection strategy {selection_strategy!r}."
                )

        missing_tags = [tag for tag, layer_map in directions.items() if not layer_map]
        if missing_tags:
            raise KeyError(
                f"No {SUPPORTED_REGRESSOR_TYPE!r} steering weights found for tags {missing_tags} "
                f"under selection strategy {selection_strategy!r} in {steering_vectors_path}."
            )
        return directions

    @property
    def layers(self) -> List[int]:
        layer_set = set()
        for tag_map in self.directions.values():
            layer_set.update(tag_map)
        return sorted(layer_set)

    def vector_for_layer(self, layer_idx: int, chunk_idx_scaled: float) -> Optional[torch.Tensor]:
        pieces = []
        for tag in STEERING_TAGS:
            layer_vector = self.directions[tag].get(layer_idx)
            if layer_vector is None:
                continue
            pieces.append(self.scaling.ratio(tag, chunk_idx_scaled) * layer_vector)
        if not pieces:
            return None
        decay = self.strength_n * math.exp(-chunk_idx_scaled)
        return decay * torch.stack(pieces, dim=0).sum(dim=0)


def make_residual_steering_hook(
    layer_idx: int,
    rule: SteeringRule,
    context: StreamingChunkContext,
    hook_call_counts: Dict[int, int],
    hook_seen_chunk_scales: Dict[int, List[float]],
):
    def hook_fn(residual: torch.Tensor, hook) -> torch.Tensor:
        del hook
        hook_call_counts[layer_idx] = hook_call_counts.get(layer_idx, 0) + 1
        scale = float(context.chunk_idx_scaled)
        scales_for_layer = hook_seen_chunk_scales.setdefault(layer_idx, [])
        if not scales_for_layer or not math.isclose(scales_for_layer[-1], scale, rel_tol=0.0, abs_tol=1e-12):
            scales_for_layer.append(scale)

        vector = rule.vector_for_layer(layer_idx, context.chunk_idx_scaled)
        if vector is None:
            return residual
        vector = vector.to(device=residual.device, dtype=residual.dtype)
        if vector.numel() != residual.shape[-1]:
            raise ValueError(
                f"Steering vector dim {vector.numel()} does not match hidden dim "
                f"{residual.shape[-1]} at layer {layer_idx}"
            )

        updated = residual.clone()
        updated[:, -1, :] = updated[:, -1, :] + vector
        return updated

    return hook_fn


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float, top_k: Optional[int]) -> torch.Tensor:
    logits = logits.float()
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
        logits = torch.where(logits < values[:, [-1]], torch.full_like(logits, -float("inf")), logits)
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        logits = torch.full_like(logits, -float("inf"))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_steered(
    model: Any,
    prompt: str,
    rule: Optional[SteeringRule],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
) -> Dict[str, Any]:
    context = StreamingChunkContext()
    prompt_tokens = model.to_tokens(prompt)
    prompt_len = int(prompt_tokens.shape[1])
    hook_call_counts: Dict[int, int] = {}
    hook_seen_chunk_scales: Dict[int, List[float]] = {}
    chunk_transitions: List[Dict[str, Any]] = []

    hook_specs = []
    if rule is not None:
        hook_specs = [
            (
                f"blocks.{layer_idx}.hook_resid_post",
                make_residual_steering_hook(
                    layer_idx,
                    rule,
                    context,
                    hook_call_counts,
                    hook_seen_chunk_scales,
                ),
            )
            for layer_idx in rule.layers
        ]
    generated_ids: List[int] = []
    eos_ids = {model.tokenizer.eos_token_id}
    if getattr(model.tokenizer, "pad_token_id", None) is not None:
        eos_ids.discard(model.tokenizer.pad_token_id)
    tokens = prompt_tokens

    with model.hooks(fwd_hooks=hook_specs):
        for _ in range(max_tokens):
            with torch.no_grad():
                logits = model(tokens, return_type="logits")
            next_token = sample_next_token(logits[:, -1, :], temperature, top_p, top_k)
            token_id = int(next_token[0, 0].item())
            if token_id in eos_ids:
                break

            generated_ids.append(token_id)
            token_text = model.tokenizer.decode([token_id], skip_special_tokens=True)
            previous_chunk_count = context.num_chunks_generated
            context.feed(token_text)
            if context.num_chunks_generated > previous_chunk_count:
                chunk_transitions.append(
                    {
                        "completion_tokens_after_feed": len(generated_ids),
                        "num_chunks_generated": context.num_chunks_generated,
                        "chunk_idx_scaled": float(context.chunk_idx_scaled),
                    }
                )
            tokens = torch.cat([tokens, next_token.to(tokens.device)], dim=1)

    if rule is not None and generated_ids:
        layers_without_calls = [layer_idx for layer_idx in rule.layers if hook_call_counts.get(layer_idx, 0) == 0]
        if layers_without_calls:
            raise RuntimeError(
                f"Steering hooks were registered but never invoked for layers {layers_without_calls}."
            )

    chunk_transition_verification = []
    for event in chunk_transitions:
        should_affect_future_tokens = event["completion_tokens_after_feed"] < len(generated_ids)
        if rule is None or not should_affect_future_tokens:
            verification_passed = True
        else:
            verification_passed = all(
                any(
                    math.isclose(scale, event["chunk_idx_scaled"], rel_tol=0.0, abs_tol=1e-12)
                    for scale in hook_seen_chunk_scales.get(layer_idx, [])
                )
                for layer_idx in rule.layers
            )
        chunk_transition_verification.append(
            {
                **event,
                "should_affect_future_tokens": should_affect_future_tokens,
                "verified_next_step_used_new_scale": verification_passed,
            }
        )

    rollout_text = model.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "text": rollout_text,
        "finish_reason": "length" if len(generated_ids) >= max_tokens else "stop",
        "usage": {"completion_tokens": len(generated_ids), "total_tokens": prompt_len + len(generated_ids)},
        "chunk_context": context.snapshot(),
        "steering_debug": {
            "enabled": rule is not None,
            "hook_layers": [] if rule is None else list(rule.layers),
            "hook_call_counts": {str(layer): count for layer, count in sorted(hook_call_counts.items())},
            "hook_seen_chunk_scales": {
                str(layer): scales for layer, scales in sorted(hook_seen_chunk_scales.items())
            },
            "chunk_transitions": chunk_transition_verification,
            "use_past_kv_cache": False,
        },
    }


def load_local_model(args: argparse.Namespace) -> Any:
    try:
        from transformer_lens import HookedTransformer
    except ImportError as exc:
        raise ImportError(
            "transformer_lens is required for generate_rollouts_steered.py. "
            "Install it in this environment before running the script."
        ) from exc

    if args.quantize:
        raise NotImplementedError("`--quantize` is not supported with the TransformerLens loading path.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = HookedTransformer.from_pretrained_no_processing(
        args.model,
        device=device,
        dtype=dtype,
        default_padding_side="left",
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        refactor_factored_attn_matrices=False,
        token=HF_KEY,
        trust_remote_code=True,
    )
    model.eval()
    if model.tokenizer is not None and model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    return model


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def discover_problem_dirs(include_problems: Optional[str]) -> List[Path]:
    available_problem_ids = HARDCODED_PROBLEM_IDS
    if include_problems:
        wanted = {item.strip().removeprefix("problem_") for item in include_problems.split(",") if item.strip()}
        missing_problem_ids = sorted(int(problem_id) for problem_id in wanted if int(problem_id) not in available_problem_ids)
        if missing_problem_ids:
            raise KeyError(
                f"Requested include_problems IDs are not in the hardcoded problem set: {missing_problem_ids}"
            )
        problem_ids = sorted(int(problem_id) for problem_id in wanted)
    else:
        problem_ids = list(available_problem_ids)
    return [Path(f"problem_{problem_id}") for problem_id in problem_ids]


def load_problem_map(problem_dirs: Sequence[Path], split: str) -> Dict[str, Dict[str, Any]]:
    problem_ids = [int(path.name.removeprefix("problem_")) for path in problem_dirs]
    loaded = load_math_problems(
        split=split,
        include_problems=problem_ids,
        num_problems=None,
        hf_key=HF_KEY,
    )
    return {str(problem_idx): problem for problem_idx, problem in loaded}


def rollout_prompt(problem: Dict[str, Any]) -> str:
    return (
        "Solve this math problem step by step. You MUST put your final answer in \\boxed{}. "
        f"Problem: {problem['problem']} Solution: \n<think>\n"
    )


def write_problem_metadata(problem: Dict[str, Any], output_problem_dir: Path, force: bool) -> None:
    problem_file = output_problem_dir / "problem.json"
    if force or not problem_file.exists():
        write_json(problem_file, problem)


def process_problem_dir(
    problem_dir: Path,
    problem: Dict[str, Any],
    output_dir: Path,
    model: Any,
    rule: Optional[SteeringRule],
    args: argparse.Namespace,
) -> None:
    problem_id = problem_dir.name.removeprefix("problem_")
    output_problem_dir = output_dir / problem_dir.name
    write_problem_metadata(problem, output_problem_dir, args.force)
    solutions_file = output_problem_dir / "solutions.json"
    existing_solutions = []
    if solutions_file.exists() and not args.force:
        existing_solutions = read_json(solutions_file)

    valid_existing = [solution for solution in existing_solutions if "error" not in solution]
    needed = args.num_rollouts - len(valid_existing)
    if needed <= 0:
        return

    prompt = rollout_prompt(problem)
    new_solutions = []
    for _ in tqdm(range(needed), desc=f"problem_{problem_id} rollouts", leave=False):
        try:
            result = generate_steered(
                model=model,
                prompt=prompt,
                rule=rule,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            rollout_text = result["text"]
            extracted_answers = extract_boxed_answers(rollout_text)
            answer = extracted_answers[0] if extracted_answers else ""
            is_correct = bool(problem.get("gt_answer") and answer and check_answer(answer, problem["gt_answer"]))

            new_solutions.append(
                {
                    "prompt": prompt,
                    "rollout": rollout_text,
                    "full_cot": f"{prompt}{rollout_text}",
                    "answer": answer,
                    "is_correct": is_correct,
                    "steered_chunks": result["chunk_context"]["chunks"],
                    "chunk_context": result["chunk_context"],
                    "steering_debug": result["steering_debug"],
                }
            )
        except Exception as exc:
            new_solutions.append(
                {
                    "prompt": prompt,
                    "error": str(exc),
                }
            )

    write_json(solutions_file, existing_solutions + new_solutions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate local rollouts with residual stream steering.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Hugging Face MATH split to load problem statements from.",
    )
    parser.add_argument("--steering_vectors", type=Path, default=DEFAULT_STEERING_VECTOR_PATH)
    parser.add_argument("--scaling_csv", type=Path, default=DEFAULT_SCALING_CSV)
    parser.add_argument(
        "--selection_strategy",
        type=str,
        default=DEFAULT_SELECTION_STRATEGY,
        choices=["best_overall_f1", "best_overall_recall", "best_class_f1", "best_class_recall"],
        help="Selection strategy key to read from linear_probe_results_test.json.",
    )
    parser.add_argument(
        "--steering_layers",
        type=str,
        default=None,
        help="Comma-separated layer numbers to steer. Default is all available steering layers.",
    )
    parser.add_argument(
        "--disable_steering",
        action="store_true",
        help="Generate normal rollouts without applying steering hooks.",
    )
    parser.add_argument("-n", "--strength_n", type=float, default=1.0)
    parser.add_argument("-nr", "--num_rollouts", type=int, default=100)
    parser.add_argument("-t", "--temperature", type=float, default=0.6)
    parser.add_argument("-tp", "--top_p", type=float, default=0.95)
    parser.add_argument("-tk", "--top_k", type=int, default=None)
    parser.add_argument("-mt", "--max_tokens", type=int, default=16384)
    parser.add_argument("-s", "--seed", type=int, default=44)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("-ip", "--include_problems", type=str, default=None)
    parser.add_argument("-q", "--quantize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not args.disable_steering:
        if not args.steering_vectors.is_file():
            raise FileNotFoundError(f"Steering vectors file not found: {args.steering_vectors}")
        if not args.scaling_csv.is_file():
            raise FileNotFoundError(f"Scaling CSV not found: {args.scaling_csv}")

    print(f"Loading local model: {args.model}")
    model = load_local_model(args)
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    steering_layers = parse_layer_list(args.steering_layers)
    rule = None
    if not args.disable_steering:
        rule = SteeringRule(
            steering_vectors_path=args.steering_vectors,
            scaling_csv_path=args.scaling_csv,
            selection_strategy=args.selection_strategy,
            steering_layers=steering_layers,
            strength_n=args.strength_n,
            device=device,
            dtype=dtype,
        )

    problem_dirs = discover_problem_dirs(args.include_problems)
    if not problem_dirs:
        raise FileNotFoundError("No problem IDs available in the hardcoded problem set.")
    problem_map = load_problem_map(problem_dirs, args.split)
    missing_problem_ids = [
        path.name.removeprefix("problem_")
        for path in problem_dirs
        if path.name.removeprefix("problem_") not in problem_map
    ]
    if missing_problem_ids:
        raise KeyError(
            f"Failed to load these problem IDs from Hugging Face split {args.split!r}: {missing_problem_ids}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing steered rollouts to: {args.output_dir}")
    for problem_dir in tqdm(problem_dirs, desc="problems"):
        problem_id = problem_dir.name.removeprefix("problem_")
        process_problem_dir(problem_dir, problem_map[problem_id], args.output_dir, model, rule, args)


if __name__ == "__main__":
    main()
