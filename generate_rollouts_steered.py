import argparse
import csv
import json
import math
import os
import random
from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import check_answer, extract_boxed_answers, split_solution_into_chunks


load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
HF_KEY = os.getenv("HF_TOKEN")

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
DEFAULT_INPUT_DIR = (
    SCRIPT_DIR
    / "math_rollouts"
    / "deepseek-r1-distill-qwen-14b"
    / "temperature_0.6_top_p_0.95"
    / "correct_base_solution"
)
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
    / "steering_vectors.json"
)
DEFAULT_SCALING_CSV = (
    SCRIPT_DIR
    / "whitebox-analyses"
    / "residual_stream_analysis"
    / "Inputs"
    / "tag_wise_scaling.csv"
)
STEERING_TAGS = ("plan_generation", "uncertainty_management", "fact_retrieval")


class StreamingChunkContext:
    """Tracks generated chunks using the same boundary rules as split_solution_into_chunks."""

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


class TagWiseScaling:
    def __init__(self, csv_path: Path, tags: Iterable[str]):
        self.values: Dict[str, List[Tuple[float, float]]] = {tag: [] for tag in tags}
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
                self.values[tag].append((x, y))

        for tag, rows in self.values.items():
            rows.sort(key=lambda item: item[0])
            if not rows:
                raise ValueError(f"No scaling rows found for function_tag={tag!r} in {csv_path}")
            raw = [value for _, value in rows]
            min_value = min(raw)
            max_value = max(raw)
            denom = max_value - min_value
            self.values[tag] = [
                (x, 0.0 if denom == 0 else (value - min_value) / denom)
                for x, value in rows
            ]

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
        probe_type: Optional[str],
        strength_n: float,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.scaling = TagWiseScaling(scaling_csv_path, STEERING_TAGS)
        self.strength_n = strength_n
        self.directions = self._load_directions(
            steering_vectors_path=steering_vectors_path,
            probe_type=probe_type,
            device=device,
            dtype=dtype,
        )

    def _load_directions(
        self,
        steering_vectors_path: Path,
        probe_type: Optional[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Dict[int, torch.Tensor]]:
        with open(steering_vectors_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        directions: Dict[str, Dict[int, torch.Tensor]] = {}
        for tag in STEERING_TAGS:
            if tag not in raw:
                raise KeyError(f"Missing steering vector tag {tag!r} in {steering_vectors_path}")
            probe_map = raw[tag]
            if not isinstance(probe_map, dict) or not probe_map:
                raise ValueError(f"Steering vectors for {tag!r} must be a non-empty dict")
            selected_probe = probe_type or next(iter(probe_map))
            if selected_probe not in probe_map:
                raise KeyError(
                    f"Probe type {selected_probe!r} not found for {tag!r}. "
                    f"Available: {sorted(probe_map)}"
                )
            layer_map = probe_map[selected_probe]
            directions[tag] = {
                normalize_layer_key(layer): vector_to_tensor(vector, device=device, dtype=dtype)
                for layer, vector in layer_map.items()
            }
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


class ResidualSteeringHook:
    def __init__(self, model: torch.nn.Module, rule: SteeringRule, context: StreamingChunkContext):
        self.model = model
        self.rule = rule
        self.context = context
        self.handles = []
        self.enabled = False

    def _hook(self, layer_idx: int):
        def hook_fn(module, input_tuple, output):
            if not self.enabled:
                return output
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = ()

            vector = self.rule.vector_for_layer(layer_idx, self.context.chunk_idx_scaled)
            if vector is None:
                return output
            vector = vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
            if vector.numel() != hidden_states.shape[-1]:
                raise ValueError(
                    f"Steering vector dim {vector.numel()} does not match hidden dim "
                    f"{hidden_states.shape[-1]} at layer {layer_idx}"
                )

            modified = hidden_states.clone()
            modified[:, -1, :] = modified[:, -1, :] + vector
            if rest:
                return (modified,) + rest
            return modified

        return hook_fn

    def register(self) -> None:
        self.remove()
        layer_modules = find_decoder_layers(self.model)
        missing_layers = [layer for layer in self.rule.layers if layer >= len(layer_modules)]
        if missing_layers:
            raise ValueError(
                f"Requested steering layers {missing_layers} but model has {len(layer_modules)} layers"
            )
        for layer_idx in self.rule.layers:
            self.handles.append(layer_modules[layer_idx].register_forward_hook(self._hook(layer_idx)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def find_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Could not locate decoder layers on the local model.")


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
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    rule: SteeringRule,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
) -> Dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    context = StreamingChunkContext()
    hook = ResidualSteeringHook(model, rule, context)
    hook.register()
    hook.enabled = True

    generated_ids: List[int] = []
    past_key_values = None
    next_input_ids = input_ids
    eos_ids = {tokenizer.eos_token_id}
    if getattr(tokenizer, "pad_token_id", None) is not None:
        eos_ids.discard(tokenizer.pad_token_id)

    try:
        for step in range(max_tokens):
            with torch.no_grad():
                outputs = model(
                    input_ids=next_input_ids,
                    attention_mask=attention_mask if past_key_values is None else None,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = outputs.past_key_values
            next_token = sample_next_token(outputs.logits[:, -1, :], temperature, top_p, top_k)
            token_id = int(next_token.item())
            if token_id in eos_ids:
                break
            generated_ids.append(token_id)
            token_text = tokenizer.decode([token_id], skip_special_tokens=True)
            context.feed(token_text)
            next_input_ids = next_token
    finally:
        hook.remove()

    rollout_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "text": rollout_text,
        "finish_reason": "length" if len(generated_ids) >= max_tokens else "stop",
        "usage": {"completion_tokens": len(generated_ids), "total_tokens": input_ids.shape[1] + len(generated_ids)},
        "chunk_context": context.snapshot(),
    }


def load_local_model(args: argparse.Namespace) -> Tuple[torch.nn.Module, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        token=HF_KEY,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    load_kwargs = {
        "token": HF_KEY,
        "trust_remote_code": True,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    if torch.cuda.is_available():
        load_kwargs["torch_dtype"] = torch.float16
    if args.quantize and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()
    return model, tokenizer


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def discover_problem_dirs(input_dir: Path, include_problems: Optional[str]) -> List[Path]:
    if include_problems:
        wanted = {item.strip().removeprefix("problem_") for item in include_problems.split(",") if item.strip()}
        return [input_dir / f"problem_{problem_id}" for problem_id in sorted(wanted, key=lambda x: int(x))]
    return sorted(input_dir.glob("problem_*"), key=lambda path: int(path.name.removeprefix("problem_")))


def build_cumulative_chunks(chunks: List[str]) -> List[str]:
    cumulative = []
    current = ""
    for chunk in chunks:
        current += chunk + " "
        cumulative.append(current.strip())
    return cumulative


def rollout_prompt(problem: Dict[str, Any], prefix_without_chunk: str) -> str:
    return (
        "Solve this math problem step by step. You MUST put your final answer in \\boxed{}. "
        f"Problem: {problem['problem']} Solution: \n<think>\n{prefix_without_chunk}"
    )


def copy_problem_inputs(source_problem_dir: Path, output_problem_dir: Path, force: bool) -> Tuple[Dict[str, Any], List[str]]:
    problem = read_json(source_problem_dir / "problem.json")
    chunks_data = read_json(source_problem_dir / "chunks.json")
    for filename in ("problem.json", "base_solution.json", "chunks.json"):
        source = source_problem_dir / filename
        target = output_problem_dir / filename
        if source.is_file() and (force or not target.exists()):
            write_json(target, read_json(source))
    return problem, chunks_data["chunks"]


def process_problem_dir(
    problem_dir: Path,
    output_dir: Path,
    model: torch.nn.Module,
    tokenizer: Any,
    rule: SteeringRule,
    args: argparse.Namespace,
) -> None:
    problem_id = problem_dir.name.removeprefix("problem_")
    output_problem_dir = output_dir / problem_dir.name
    problem, chunks = copy_problem_inputs(problem_dir, output_problem_dir, args.force)
    cumulative_chunks = build_cumulative_chunks(chunks)

    chunk_iter = list(enumerate(zip(chunks, cumulative_chunks)))
    if args.include_chunks:
        wanted_chunks = {int(item.strip()) for item in args.include_chunks.split(",") if item.strip()}
        chunk_iter = [(idx, pair) for idx, pair in chunk_iter if idx in wanted_chunks]
    if args.max_chunks is not None:
        chunk_iter = [(idx, pair) for idx, pair in chunk_iter if idx < args.max_chunks]

    for chunk_idx, (chunk, full_prefix) in tqdm(chunk_iter, desc=f"problem_{problem_id} chunks"):
        chunk_dir = output_problem_dir / f"chunk_{chunk_idx}"
        solutions_file = chunk_dir / "solutions.json"
        existing_solutions = []
        if solutions_file.exists() and not args.force:
            existing_solutions = read_json(solutions_file)

        valid_existing = [solution for solution in existing_solutions if "error" not in solution]
        needed = args.num_rollouts - len(valid_existing)
        if needed <= 0:
            continue

        prefix_without_chunk = full_prefix.replace(chunk, "").strip()
        prompt = rollout_prompt(problem, prefix_without_chunk)
        new_solutions = []

        for _ in tqdm(range(needed), desc=f"problem_{problem_id} chunk_{chunk_idx} rollouts", leave=False):
            try:
                result = generate_steered(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    rule=rule,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
                rollout_text = result["text"]
                generated_chunks = result["chunk_context"]["chunks"]
                fallback_chunks = split_solution_into_chunks(rollout_text) if rollout_text else []
                chunk_resampled = (generated_chunks or fallback_chunks or [""])[0]
                extracted_answers = extract_boxed_answers(rollout_text)
                answer = extracted_answers[0] if extracted_answers else ""
                is_correct = bool(problem.get("gt_answer") and answer and check_answer(answer, problem["gt_answer"]))

                new_solutions.append(
                    {
                        "chunk_removed": chunk,
                        "prefix_without_chunk": prefix_without_chunk,
                        "chunk_resampled": chunk_resampled,
                        "rollout": rollout_text,
                        "full_cot": f"{prompt}{rollout_text}",
                        "answer": answer,
                        "is_correct": is_correct,
                        "steered_chunks": generated_chunks,
                        "chunk_context": result["chunk_context"],
                    }
                )
            except Exception as exc:
                new_solutions.append(
                    {
                        "chunk_removed": chunk,
                        "prefix_without_chunk": prefix_without_chunk,
                        "error": str(exc),
                    }
                )

        write_json(solutions_file, existing_solutions + new_solutions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate local rollouts with residual stream steering.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--steering_vectors", type=Path, default=DEFAULT_STEERING_VECTOR_PATH)
    parser.add_argument("--scaling_csv", type=Path, default=DEFAULT_SCALING_CSV)
    parser.add_argument("--probe_type", type=str, default=None)
    parser.add_argument("-n", "--strength_n", type=float, default=1.0)
    parser.add_argument("-nr", "--num_rollouts", type=int, default=100)
    parser.add_argument("-t", "--temperature", type=float, default=0.6)
    parser.add_argument("-tp", "--top_p", type=float, default=0.95)
    parser.add_argument("-tk", "--top_k", type=int, default=None)
    parser.add_argument("-mt", "--max_tokens", type=int, default=16384)
    parser.add_argument("-mc", "--max_chunks", type=int, default=275)
    parser.add_argument("-s", "--seed", type=int, default=44)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("-ip", "--include_problems", type=str, default=None)
    parser.add_argument("-ic", "--include_chunks", type=str, default=None)
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

    if not args.steering_vectors.is_file():
        raise FileNotFoundError(f"Steering vectors file not found: {args.steering_vectors}")
    if not args.scaling_csv.is_file():
        raise FileNotFoundError(f"Scaling CSV not found: {args.scaling_csv}")

    print(f"Loading local model: {args.model}")
    model, tokenizer = load_local_model(args)
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    rule = SteeringRule(
        steering_vectors_path=args.steering_vectors,
        scaling_csv_path=args.scaling_csv,
        probe_type=args.probe_type,
        strength_n=args.strength_n,
        device=device,
        dtype=dtype,
    )

    problem_dirs = discover_problem_dirs(args.input_dir, args.include_problems)
    problem_dirs = [path for path in problem_dirs if path.is_dir()]
    if not problem_dirs:
        raise FileNotFoundError(f"No problem_* directories found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing steered rollouts to: {args.output_dir}")
    for problem_dir in tqdm(problem_dirs, desc="problems"):
        process_problem_dir(problem_dir, args.output_dir, model, tokenizer, rule, args)


if __name__ == "__main__":
    main()
