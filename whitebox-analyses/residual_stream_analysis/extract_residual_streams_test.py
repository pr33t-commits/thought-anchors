import argparse
import json
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


PRESETS = {
    "local": {
        "chunk_input_dir": PROJECT_ROOT / "math_rollouts",
        "model": "deepseek-r1-distill-llama-8b",
        "hf_model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "device": "cpu",
        "device_map": None,
        "output": SCRIPT_DIR / "Input" / "residual_stream_extracts.json",
    },
    "vm": {
        "chunk_input_dir": Path("/workspace/math_rollouts"),
        "model": "deepseek-r1-distill-llama-8b",
        "hf_model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "device": None,
        "device_map": "cuda",
        "output": Path("/home/lodaya_dimpal/storage/residual_stream_analysis/Input/residual_stream_extracts.json"),
    },
}


def parse_problem_ids(problem_ids: Optional[str]) -> Optional[List[str]]:
    if not problem_ids:
        return None
    return [problem.strip().removeprefix("problem_") for problem in problem_ids.split(",") if problem.strip()]


def parse_dtype(dtype_name: str):
    import torch

    dtypes = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return dtypes[dtype_name]


def to_jsonable(value: Any) -> Any:
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def build_chunks_input_dir(args: argparse.Namespace) -> Path:
    chunk_input_dir = args.chunk_input_dir or PRESETS[args.preset]["chunk_input_dir"]
    model = args.model or PRESETS[args.preset]["model"]
    chunks_input_dir = (
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

    return chunks_input_dir / dir_name


def discover_problem_ids(input_dir: Path) -> List[str]:
    return sorted(
        problem.name.split("_")[-1]
        for problem in input_dir.iterdir()
        if problem.is_dir() and problem.name.startswith("problem_")
    )


def load_chunks_to_include(path: Optional[Path]) -> Optional[Dict[str, Optional[List[int]]]]:
    if not path or not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        chunks_to_include = json.load(f)
    if not chunks_to_include:
        return None
    return {
        str(problem).removeprefix("problem_"): chunks
        for problem, chunks in chunks_to_include.items()
    }


def load_problem(problem_dir: Path) -> Tuple[Dict[str, Any], List[str]]:
    with open(problem_dir / "base_solution.json", "r", encoding="utf-8") as f:
        base_solution = json.load(f)
    with open(problem_dir / "chunks.json", "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    chunks = chunks_data.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError(f"chunks must be a list in {problem_dir / 'chunks.json'}")
    return base_solution, chunks


def find_chunk_char_spans(full_text: str, prompt: str, chunks: List[str]) -> List[Tuple[int, int]]:
    spans = []
    search_start = len(prompt)
    for chunk_idx, chunk in enumerate(chunks):
        
        start = full_text.find(chunk, search_start)
        if start != -1:
            
            end = start + len(chunk)
        else:
            pieces = [re.escape(piece) for piece in re.split(r"\s+", chunk.strip()) if piece]
            pattern = r"\s+".join(pieces)
            match = re.search(pattern, full_text[search_start:], flags=re.DOTALL)
            if match:
                start = search_start + match.start()
                end = search_start + match.end()
            else:
                context = full_text[search_start : search_start + 200].replace("\n", "\\n")
                raise ValueError(
                    f"Could not align chunk {chunk_idx}: {chunk[:80]!r}. "
                    f"Search context starts with: {context!r}"
                )
        if end <= start:
            raise ValueError(f"Could not align chunk {chunk_idx}: {chunk[:80]!r}")
        spans.append((start, end))
        search_start = end
        # if chunk_idx in [72,"72"]:
        #     print(f"Chunk {chunk_idx} span: {start}-{end}")
        #     print(chunk)
        #     print(full_text.find(chunk, search_start))
        if chunk_idx % 12 == 0:
            # print(full_text.find(chunk, search_start))
            
            print(chunk_idx)
            print(start, end)
            # print(chunk[:50])
            print(full_text[start:end])
    return spans


def tokenize_with_offsets(model, text: str):
    import torch

    tokenizer = model.tokenizer
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"][0].tolist()]

    device = getattr(model, "cfg", None)
    if device is not None and getattr(model.cfg, "device", None):
        input_ids = input_ids.to(model.cfg.device)
    elif hasattr(model, "W_E"):
        input_ids = input_ids.to(model.W_E.device)
    elif torch.cuda.is_available():
        input_ids = input_ids.to("cuda")

    return input_ids, offsets


def display_slice(text: str, start: int, end: int) -> str:
    return (
        repr(text[start:end])
        .replace("\\r", "<CR>")
        .replace("\\n", "<NL>")
        .replace("\\t", "<TAB>")
    )


def print_token_window(
    tokenizer,
    input_ids,
    offsets: List[Tuple[int, int]],
    text: str,
    start: int,
    count: int,
) -> None:
    end = min(start + count, len(offsets))
    print(f"Token window [{start}:{end}]", flush=True)
    for token_idx in range(start, end):
        token_id = int(input_ids[0, token_idx].item())
        token_text = tokenizer.convert_ids_to_tokens(token_id)
        token_start, token_end = offsets[token_idx]
        print(
            token_idx,
            token_id,
            repr(token_text),
            (token_start, token_end),
            display_slice(text, token_start, token_end),
            flush=True,
        )


def print_offset_anomalies(
    tokenizer,
    input_ids,
    offsets: List[Tuple[int, int]],
    text: str,
    limit: int,
) -> None:
    printed = 0
    for token_idx in range(1, len(offsets)):
        prev_start, prev_end = offsets[token_idx - 1]
        token_start, token_end = offsets[token_idx]
        if prev_end == token_start:
            continue

        relation = "gap" if token_start > prev_end else "overlap"
        print(
            f"{relation} before token {token_idx}: "
            f"prev={(prev_start, prev_end)} current={(token_start, token_end)} "
            f"between={display_slice(text, min(prev_end, token_start), max(prev_end, token_start))}",
            flush=True,
        )
        window_start = max(0, token_idx - 3)
        print_token_window(tokenizer, input_ids, offsets, text, window_start, 7)
        printed += 1
        if printed >= limit:
            break

    if printed == 0:
        print("No offset gaps/overlaps found.", flush=True)


def print_chunk_alignment_debug(
    tokenizer,
    input_ids,
    offsets: List[Tuple[int, int]],
    text: str,
    chunk_idx: int,
    chunk_start: int,
    chunk_end: int,
) -> None:
    before = [
        token_idx
        for token_idx, (_, token_end) in enumerate(offsets)
        if token_end <= chunk_start
    ]
    after = [
        token_idx
        for token_idx, (token_start, _) in enumerate(offsets)
        if token_start >= chunk_end
    ]
    crossing = [
        token_idx
        for token_idx, (token_start, token_end) in enumerate(offsets)
        if token_start <= chunk_start and token_end >= chunk_end
    ]

    print(f"Chunk {chunk_idx} alignment debug", flush=True)
    print(f"Chunk span: {(chunk_start, chunk_end)} length={chunk_end - chunk_start}", flush=True)
    print(f"Chunk text: {display_slice(text, chunk_start, chunk_end)}", flush=True)
    print(
        "Context: "
        f"{display_slice(text, max(0, chunk_start - 120), min(len(text), chunk_end + 120))}",
        flush=True,
    )
    print(f"Crossing tokens covering whole chunk: {crossing[-10:]}", flush=True)

    if before:
        print("Nearest tokens before/at chunk start:", flush=True)
        print_token_window(tokenizer, input_ids, offsets, text, max(0, before[-1] - 5), 8)
    else:
        print("No tokens end before/at chunk start.", flush=True)

    if after:
        print("Nearest tokens after/at chunk end:", flush=True)
        print_token_window(tokenizer, input_ids, offsets, text, max(0, after[0] - 3), 8)
    else:
        print("No tokens start after/at chunk end.", flush=True)

    around_start = [
        token_idx
        for token_idx, (token_start, token_end) in enumerate(offsets)
        if chunk_start - 200 <= token_start <= chunk_start + 200
        or chunk_start - 200 <= token_end <= chunk_start + 200
    ]
    around_end = [
        token_idx
        for token_idx, (token_start, token_end) in enumerate(offsets)
        if chunk_end - 200 <= token_start <= chunk_end + 200
        or chunk_end - 200 <= token_end <= chunk_end + 200
    ]
    print(f"Token indices with offsets near chunk start: {around_start[:40]}", flush=True)
    print(f"Token indices with offsets near chunk end: {around_end[:40]}", flush=True)


def token_indices_overlapping_span(offsets: List[Tuple[int, int]], start: int, end: int) -> List[int]:
    return [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > start and token_start < end
    ]


def last_token_before_span(offsets: List[Tuple[int, int]], start: int) -> Optional[int]:
    previous = [index for index, (_, token_end) in enumerate(offsets) if token_end <= start]
    if previous:
        return previous[-1]
    return None


def extract_problem_residuals(
    model,
    problem_dir: Path,
    problem_id: str,
    chunks_to_include: Optional[List[int]] = None,
) -> Dict[str, Dict[int, Dict[int, Any]]]:
    base_solution, chunks = load_problem(problem_dir)
    prompt = base_solution["prompt"]
    full_text = base_solution.get("full_cot") or f"{prompt}{base_solution['solution']}"
    spans = find_chunk_char_spans(full_text, prompt, chunks)
    input_ids, offsets = tokenize_with_offsets(model, full_text)

    # with torch.inference_mode():
    #     _, cache = model.run_with_cache(
    #         input_ids,
    #         names_filter=lambda name: name.endswith("hook_resid_post"),
    #     )
    # sum1 = 0
    # sum2 = 0
    # for f in offsets:
    #     sum1 += f[1] - f[0] + 1
    #     sum2 += max(f[1], f[0]) - min(f[1], f[0]) + 1
    # print(f"Sum of offset lengths: {sum1}, sum of max-min lengths: {sum2}, difference: {sum2 - sum1}")

    covered = set()
    for s, e in offsets:
        covered.update(range(s, e))  # no +1

    missing = [i for i in range(offsets[0][0], offsets[-1][1]) if i not in covered]

    print("covered chars:", len(covered))
    print("span length:", offsets[-1][1] - offsets[0][0])
    print("missing chars:", len(missing))
    print("first missing:", missing[:20])
    
    problem_results = {
        "mean_vector": {},
        "last_token_vector": {},
    }
    allowed_chunks = set(chunks_to_include) if chunks_to_include is not None else None
    print(f"Offset min max {offsets[0][0], offsets[-1][-1]}")
    print(offsets[3139:3150])
    print(f"Full text length {len(full_text)}")
    print(f"Token count {input_ids.shape[-1]}")
    print_token_window(model.tokenizer, input_ids, offsets, full_text, 3139, 16)
    print_offset_anomalies(model.tokenizer, input_ids, offsets, full_text, limit=20)
    for chunk_idx, (chunk_start, chunk_end) in enumerate(spans):
        if allowed_chunks is not None and chunk_idx not in allowed_chunks:
            continue

        chunk_token_indices = token_indices_overlapping_span(offsets, chunk_start, chunk_end)
        
        previous_token_idx = last_token_before_span(offsets, chunk_start)
        if chunk_idx % 12 == 0:
            print(chunk_idx)
            print(chunk_token_indices)
            print(previous_token_idx)
        if not chunk_token_indices:
            print(f"Warning: no token indices found for problem {problem_id}, chunk {chunk_idx}", flush=True)
            print_chunk_alignment_debug(
                model.tokenizer,
                input_ids,
                offsets,
                full_text,
                chunk_idx,
                chunk_start,
                chunk_end,
            )
            break
        if previous_token_idx is None:
            print(f"Warning: no previous token for problem {problem_id}, chunk {chunk_idx}", flush=True)
            continue

        problem_results["mean_vector"][chunk_idx] = {}
        problem_results["last_token_vector"][chunk_idx] = {}

        # for layer in range(model.cfg.n_layers):
        #     resid = cache[utilities.get_act_name("resid_post", layer=layer)][0]
        #     problem_results["mean_vector"][chunk_idx][layer] = resid[chunk_token_indices, :].mean(dim=0)
        #     problem_results["last_token_vector"][chunk_idx][layer] = resid[previous_token_idx, :]
        # del resid

    # del cache

    return problem_results


def merge_problem_result(
    output: Dict[str, Dict[str, Dict[str, Dict[int, Any]]]],
    problem_id: str,
    problem_result: Dict[str, Dict[int, Dict[int, Any]]],
) -> None:
    for exp_id, chunks in problem_result.items():
        output.setdefault(exp_id, {})
        output[exp_id][problem_id] = chunks

def load_existing_output(path: Path, overwrite: bool) -> Dict[str, Dict[str, Dict[str, Dict[int, Any]]]]:
    if overwrite or not path.is_file():
        return {"mean_vector": {}, "last_token_vector": {}}
    with open(path, "r", encoding="utf-8") as f:
        existing = json.load(f)
    if isinstance(existing, list):
        merged = {"mean_vector": {}, "last_token_vector": {}}
        for entry in existing:
            if isinstance(entry, dict):
                for exp_id, problems in entry.items():
                    merged.setdefault(exp_id, {}).update(problems)
        return merged
    if isinstance(existing, dict):
        return existing
    return {"mean_vector": {}, "last_token_vector": {}}


def write_output(path: Path, output: Dict[str, Dict[str, Dict[str, Dict[int, Any]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{exp_id: problems} for exp_id, problems in output.items()]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract residual stream vectors for chunk-level linear probes.")
    parser.add_argument("--preset", choices=PRESETS.keys(), default="vm", help="Use local or vast.ai VM defaults.")
    parser.add_argument("-m", "--model", type=str, default=None, help="Rollout model/deployment path in the data directory.")
    parser.add_argument("--hf_model", type=str, default=None, help="Hugging Face model id to load.")
    parser.add_argument("-b", "--base_solution_type", type=str, default="correct", choices=["correct", "incorrect"])
    parser.add_argument("-r", "--rollout_type", type=str, default="default", choices=["default", "forced_answer"])
    parser.add_argument("-sp", "--specific_problems", type=str, default=None, help="Comma-separated problem IDs to include.")
    parser.add_argument("-t", "--temperature", type=float, default=0.6)
    parser.add_argument("-tp", "--top_p", type=float, default=0.95)
    parser.add_argument("-cid", "--chunk_input_dir", type=Path, default=None, help="Root directory to read chunk outputs from.")
    parser.add_argument("--input_dir", type=Path, default=None, help="Full resolved input directory. Overrides -cid/-m/-t/-tp.")
    parser.add_argument("-is", "--input_exp_suffix", type=str, default=None)
    parser.add_argument("--chunks_to_include", type=Path, default=SCRIPT_DIR / "input_args" / "chunks_to_include.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Ignore an existing residual_stream_extracts.json.")
    parser.add_argument("--device", type=str, default=None, help="Device for local model loading, e.g. cpu or cuda.")
    parser.add_argument("--device_map", type=str, default=None, help="Device map for VM/GPU loading, e.g. auto.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from transformers import AutoTokenizer

    input_dir = args.input_dir or build_chunks_input_dir(args)
    input_dir = Path(input_dir)
    problem_ids = parse_problem_ids(args.specific_problems) or discover_problem_ids(input_dir)
    chunks_filtering_dict = load_chunks_to_include(args.chunks_to_include)

    hf_model = args.hf_model or PRESETS[args.preset]["hf_model"] #"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"#args.hf_model or PRESETS[args.preset]["hf_model"]
    tokenizer_kwargs = {
        "trust_remote_code": args.trust_remote_code,
    }

    hf_token = args.hf_token or os.getenv("HF_KEY") or os.getenv("HF_TOKEN")
    if hf_token:
        tokenizer_kwargs["token"] = hf_token

    print(f"Loading tokenizer {hf_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_model, **tokenizer_kwargs)
    model = SimpleNamespace(tokenizer=tokenizer, cfg=SimpleNamespace(device="cpu"))
    print(f"Tokenizer class: {tokenizer.__class__.__name__}", flush=True)
    print(f"Fast tokenizer: {getattr(tokenizer, 'is_fast', None)}", flush=True)
    print(f"Model max length: {getattr(tokenizer, 'model_max_length', None)}", flush=True)

    start_time = time.time()
    for problem_index, problem_id in enumerate(problem_ids, start=1):
        problem_dir = input_dir / f"problem_{problem_id}"
        if not problem_dir.is_dir():
            print(f"Skipping missing problem directory: {problem_dir}", flush=True)
            continue

        print(f"[{problem_index}/{len(problem_ids)}] Debugging tokenizer offsets for problem_{problem_id}", flush=True)
        problem_start = time.time()
        extract_problem_residuals(
            model,
            problem_dir,
            problem_id,
            chunks_to_include=chunks_filtering_dict.get(problem_id) if chunks_filtering_dict else None,
        )
        print(f"Finished problem_{problem_id} in {time.time() - problem_start:.2f}s", flush=True)

    print(f"Total time: {time.time() - start_time:.2f}s", flush=True)


if __name__ == "__main__":
    main()
