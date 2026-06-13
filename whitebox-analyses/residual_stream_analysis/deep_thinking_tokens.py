import argparse
import json
import os
from pathlib import Path

print(Path(__file__))
print(Path(__file__).resolve().parent)

from typing import Any, Dict, List, Optional
from utils import jensenshannon_pt
from dotenv import load_dotenv
import time


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]



load_dotenv(PROJECT_ROOT / ".env")


PRESETS = {
    "local": {
        "chunk_input_dir": PROJECT_ROOT / "math_rollouts",
        "model": "accounts/pdlodaya-l7vcn0oxxgj/deployments/e2zlc2rg",
        "hf_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "device": "cpu",
        "device_map": None,
    },
    "vm": {
        "chunk_input_dir": Path("/home/lodaya_dimpal/storage/math_rollouts"),
        "model": "deepseek-r1-distill-llama-8b",
        "hf_model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "device": None,
        "device_map": "cuda",
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


def analyze_residual_stream(model, chunk_info, batch_size, output_dir):
    import torch
    # from scipy.spatial.distance import jensenshannon
    from transformer_lens import utils

    chunk_inputs = {
        problem: {
            chunk_id: {
                "chunk_input": chunk_dict["chunk_input"],
                "chunk": chunk_dict["chunk"],
            }
            for chunk_id, chunk_dict in chunks.items()
        }
        for problem, chunks in chunk_info.items()
    }

    flat_inputs = []
    input_metadata = []
    for problem_id, chunks_dict in chunk_inputs.items():
        flat_inputs.append(chunks_dict[len(chunks_dict) - 1]["chunk_input"])
        input_metadata.append((problem_id, list(chunks_dict.keys())))

    cache_results = {problem_id: {} for problem_id in chunk_inputs.keys()}
    token_lengths = {
        problem_id: {
            chunk_id: {
                "chunk": model.to_tokens(chunk["chunk"]).shape[-1],
                "chunk_input": model.to_tokens(chunk["chunk_input"]).shape[-1],
            }
            for chunk_id, chunk in chunks.items()
        }
        for problem_id, chunks in chunk_inputs.items()
    }

    for batch_start in range(0, len(flat_inputs), batch_size):
        batch_end = min(batch_start + batch_size, len(flat_inputs))
        batch_inputs = flat_inputs[batch_start:batch_end]
        batch_meta = input_metadata[batch_start:batch_end]

        print(f"Processing batch {batch_start // batch_size + 1} ({len(batch_inputs)} items)", flush=True)
        batch_time = time.time()
        with torch.inference_mode():
            # print(batch_inputs[0])
            print("Problem ids in batch:", [meta[0] for meta in batch_meta])
            print("input lengths:", [model.to_tokens(input).shape for input in batch_inputs])
            logits, cache = model.run_with_cache(batch_inputs,
                                                names_filter=lambda name: name.endswith("hook_resid_post"))
            print(f"Batch Processing time: {time.time()-batch_time:.2f}s")
            # logit_probs = logits.softmax(dim=-1)
            # del logits
            
            for batch_index, (problem_id, chunk_id_list) in enumerate(batch_meta):
                for chunk_id in chunk_id_list:
                    chunk_start_time = time.time()
                    chunk_token_length = token_lengths[problem_id][chunk_id]["chunk"]
                    chunk_input_length = token_lengths[problem_id][chunk_id]["chunk_input"]
                    token_slice = slice((chunk_input_length - chunk_token_length - 1), chunk_input_length - 1)
                    cache_results[problem_id][chunk_id] = {}
                    final_logit_probs = logits[batch_index, token_slice, :].softmax(dim=-1)
                    
                    for layer in range(model.cfg.n_layers):
                        layer_rs = layer_rs = cache[utils.get_act_name("resid_post", layer=layer)][batch_index, token_slice, :]
                        
                        # FIX 1: Apply final LayerNorm and unembedding correctly                        
                        layer_logits = model.unembed(model.ln_final(layer_rs))
                        
                        layer_logit_probs = torch.softmax(layer_logits, dim=-1)
                        del layer_logits, layer_rs
                        
                        distance = jensenshannon_pt(
                            layer_logit_probs,
                            final_logit_probs,
                            base=2,
                            dim=-1
                        )
                        
                        cache_results[problem_id][chunk_id][layer] = distance.detach().cpu().float()
                        del distance, layer_logit_probs
                    if chunk_id % 25 == 0:
                        print(f"Chunk {chunk_id}: {time.time()-chunk_start_time:.2f}s")
                    del final_logit_probs
            del cache, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        with open(output_dir, "w", encoding="utf-8") as f:
                json.dump(to_jsonable(cache_results), f, indent=2)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return cache_results


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


def discover_problem_ids(input_dir: Path) -> List[str]:
    return sorted(
        problem.name.split("_")[-1]
        for problem in input_dir.iterdir()
        if problem.is_dir() and problem.name.startswith("problem_")
    )


def build_problem_chunk_dict(input_dir: Path, problem_ids=None, chunk_ids=None):
    if problem_ids is None and chunk_ids is not None:
        return chunk_ids
    if problem_ids is not None and chunk_ids is None:
        return {problem: None for problem in problem_ids}
    if problem_ids is not None and chunk_ids is not None:
        return {problem: chunk_ids.get(problem, None) for problem in set(problem_ids).union(set(chunk_ids.keys()))}
    return {problem: None for problem in discover_problem_ids(input_dir)}


def load_chunk_outputs(input_dir: Path, problem_ids: List = None, chunk_ids: Dict = None):
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Chunk input directory does not exist: {input_dir}")

    problem_chunk_dict = build_problem_chunk_dict(input_dir, problem_ids, chunk_ids)
    chunk_importance_dict = {}

    for problem, chunks_to_include in problem_chunk_dict.items():
        problem_dir = input_dir / f"problem_{problem}"
        with open(problem_dir / "chunks_labeled.json", "r", encoding="utf-8") as f:
            chunk_importance_output = json.load(f)
        with open(problem_dir / "problem.json", "r", encoding="utf-8") as f:
            problem_dict = json.load(f)

        chunk_importance_dict[problem] = {}
        for chunk_obj in chunk_importance_output:
            chunk_id = chunk_obj.get("chunk_idx")
            if chunks_to_include is not None and chunk_id not in chunks_to_include:
                continue

            with open(problem_dir / f"chunk_{chunk_id}" / "solutions.json", "r", encoding="utf-8") as f:
                chunk_solution = json.load(f)[0]

            chunk_obj["chunk_input"] = " ".join(
                part
                for part in [
                    problem_dict.get("problem"),
                    chunk_solution.get("prefix_without_chunk"),
                    chunk_solution.get("chunk_removed"),
                ]
                if part
            )
            chunk_importance_dict[problem][chunk_id] = chunk_obj
        
       

    return chunk_importance_dict


def build_parser():
    parser = argparse.ArgumentParser(description="Residual stream analysis")
    parser.add_argument("--preset", choices=PRESETS.keys(), default="local", help="Use local or vast.ai VM defaults.")
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
    parser.add_argument("-bs", "--batch_size", type=int, default=1)
    parser.add_argument("--chunks_to_include", type=Path, default=SCRIPT_DIR / "input_args" / "chunks_to_include.json")
    parser.add_argument("--output", type=Path, default=Path("/home/lodaya_dimpal/storage/Results/deep_thinking_tokens_distances.json"))
    parser.add_argument("--device", type=str, default=None, help="Device for local model loading, e.g. cpu or cuda.")
    parser.add_argument("--device_map", type=str, default=None, help="Device map for VM/GPU loading, e.g. auto.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()

    
    start = time.time()

    from transformer_lens.model_bridge import TransformerBridge

    chunks_input_dir = args.input_dir or build_chunks_input_dir(args)
    specific_problems = parse_problem_ids(args.specific_problems)
    chunks_filtering_dict = load_chunks_to_include(args.chunks_to_include)

    chunk_outputs = load_chunk_outputs(
        Path(chunks_input_dir),
        problem_ids=specific_problems,
        chunk_ids=chunks_filtering_dict,
    )

    hf_model = args.hf_model or PRESETS[args.preset]["hf_model"]
    device = args.device if args.device is not None else PRESETS[args.preset]["device"]
    device_map = args.device_map if args.device_map is not None else PRESETS[args.preset]["device_map"]

    model_kwargs = {
        "dtype": parse_dtype(args.dtype),
        "trust_remote_code": args.trust_remote_code,
    }
    if device_map:
        model_kwargs["device_map"] = device_map
    elif device:
        model_kwargs["device"] = device

    # hf_token = args.hf_token or os.getenv("HF_KEY") or os.getenv("HF_TOKEN")
    # if hf_token:
    #     model_kwargs["token"] = hf_token

    print(f"Loading model {hf_model}", flush=True)
    bridge = TransformerBridge.boot_transformers(hf_model, **model_kwargs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    importance_results = analyze_residual_stream(bridge, chunk_outputs, args.batch_size, output_dir = args.output)

    # args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(importance_results), f, indent=2)

    print(f"Saved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()



# pip install -U bitsandbytes>=0.46.1