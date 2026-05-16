from transformer_lens.model_bridge import TransformerBridge
from transformer_lens import utils
import os
import argparse, json
from pathlib import Path
from typing import List, Dict
import torch
import pandas as pd, numpy as np
from scipy.spatial.distance import jensenshannon
from dotenv import load_dotenv
load_dotenv()
# HF_KEY = os.getenv("HF_KEY")

parser = argparse.ArgumentParser(
    description="Residual stream analy"
)
parser.add_argument("-m","--model", type=str, default = "accounts/pdlodaya-l7vcn0oxxgj/deployments/e2zlc2rg", help = "Model to use")
parser.add_argument('-b', '--base_solution_type', type=str, default='correct', choices=['correct', 'incorrect'], help = 'Type of base solution to generate')
parser.add_argument('-r', '--rollout_type', type=str, default='default', choices=['default', 'forced_answer'], help = 'Type of rollout to generate')
parser.add_argument('-sp', '--specific_problems', type=str, default=None, help='Comma-separated list of problem IDs to include')
# parser.add_argument('-sc', '--specific_chunks', type=str, default=None, help='Comma-separated list of chunk IDs to include')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for rollout generation')
parser.add_argument('-tp', '--top_p', type=float, default=0.95, help='Top-p sampling parameter')
parser.add_argument('-cid', '--chunk_input_dir', type=str, default='C:\\Users\\Preet Lodaya\\Thought_Anchors\\thought-anchors\\math_rollouts', help = 'Directory to read chunk outputs from')
parser.add_argument('-is', '--input_exp_suffix', type=str, default=None, help='Suffix to add to the input directory')
parser.add_argument('-bs', '--batch_size', type=int, default=2, help='Batch size for local model')
# parser.add_arguement('-cad', '--chunk_analysis_dir', type=str, default=None, help='Directory of saved chunk importance analysis')
args = parser.parse_args()

# Create output directory
chunks_input_dir = Path(args.chunk_input_dir) / args.model.split(":")[0] / f"temperature_{str(args.temperature)}_top_p_{str(args.top_p)}"
if args.rollout_type == 'forced_answer':
    # NOTE: For forced answer rollouts, we use the correct base solution (we copy the files from the correct base solution directory before running this script)
    chunks_input_dir = chunks_input_dir / f"{args.base_solution_type}_base_solution_{args.rollout_type}_{args.input_exp_suffix}" if args.input_exp_suffix else chunks_input_dir / f"{args.base_solution_type}_base_solution_{args.rollout_type}"
else:
    chunks_input_dir = chunks_input_dir / f"{args.base_solution_type}_base_solution_{args.input_exp_suffix}" if args.input_exp_suffix else chunks_input_dir / f"{args.base_solution_type}_base_solution"

# Test basic forward pass
# text = "Hello, how are you?"
# logits = bridge(text, return_type="logits")
# print(f"Logits shape: {logits.shape}")

# # Test text generation
# generated = bridge.generate("Once upon a time", max_new_tokens=50)
# print(f"Generated text: {generated}")

def analyze_residual_stream(model, chunk_info):

    chunk_inputs = {problem:{chunk_id: {"chunk_input":chunk_dict["chunk_input"],
                                        "chunk":chunk_dict["chunk"]} for chunk_id, chunk_dict in chunks.items()} for problem, chunks in chunk_info.items()}
    
    flat_inputs = []
    input_metadata = []
    # chunks_removed = []
    print(chunk_inputs)
    for problem_id, chunks_dict in chunk_inputs.items():
        for chunk_id, chunk_input_plus_chunk in chunks_dict.items():
            flat_inputs.append(chunk_input_plus_chunk['chunk_input'])
            input_metadata.append((problem_id, chunk_id))
            # chunks_removed.append(chunk_input_plus_chunk['chunk'])
    print(flat_inputs)
    print(len(flat_inputs))
    batch_size = args.batch_size
    cache_results = {problem_id: {} for problem_id in chunk_inputs.keys()}
    
    for batch_start in range(0, len(flat_inputs), batch_size):

        batch_end = min(batch_start + batch_size, len(flat_inputs))
        batch_inputs = flat_inputs[batch_start:batch_end]
        batch_meta = input_metadata[batch_start:batch_end]
        # chunks = chunks_removed[batch_start:batch_end]
        
        print(f"Processing batch {batch_start // batch_size + 1} ({len(batch_inputs)} items)")
    
        logits, cache = model.run_with_cache(batch_inputs)
        logit_probs = logits.softmax(dim = -1)
        
        # for h in ['unembed.hook_in', 'unembed.hook_out', 'hook_unembed']:
        #     print(f"Cache for hook {h}: {cache[h].shape}")
        # torch.testing.assert_close(cache['unembed.hook_out'], cache['hook_unembed'], rtol=1e-5, atol=1e-8)
        # print(logits.shape)
        # print(f'check:- {logits.softmax(dim = -1).sum(dim = -1).sum()}')
        W_U = model.W_U
        # print(f"Cache keys: {list(cache.keys())}")
        # torch.testing.assert_close(cache[utils.get_act_name("resid_post", 0)], cache[utils.get_act_name("resid_pre", 1)])
        # Store cache for each item
        for problem_id, chunk_id in batch_meta:
            # final_layer_rs = cache["unembed.hook_out"]
            # model.cfg.n_layers - 1
            chunk_token_length = model.to_tokens(chunk_inputs[problem_id][chunk_id]['chunk']).shape[-1]
            cache_results[problem_id][chunk_id] = {}
            for layer in range(12,14):
                layer_rs = cache[utils.get_act_name("resid_post", layer=layer)]
                layer_logit_probs = torch.softmax(layer_rs @ W_U, dim =-1)
                
                # print(f'check:- {layer_logits.sum(dim = -1).sum()}')
                # print(layer_logit_probs[:,-1,:].float().detach().numpy().shape)
                # print(logit_probs.shape)
                distance = jensenshannon(layer_logit_probs[:,-(chunk_token_length + 1):-1,:].float().detach().numpy(), 
                                        logit_probs[:,-(chunk_token_length + 1):-1,:].float().detach().numpy(),base=2, axis = -1, keepdims=True)#[:,-1]
                print(f"Layer {layer} distance: {distance}")
                # layers.append(layer)
                # distances.append(distance)
                # if layer < model.cfg.n_layers - 2:
                #     layer_logit_next_probs = torch.softmax(cache[utils.get_act_name("resid_post", layer=layer + 1)] @ W_U, dim =-1)
                #     print(jensenshannon(layer_logit_probs[:,-1,:].float().detach().numpy(), 
                #                         layer_logit_next_probs[:,-1,:].float().detach().numpy(),base=2, axis = -1, keepdims=True)[:,-1])
                cache_results[problem_id][chunk_id][layer] = distance
    # cache_results[utils.get_act_name("resid_post")]
    
    return cache_results
    
def load_chunk_outputs(input_dir, problem_ids:List = None, chunk_ids:Dict =None):
    

    if problem_ids is None and chunk_ids is not None:
        problem_chunk_dict = chunk_ids
    elif problem_ids is not None and chunk_ids is None:
        problem_chunk_dict = {problem: None for problem in problem_ids}
    elif problem_ids is not None and chunk_ids is not None:
        problem_chunk_dict = {problem: chunk_ids.get(problem, None) for problem in set(problem_ids.keys()).union(set(chunk_ids.keys()))}
    else:
        problem_chunk_dict = {problem.split('_')[-1]: None for problem in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, problem))}
        raise Warning("Both problem_ids and chunk_ids are None. Loading all chunk outputs for all problems in the input directory.")
    
    chunk_importance_dict = {}
    
    for problem, chunks_to_include in problem_chunk_dict.items():    
        with open(input_dir / f"problem_{problem}/chunks_labeled.json", "r") as f: 
            chunk_importance_output = json.load(f)
        
        chunk_importance_dict[problem] = {}
        if chunks_to_include is not None:

            for chunk_obj in chunk_importance_output:
                if chunk_obj.get("chunk_idx") in chunks_to_include:

                    chunk_id = chunk_obj.get("chunk_idx")
                    with open(input_dir / f"problem_{problem}/chunk_{chunk_id}/solutions.json", "r") as f:
                        chunk_solution = json.load(f)[0]
                    with open(input_dir / f"problem_{problem}/problem.json", "r") as f:
                        problem_dict = json.load(f)
                
                    chunk_obj["chunk_input"] = problem_dict.get("problem") + " " + chunk_solution.get("prefix_without_chunk") + " " + chunk_solution.get("chunk_removed")
                    
                    chunk_importance_dict[problem][chunk_id] = chunk_obj
        else:
            chunk_importance_dict[problem] = {chunk_obj["chunk_idx"]: chunk_obj for chunk_obj in chunk_importance_output}
    return chunk_importance_dict
    
def main():
    
    if args.specific_problems:
        specific_problems = [int(id) for id in args.specific_problems.split(",")]
    else:
        specific_problems = None
    
    with open("input_args/chunks_to_include.json", "r") as f:
        chunks_filtering_dict = json.load(f)
    if len(chunks_filtering_dict) ==0:
        chunks_filtering_dict = None

    # Load chunk outputs
    chunk_outputs = load_chunk_outputs(chunks_input_dir, problem_ids=specific_problems, chunk_ids=chunks_filtering_dict)

    
    # Method 1: Simple loading (recommended)
    bridge = TransformerBridge.boot_transformers(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        # token=HF_KEY,
        device="cpu",  # or "cpu"
        dtype = torch.bfloat16,
        trust_remote_code=False,  # Usually not needed for official models
    )

    import sys
    # Analyze residual stream importance using various methods
    importance_results = analyze_residual_stream(bridge, chunk_outputs)

    # print(importance_results["1923"][3].keys())
    # Save results
    # save_importance_results(importance_results, output_dir)


if __name__ == "__main__":
    main()


# from transformer_lens.model_bridge import TransformerBridge
# import torch 
# from dotenv import load_dotenv   
# load_dotenv()
# HF_KEY = os.getenv("HF_KEY")
# # Method 1: Simple loading (recommended)
# bridge = TransformerBridge.boot_transformers(
#     "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
#     # token=HF_KEY,
#     device="cpu",  # or "cpu"
#     dtype = torch.bfloat16,
#     trust_remote_code=False,  # Usually not needed for official models
# )
# bridge.W_U.shape

# from scipy.spatial.distance import jensenshannon
# import numpy as np
# jensenshannon(np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))

# np.array([[0.0, 1.0, 0.0]]).shape