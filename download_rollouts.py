import argparse
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")


PRESETS = {
    "local": {
        "output_dir": PROJECT_ROOT / "math_rollouts",
        "path_contains": "problem_1591",
        "model_path_contains": "deepseek-r1-distill-qwen-14b",
    },
    "vm": {
        "output_dir": Path("/mnt/math_rollouts"),
        "path_contains": None,
        "model_path_contains": None,
    },
}


def build_parser():
    parser = argparse.ArgumentParser(description="Download math rollout files.")
    parser.add_argument("--preset", choices=PRESETS.keys(), default="local", help="Use local or vast.ai VM defaults.")
    parser.add_argument("--dataset", default="uzaymacar/math-rollouts", help="Hugging Face dataset name.")
    parser.add_argument("--split", default="default", help="Dataset split to stream.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory where rollout files are written.")
    parser.add_argument("--path-contains", default=None, help="Only keep rows whose path contains this text.")
    parser.add_argument("--model-path-contains", default=None, help="Optional second path filter, usually a model/deployment id.")
    parser.add_argument("--no-default-filters", action="store_true", help="Ignore preset path filters.")
    parser.add_argument("--limit", type=int, default=None, help="Stop after writing this many files.")
    parser.add_argument("--print-every", type=int, default=1000, help="Progress print interval.")
    return parser


def main():
    args = build_parser().parse_args()
    preset = PRESETS[args.preset]

    from datasets import load_dataset
    from tqdm import tqdm

    output_dir = args.output_dir or preset["output_dir"]
    path_contains = None if args.no_default_filters else preset["path_contains"]
    model_path_contains = None if args.no_default_filters else preset["model_path_contains"]
    path_contains = args.path_contains if args.path_contains is not None else path_contains
    model_path_contains = args.model_path_contains if args.model_path_contains is not None else model_path_contains
    filters = [item for item in (path_contains, model_path_contains) if item]

    print("Loading dataset...", flush=True)
    ds = load_dataset(args.dataset, streaming=True, split=args.split)

    if filters:
        print(f"Filtering paths containing: {filters}", flush=True)
        ds = ds.filter(lambda example: all(fragment in example.get("path", "") for fragment in filters))

    print(f"Writing rollout files under {output_dir.resolve()}", flush=True)
    written = 0
    for row in tqdm(ds):
        if args.limit is not None and written >= args.limit:
            break

        row_path = row.get("path")
        content = row.get("content")
        if not row_path or content is None:
            continue

        file_path = output_dir / row_path
        os.makedirs(file_path.parent, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        written += 1
        if written % args.print_every == 0:
            print(file_path, flush=True)
            print(f"Created {written} files", flush=True)

    print(f"Done! Created {written} files.", flush=True)


if __name__ == "__main__":
    main()
