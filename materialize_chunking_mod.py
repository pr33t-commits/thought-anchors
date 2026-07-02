import argparse
from pathlib import Path

from chunking_mod import materialize_merged_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize merged-by-function-tag chunks into *_mod artifacts."
    )
    parser.add_argument(
        "--problem_dir",
        type=Path,
        required=True,
        help="Problem directory containing chunks.json and chunks_labeled.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = materialize_merged_chunks(args.problem_dir)
    print(f"Wrote merged chunks to: {result['chunks_mod_file']}")
    print(
        f"Wrote merged labeled chunks to: {result['labeled_chunks_mod_file']}"
    )
    print(f"Created {len(result['merged_chunks'])} chunk_*_mod directories")


if __name__ == "__main__":
    main()
