import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def get_primary_function_tag(chunk: Dict[str, Any]) -> str:
    function_tags = chunk.get("function_tags", [])
    if isinstance(function_tags, list) and function_tags:
        tag = str(function_tags[0]).strip()
        return tag or "unknown"
    if isinstance(function_tags, str) and function_tags.strip():
        return function_tags.strip()
    return "unknown"


def merge_consecutive_labeled_chunks(
    labeled_chunks: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    sorted_chunks = sorted(
        labeled_chunks, key=lambda chunk: int(chunk.get("chunk_idx", 0))
    )

    merged_chunks: List[Dict[str, Any]] = []
    old_to_new_idx: Dict[int, int] = {}
    current_group: List[Dict[str, Any]] = []
    current_tag: Optional[str] = None

    def flush_group() -> None:
        nonlocal current_group, current_tag
        if not current_group:
            return

        merged_idx = len(merged_chunks)
        original_chunk_indices = [
            int(chunk.get("chunk_idx", merged_idx)) for chunk in current_group
        ]
        
        merged_text = " ".join(
            chunk.get("chunk", "").strip()
            for chunk in current_group
            if chunk.get("chunk", "").strip()
        ).strip()

        for old_idx in original_chunk_indices:
            old_to_new_idx[old_idx] = merged_idx

        merged_chunks.append(
            {
                "chunk_idx": merged_idx,
                "chunk": merged_text,
                "function_tags": [current_tag or "unknown"],
                "depends_on": [],
                "original_chunk_indices": original_chunk_indices,
                "num_intermediate_chunks": len(original_chunk_indices),
            }
        )
        current_group = []
        current_tag = None

    for chunk in sorted_chunks:
        primary_tag = get_primary_function_tag(chunk)
        if current_group and primary_tag != current_tag:
            flush_group()
        if not current_group:
            current_tag = primary_tag
        current_group.append(chunk)

    flush_group()

    for merged_chunk in merged_chunks:
        merged_idx = int(merged_chunk["chunk_idx"])
        dependency_indices = set()
        for old_idx in merged_chunk["original_chunk_indices"]:
            original_chunk = next(
                (
                    chunk
                    for chunk in sorted_chunks
                    if int(chunk.get("chunk_idx", -1)) == old_idx
                ),
                None,
            )
            if not original_chunk:
                continue
            for dependency in original_chunk.get("depends_on", []):
                try:
                    dependency_idx = int(dependency)
                except (TypeError, ValueError):
                    continue
                mapped_dependency = old_to_new_idx.get(dependency_idx)
                if (
                    mapped_dependency is not None
                    and mapped_dependency != merged_idx
                ):
                    dependency_indices.add(mapped_dependency)

        merged_chunk["depends_on"] = [
            str(idx) for idx in sorted(dependency_indices)
        ]

    return merged_chunks, old_to_new_idx


def resolve_chunk_artifact_paths(
    problem_dir: Path,
) -> Tuple[Path, Path, bool]:
    mod_chunks_file = problem_dir / "chunks_mod.json"
    mod_labeled_chunks_file = problem_dir / "chunks_labeled_mod.json"
    if mod_chunks_file.exists() and mod_labeled_chunks_file.exists():
        return mod_chunks_file, mod_labeled_chunks_file, True

    return problem_dir / "chunks.json", problem_dir / "chunks_labeled.json", False


def resolve_chunk_dir(
    problem_dir: Path, chunk_idx: int, use_mod_chunks: bool = False
) -> Path:
    suffix = "_mod" if use_mod_chunks else ""
    return problem_dir / f"chunk_{chunk_idx}{suffix}"


def get_resampled_merged_chunk(
    rollout_text: str,
    num_intermediate_chunks: int,
    splitter: Callable[[str], List[str]],
) -> str:
    if not rollout_text:
        return ""

    rollout_chunks = splitter(rollout_text)
    if not rollout_chunks:
        return ""

    if num_intermediate_chunks <= 1:
        return rollout_chunks[0]

    return " ".join(rollout_chunks[:num_intermediate_chunks]).strip()


def materialize_merged_chunks(
    problem_dir: Path,
    labeled_chunks: Optional[List[Dict[str, Any]]] = None,
    chunks_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if labeled_chunks is None:
        labeled_chunks_file = problem_dir / "chunks_labeled.json"
        if not labeled_chunks_file.exists():
            raise FileNotFoundError(
                f"Could not find intermediate labels at {labeled_chunks_file}"
            )
        labeled_chunks = load_json(labeled_chunks_file)

    if chunks_payload is None:
        chunks_file = problem_dir / "chunks.json"
        if not chunks_file.exists():
            raise FileNotFoundError(
                f"Could not find chunk source file at {chunks_file}"
            )
        chunks_payload = load_json(chunks_file)

    merged_chunks, old_to_new_idx = merge_consecutive_labeled_chunks(
        labeled_chunks
    )

    chunks_mod_payload = {
        "source_text": chunks_payload.get("source_text", ""),
        "solution_text": chunks_payload.get("solution_text", ""),
        "chunks": [chunk["chunk"] for chunk in merged_chunks],
        "merged_chunks": merged_chunks,
        "old_to_new_chunk_idx": {
            str(old_idx): new_idx
            for old_idx, new_idx in sorted(old_to_new_idx.items())
        },
    }

    chunks_mod_file = problem_dir / "chunks_mod.json"
    labeled_chunks_mod_file = problem_dir / "chunks_labeled_mod.json"
    write_json(chunks_mod_file, chunks_mod_payload)
    write_json(labeled_chunks_mod_file, merged_chunks)

    for merged_chunk in merged_chunks:
        chunk_dir = resolve_chunk_dir(
            problem_dir, int(merged_chunk["chunk_idx"]), use_mod_chunks=True
        )
        chunk_dir.mkdir(parents=True, exist_ok=True)
        write_json(chunk_dir / "chunk.json", merged_chunk)

    return {
        "chunks_mod_file": chunks_mod_file,
        "labeled_chunks_mod_file": labeled_chunks_mod_file,
        "merged_chunks": merged_chunks,
    }
