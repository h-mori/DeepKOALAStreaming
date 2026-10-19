#!/usr/bin/env python3
"""Streaming, length-aware DeepKOALA inference for very large FASTA files.

This runner reuses the installed DeepKOALA model, weights, KO configuration,
and thresholds.  Unlike the upstream CLI, it never loads the whole FASTA or
the whole result table into memory.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import deepkoala
    from deepkoala.model import GRUClassifier
    from deepkoala.utils import find_latest_date, load_ko_config, resolve_device
except ImportError as exc:
    raise SystemExit(
        "DeepKOALA is not importable in this Python environment. Activate the "
        "environment in which `python -m deepkoala.cli` works, then rerun."
    ) from exc


AA_VOCAB = {
    "A": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "K": 10,
    "L": 11,
    "M": 12,
    "N": 13,
    "P": 14,
    "Q": 15,
    "R": 16,
    "S": 17,
    "T": 18,
    "V": 19,
    "W": 20,
    "Y": 21,
}

AA_LOOKUP = np.ones(256, dtype=np.uint8)
for _aa, _token in AA_VOCAB.items():
    AA_LOOKUP[ord(_aa)] = _token

CHECKPOINT_VERSION = 1


@dataclass(slots=True)
class ProteinRecord:
    local_index: int
    name: str
    sequence: bytes

    @property
    def length(self) -> int:
        return len(self.sequence)


class FastaReader:
    """Binary FASTA reader whose next-record offset can be checkpointed."""

    def __init__(self, path: Path, offset: int = 0):
        self.path = path
        self.handle: BinaryIO = path.open("rb")
        self.handle.seek(offset)
        self.pending_header: bytes | None = None
        self.pending_offset: int | None = None

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "FastaReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def next_offset(self) -> int:
        if self.pending_offset is not None:
            return self.pending_offset
        return self.handle.tell()

    def _first_header(self) -> tuple[bytes, int] | None:
        if self.pending_header is not None:
            header = self.pending_header
            offset = self.pending_offset
            self.pending_header = None
            self.pending_offset = None
            assert offset is not None
            return header, offset

        while True:
            offset = self.handle.tell()
            line = self.handle.readline()
            if not line:
                return None
            if line.startswith(b">"):
                return line, offset
            if line.strip():
                raise ValueError(
                    f"Non-FASTA content before the next header at byte {offset:,}"
                )

    def next_record(self, local_index: int) -> ProteinRecord | None:
        first = self._first_header()
        if first is None:
            return None

        header, header_offset = first
        header_text = header[1:].strip().decode("utf-8", errors="replace")
        if not header_text:
            raise ValueError(f"Empty FASTA header at byte {header_offset:,}")
        name = header_text.split(maxsplit=1)[0]

        sequence_lines: list[bytes] = []
        while True:
            offset = self.handle.tell()
            line = self.handle.readline()
            if not line:
                break
            if line.startswith(b">"):
                self.pending_header = line
                self.pending_offset = offset
                break
            sequence_lines.append(b"".join(line.split()))

        sequence = b"".join(sequence_lines)
        if not sequence:
            raise ValueError(
                f"Empty sequence for {name!r} at byte {header_offset:,}"
            )
        return ProteinRecord(local_index, name, sequence)

    def read_part(self, max_records: int) -> list[ProteinRecord]:
        records: list[ProteinRecord] = []
        for local_index in range(max_records):
            record = self.next_record(local_index)
            if record is None:
                break
            records.append(record)
        return records


def dynamic_batches(
    records: Sequence[ProteinRecord],
    max_batch_size: int,
    max_padded_residues: int,
) -> Iterator[list[ProteinRecord]]:
    """Group similarly sized proteins without exceeding padding limits."""

    ordered = sorted(records, key=lambda record: record.length)
    batch: list[ProteinRecord] = []
    batch_max_length = 0

    for record in ordered:
        candidate_max = max(batch_max_length, record.length)
        candidate_size = len(batch) + 1
        exceeds = (
            candidate_size > max_batch_size
            or candidate_max * candidate_size > max_padded_residues
        )
        if batch and exceeds:
            yield batch
            batch = []
            batch_max_length = 0

        batch.append(record)
        batch_max_length = max(batch_max_length, record.length)

    if batch:
        yield batch


def encode_batch(batch: Sequence[ProteinRecord]) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(record.length for record in batch)
    encoded = np.zeros((len(batch), max_length), dtype=np.uint8)
    lengths = np.empty(len(batch), dtype=np.int64)

    for row, record in enumerate(batch):
        raw = np.frombuffer(record.sequence, dtype=np.uint8)
        encoded[row, : record.length] = AA_LOOKUP[raw]
        # DeepKOALA's GRU output is indexed from zero, as in its data.py.
        lengths[row] = record.length - 1

    return torch.from_numpy(encoded), torch.from_numpy(lengths)


def resources_directory() -> Path:
    package_dir = Path(deepkoala.__file__).resolve().parent
    resources_dir = package_dir.parent / "resources"
    if not resources_dir.is_dir():
        raise FileNotFoundError(
            f"DeepKOALA resources directory was not found: {resources_dir}"
        )
    return resources_dir


def load_model_and_config(
    model_name: str,
    date: str,
    device: torch.device,
):
    resources_dir = resources_directory()
    db_date = find_latest_date(date, str(resources_dir))
    ko_cfg = resources_dir / db_date / f"ko_config_{model_name}.json"
    weights = resources_dir / db_date / f"weights_{model_name}.pt"

    ko2idx, idx2ko, thresholds = load_ko_config(str(ko_cfg))
    network = GRUClassifier(128, 2, len(ko2idx)).to(device)
    checkpoint = torch.load(str(weights), map_location=device)
    network.load_state_dict(checkpoint)
    network.eval()
    return network, idx2ko, thresholds, db_date


def infer_part(
    records: Sequence[ProteinRecord],
    network: torch.nn.Module,
    idx2ko: dict[int, str],
    thresholds: dict[str, float],
    device: torch.device,
    topk: int,
    detail: bool,
    max_batch_size: int,
    max_padded_residues: int,
    max_sequence_length: int,
) -> tuple[list[list[tuple]], int, list[ProteinRecord]]:
    results: list[list[tuple] | None] = [None] * len(records)
    annotated_count = 0

    if max_sequence_length > 0:
        oversized = [
            record for record in records if record.length > max_sequence_length
        ]
        inference_records = [
            record for record in records if record.length <= max_sequence_length
        ]
    else:
        oversized = []
        inference_records = list(records)

    # Oversized proteins are retained in the main output as unannotated so the
    # output remains one-to-one with the input. Their names, lengths, and skip
    # reason are also written to a separate audit file by main().
    for record in oversized:
        if detail:
            results[record.local_index] = [(record.name, "", "", "", "")]
        else:
            results[record.local_index] = [(record.name, "")]

    batches = dynamic_batches(
        inference_records,
        max_batch_size,
        max_padded_residues,
    )
    progress = tqdm(
        total=len(records),
        desc="Inference",
        unit="seq",
        dynamic_ncols=True,
    )
    progress.update(len(oversized))

    with torch.inference_mode():
        for batch in batches:
            encoded_cpu, lengths_cpu = encode_batch(batch)
            encoded = encoded_cpu.to(device=device, dtype=torch.long)
            lengths = lengths_cpu.to(device=device)

            logits = network(encoded, lengths)
            probabilities = F.softmax(logits, dim=1)
            k = min(topk, probabilities.size(1))
            top_prob, top_idx = torch.topk(probabilities, k=k, dim=1)
            top_prob_cpu = top_prob.cpu().tolist()
            top_idx_cpu = top_idx.cpu().tolist()

            for record, idx_row, prob_row in zip(
                batch, top_idx_cpu, top_prob_cpu
            ):
                rows: list[tuple] = []
                sequence_annotated = False
                for pred_idx, pred_prob in zip(idx_row, prob_row):
                    prediction = idx2ko[pred_idx]
                    threshold = thresholds[prediction]
                    mark = "*" if pred_prob >= threshold else ""
                    sequence_annotated |= bool(mark)
                    if detail:
                        rows.append(
                            (
                                record.name,
                                prediction,
                                f"{pred_prob:.4f}",
                                f"{threshold:.4f}",
                                mark,
                            )
                        )
                    else:
                        rows.append((record.name, prediction if mark else ""))
                results[record.local_index] = rows
                if sequence_annotated:
                    annotated_count += 1

            progress.update(len(batch))
            del encoded, lengths, logits, probabilities, top_prob, top_idx

    progress.close()
    if any(rows is None for rows in results):
        raise RuntimeError("Internal error: at least one sequence has no result")
    return (
        [rows for rows in results if rows is not None],
        annotated_count,
        oversized,
    )


def open_output(path: Path, compression: str):
    if compression == "gzip":
        return gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=1)
    return path.open("w", encoding="utf-8", newline="")


def write_part_atomic(
    output_path: Path,
    results: Sequence[Sequence[tuple]],
    detail: bool,
    compression: str,
) -> None:
    temp_path = output_path.with_name(output_path.name + ".tmp")
    with open_output(temp_path, compression) as output_handle:
        writer = csv.writer(output_handle, lineterminator="\n")
        if detail:
            writer.writerow(
                ["name", "predict_label", "probability", "threshold", "annotate"]
            )
        else:
            writer.writerow(["name", "predict_label"])
        for sequence_rows in results:
            writer.writerows(sequence_rows)
    os.replace(temp_path, output_path)


def write_skipped_atomic(
    output_path: Path,
    records: Sequence[ProteinRecord],
    max_sequence_length: int,
    compression: str,
) -> None:
    temp_path = output_path.with_name(output_path.name + ".tmp")
    with open_output(temp_path, compression) as output_handle:
        writer = csv.writer(output_handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["name", "length", "reason"])
        reason = f"length_exceeds_{max_sequence_length}_aa"
        for record in records:
            writer.writerow([record.name, record.length, reason])
    os.replace(temp_path, output_path)


def write_json_atomic(path: Path, content: dict) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp_path, path)


def input_identity(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def semantic_config(args: argparse.Namespace, resolved_date: str) -> dict:
    return {
        "model": args.model,
        "date": resolved_date,
        "topk": args.topk,
        "detail": args.detail,
        "compression": args.compression,
        "max_sequence_length": args.max_sequence_length,
    }


def initial_state(args: argparse.Namespace, resolved_date: str) -> dict:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "input": input_identity(args.input),
        "config": semantic_config(args, resolved_date),
        "next_part": 0,
        "next_offset": 0,
        "sequences_processed": 0,
        "annotated_sequences": 0,
        "skipped_sequences": 0,
        "completed": False,
    }


def load_or_create_state(
    checkpoint_path: Path,
    args: argparse.Namespace,
    resolved_date: str,
) -> dict:
    if not checkpoint_path.exists():
        return initial_state(args, resolved_date)

    with checkpoint_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)

    if state.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint version")
    if state.get("input") != input_identity(args.input):
        raise ValueError(
            "Input FASTA path, size, or modification time differs from the checkpoint"
        )
    # Checkpoints written by v1 did not contain this safety limit. Recording
    # the current value here permits an in-place resume after an oversized
    # sequence exposed the need for the guard.
    if "max_sequence_length" not in state.get("config", {}):
        state["config"]["max_sequence_length"] = args.max_sequence_length
    if state.get("config") != semantic_config(args, resolved_date):
        raise ValueError(
            "Model/output options differ from the existing checkpoint. "
            "Use the original options or a new output directory."
        )
    state.setdefault("skipped_sequences", 0)
    return state


def validate_output_parts(
    output_dir: Path,
    state: dict,
    compression: str,
    checkpoint_existed: bool,
) -> None:
    suffix = ".csv.gz" if compression == "gzip" else ".csv"
    pattern = re.compile(r"^part_(\d{6})\.csv(?:\.gz)?$")
    existing: dict[int, Path] = {}
    for path in output_dir.iterdir():
        match = pattern.match(path.name)
        if match:
            existing[int(match.group(1))] = path

    next_part = int(state["next_part"])
    if not checkpoint_existed and existing:
        raise ValueError(
            "Output part files exist but the checkpoint records no completed "
            "parts. Use a new output directory or restore its checkpoint."
        )

    for part_number in range(next_part):
        expected = output_dir / f"part_{part_number:06d}{suffix}"
        if not expected.is_file():
            raise FileNotFoundError(
                f"Checkpoint expects a completed output that is missing: {expected}"
            )

    # A single part at next_part can remain if the process stopped after the
    # atomic output rename but before checkpoint replacement; it is recomputed.
    unexpected = [
        path
        for number, path in existing.items()
        if number > next_part
    ]
    if unexpected:
        names = ", ".join(path.name for path in sorted(unexpected)[:5])
        raise ValueError(f"Unexpected output parts beyond the checkpoint: {names}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run DeepKOALA on a huge FASTA with streaming input, length-aware "
            "dynamic batches, atomic part files, and checkpointed resume."
        )
    )
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["full", "frag"], default="frag")
    parser.add_argument("--date", default="latest")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--detail", action="store_true")
    parser.add_argument(
        "--part-size",
        type=int,
        default=1_000_000,
        help="Sequences per resumable output part (default: 1,000,000)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=512,
        help="Hard ceiling on sequences per GPU batch (default: 512)",
    )
    parser.add_argument(
        "--max-padded-residues",
        type=int,
        default=65_536,
        help=(
            "Ceiling for batch_size * longest_sequence_length. The default "
            "is conservative for DGX Spark (default: 65,536)."
        ),
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=10_000,
        help=(
            "Do not send longer proteins to the GRU; retain them as "
            "unannotated and record them in skipped_part files. Use 0 to "
            "disable this guard (default: 10,000)."
        ),
    )
    parser.add_argument(
        "--compression",
        choices=["gzip", "none"],
        default="gzip",
        help="Output each CSV part as gzip level 1 or plain text (default: gzip)",
    )
    parser.add_argument(
        "--max-parts",
        type=int,
        default=None,
        help="Stop cleanly after this many new parts; useful for benchmarking",
    )
    args = parser.parse_args()

    if args.topk < 1:
        parser.error("--topk must be >= 1")
    if args.part_size < 1:
        parser.error("--part-size must be >= 1")
    if args.max_batch_size < 1:
        parser.error("--max-batch-size must be >= 1")
    if args.max_padded_residues < 1:
        parser.error("--max-padded-residues must be >= 1")
    if args.max_sequence_length < 0:
        parser.error("--max-sequence-length must be >= 0")
    if args.max_parts is not None and args.max_parts < 1:
        parser.error("--max-parts must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    args.input = args.input.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input FASTA not found: {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    network, idx2ko, thresholds, resolved_date = load_model_and_config(
        args.model, args.date, device
    )

    checkpoint_path = args.output_dir / "checkpoint.json"
    checkpoint_existed = checkpoint_path.exists()
    state = load_or_create_state(checkpoint_path, args, resolved_date)
    validate_output_parts(
        args.output_dir,
        state,
        args.compression,
        checkpoint_existed,
    )
    if state["completed"]:
        print(
            f"Already complete: {state['sequences_processed']:,} sequences, "
            f"{state['annotated_sequences']:,} annotated."
        )
        return 0

    # Persist the input identity and model/output configuration before the
    # first GPU batch so an interrupted first part can be resumed safely.
    write_json_atomic(checkpoint_path, state)

    print(f"Device: {device}")
    print(f"DeepKOALA model/date: {args.model}/{resolved_date}")
    print(
        "Dynamic batch limits: "
        f"{args.max_batch_size:,} sequences, "
        f"{args.max_padded_residues:,} padded residues"
    )
    if args.max_sequence_length > 0:
        print(
            "Maximum GRU sequence length: "
            f"{args.max_sequence_length:,} aa; longer sequences will be skipped"
        )
    if state["next_offset"]:
        print(
            f"Resuming at byte {state['next_offset']:,}; "
            f"{state['sequences_processed']:,} sequences already complete."
        )

    parts_this_run = 0
    with FastaReader(args.input, state["next_offset"]) as reader:
        while True:
            part_number = state["next_part"]
            part_start_offset = reader.next_offset()
            records = reader.read_part(args.part_size)
            if not records:
                state["completed"] = True
                write_json_atomic(checkpoint_path, state)
                break

            max_length = max(record.length for record in records)
            print(
                f"Part {part_number:06d}: {len(records):,} sequences; "
                f"bytes {part_start_offset:,}-{reader.next_offset():,}; "
                f"maximum length {max_length:,} aa"
            )
            started = time.monotonic()
            results, annotated, skipped = infer_part(
                records=records,
                network=network,
                idx2ko=idx2ko,
                thresholds=thresholds,
                device=device,
                topk=args.topk,
                detail=args.detail,
                max_batch_size=args.max_batch_size,
                max_padded_residues=args.max_padded_residues,
                max_sequence_length=args.max_sequence_length,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            suffix = ".csv.gz" if args.compression == "gzip" else ".csv"
            output_path = args.output_dir / f"part_{part_number:06d}{suffix}"
            write_part_atomic(output_path, results, args.detail, args.compression)

            if skipped:
                skipped_suffix = ".tsv.gz" if args.compression == "gzip" else ".tsv"
                skipped_path = (
                    args.output_dir
                    / f"skipped_part_{part_number:06d}{skipped_suffix}"
                )
                write_skipped_atomic(
                    skipped_path,
                    skipped,
                    args.max_sequence_length,
                    args.compression,
                )
                examples = ", ".join(
                    f"{record.name} ({record.length:,} aa)"
                    for record in skipped[:3]
                )
                print(
                    f"Skipped {len(skipped):,} oversized sequence(s): {examples}"
                )

            state["next_part"] = part_number + 1
            state["next_offset"] = reader.next_offset()
            state["sequences_processed"] += len(records)
            state["annotated_sequences"] += annotated
            state["skipped_sequences"] += len(skipped)
            write_json_atomic(checkpoint_path, state)

            elapsed = time.monotonic() - started
            print(
                f"Completed {output_path.name}: {annotated:,} annotated; "
                f"{len(records) / elapsed:,.0f} sequences/s"
            )

            del records, results, skipped
            parts_this_run += 1
            if args.max_parts is not None and parts_this_run >= args.max_parts:
                print("Stopped at --max-parts limit; rerun the same command to resume.")
                return 0

    print(
        f"Finished: {state['sequences_processed']:,} sequences; "
        f"{state['annotated_sequences']:,} annotated; "
        f"{state['skipped_sequences']:,} skipped for excessive length; "
        f"{state['next_part']:,} output parts."
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted. The last completed part remains checkpointed.", file=sys.stderr)
        sys.exit(130)
