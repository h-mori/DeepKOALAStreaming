# DeepKOALA streaming runner

`deepkoala_stream.py` is intended for the 540 GB protein FASTA on the DGX
Spark. It uses the installed DeepKOALA model and changes only the I/O and
batching path:

- reads at most `--part-size` sequences into RAM instead of calling `f.read()`;
- sorts each part by sequence length to avoid excessive GRU padding;
- sets each GPU batch dynamically so that
  `batch size × longest sequence length <= --max-padded-residues`;
- prevents proteins longer than `--max-sequence-length` from entering the GRU;
- writes every part atomically and stores a byte-offset checkpoint;
- resumes with the same command after interruption;
- preserves the original FASTA order in every output part.

## 1. Verify on one million sequences

Activate the Python environment in which the original command
`python -m deepkoala.cli` works. Then run:

```bash
python deepkoala_stream.py \
  -i allprotein.20.faa \
  -o deepkoala_stream_results \
  --model frag \
  --device cuda \
  --max-batch-size 512 \
  --max-padded-residues 65536 \
  --max-sequence-length 10000 \
  --part-size 1000000 \
  --max-parts 1
```

The conservative `65,536` padded-residue limit makes a 9,999-aa sequence use
a batch of at most 6, while a 300-aa sequence can use a batch of up to 218.
It is therefore much faster than forcing all proteins through batch size 8,
without allowing a rare long protein to inflate a large batch.

The first run creates:

```text
deepkoala_stream_results/
  checkpoint.json
  part_000000.csv.gz
```

Check the reported speed and inspect the output:

```bash
zcat deepkoala_stream_results/part_000000.csv.gz | head
```

## 2. Run or resume the complete FASTA

Remove only `--max-parts 1` and keep every other option unchanged:

```bash
python deepkoala_stream.py \
  -i allprotein.20.faa \
  -o deepkoala_stream_results \
  --model frag \
  --device cuda \
  --max-batch-size 512 \
  --max-padded-residues 65536 \
  --max-sequence-length 10000 \
  --part-size 1000000
```

The same command resumes at the byte offset after the last completed part.
An incomplete `.tmp` file is never treated as a completed result. Do not edit,
replace, or recompress the input FASTA while the analysis is in progress;
the runner checks its absolute path, size, and modification time.

## Output format

Without `--detail`, output matches the compact DeepKOALA schema:

```text
name,predict_label
```

Predictions below the KO-specific threshold have an empty `predict_label`.
Proteins exceeding `--max-sequence-length` are also retained in the main CSV
with an empty prediction. Their identifiers, lengths, and skip reasons are
written to `skipped_part_NNNNNN.tsv.gz` for auditing. This guard is independent
of the padded-residue limit: even a batch of one extremely long sequence can
be unsupported by the cuDNN GRU implementation.
Add `--detail` from the first run if probability and threshold columns are
required. Do not change this option after starting an output directory.

The default is gzip level 1 to reduce output space. Use
`--compression none` from the first run if uncompressed CSV is preferred.

## Tuning after the safe test

The padding limit is the principal safety control. The default `65,536` is
below the empirically successful case of `8 × 11,983 = 95,864` residues.
If the first million sequences are stable, benchmark a *new output directory*
with `--max-padded-residues 90000`. Do not raise only `--max-batch-size`,
because the padded-residue limit will remain the active protection against
long proteins.

If a CUDA illegal-memory-access error occurs, the CUDA process cannot safely
continue. Restart the same command; it will redo only the current, incomplete
part. If needed, lower `--max-padded-residues` (for example to `48000`) and
resume in the same output directory. Batch limits and part size may be changed
between invocations because they do not change the scientific output. Model,
date, top-k, detail mode, and compression must remain unchanged.
The maximum sequence length is also stored in the checkpoint after upgrading
from the first version of this runner.

## Combining completed parts

Keeping parts separate is safer at this scale. If a single-header gzip file is
absolutely required after completion, merge it in a streaming pipeline:

```bash
first=1
while IFS= read -r -d '' f; do
  if [ "$first" -eq 1 ]; then
    gzip -cd "$f"
    first=0
  else
    gzip -cd "$f" | tail -n +2
  fi
done < <(find deepkoala_stream_results -maxdepth 1 \
  -name 'part_*.csv.gz' -print0 | sort -z) \
| gzip -1 > allprotein.deepkoala.csv.gz
```

This requires reading and recompressing the entire result, so downstream tools
should consume the ordered part files directly whenever possible.
