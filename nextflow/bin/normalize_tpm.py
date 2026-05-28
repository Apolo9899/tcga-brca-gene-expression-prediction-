#!/usr/bin/env python3
"""
Normalize TCGA-BRCA raw RNA-seq counts to TPM.

Steps:
  1. Parse gene lengths from GRCh38 GTF annotation.
  2. Aggregate per-sample count files into a gene x sample matrix.
  3. Compute TPM: counts / gene_length_kb → RPK; RPK / sum(RPK) * 1e6.

Output:
  --out-counts : raw count matrix  (genes x samples, TSV)
  --out-tpm    : TPM matrix        (genes x samples, TSV)
  --out-lengths: gene length table (gene_symbol, length_kb, TSV)
"""
import argparse, gzip, re
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_gene_lengths(gtf_path: Path) -> pd.Series:
    """Return Series {gene_symbol: length_kb} from GRCh38 GTF."""
    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    exon_lengths: dict[str, int] = {}

    with opener(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) < 9 or fields[2] != "exon":
                continue
            start, end = int(fields[3]), int(fields[4])
            attr = fields[8]
            m = re.search(r'gene_name "([^"]+)"', attr)
            if not m:
                continue
            gene = m.group(1)
            exon_lengths[gene] = exon_lengths.get(gene, 0) + (end - start + 1)

    lengths_kb = pd.Series({g: l / 1000.0 for g, l in exon_lengths.items()},
                           name="length_kb")
    return lengths_kb


def load_count_files(rnaseq_dir: Path) -> pd.DataFrame:
    """Merge per-sample TSV count files into a gene x sample DataFrame."""
    files = sorted(rnaseq_dir.glob("*.tsv"))
    if not files:
        raise FileNotFoundError(f"No .tsv files found in {rnaseq_dir}")

    frames = []
    for f in tqdm(files, desc="Loading count files"):
        df = pd.read_csv(f, sep="\t", index_col=0, comment="#")
        # GDC files: column is 'unstranded' or similar; keep first numeric col
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            continue
        sample_id = f.stem.split(".")[0]
        frames.append(df[numeric_cols[0]].rename(sample_id))

    counts = pd.concat(frames, axis=1).fillna(0).astype(int)
    return counts


def counts_to_tpm(counts: pd.DataFrame, lengths_kb: pd.Series) -> pd.DataFrame:
    """Convert raw counts to TPM using gene lengths in kilobases."""
    common = counts.index.intersection(lengths_kb.index)
    counts  = counts.loc[common]
    lengths = lengths_kb.loc[common]

    rpk = counts.div(lengths, axis=0)            # reads per kilobase
    tpm = rpk.div(rpk.sum(axis=0), axis=1) * 1e6
    return tpm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rnaseq-dir",  required=True, type=Path)
    ap.add_argument("--gtf",         required=True, type=Path)
    ap.add_argument("--out-counts",  required=True, type=Path)
    ap.add_argument("--out-tpm",     required=True, type=Path)
    ap.add_argument("--out-lengths", required=True, type=Path)
    args = ap.parse_args()

    print("Parsing gene lengths from GTF…")
    lengths = parse_gene_lengths(args.gtf)
    lengths.to_csv(args.out_lengths, sep="\t", header=True)
    print(f"  {len(lengths):,} genes with annotated exon length.")

    print("Loading count files…")
    counts = load_count_files(args.rnaseq_dir)
    counts.to_csv(args.out_counts, sep="\t")
    print(f"  Count matrix: {counts.shape[0]:,} genes × {counts.shape[1]:,} samples.")

    print("Converting to TPM…")
    tpm = counts_to_tpm(counts, lengths)
    tpm.to_csv(args.out_tpm, sep="\t")
    print(f"  TPM matrix saved → {args.out_tpm}")


if __name__ == "__main__":
    main()
