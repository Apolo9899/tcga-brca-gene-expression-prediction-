#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// ─── Parameters ──────────────────────────────────────────────────────────────
params.wsi_manifest   = "$projectDir/../data/gdc_manifest_wsi.txt"
params.rnaseq_manifest = "$projectDir/../data/gdc_manifest_rnaseq.txt"
params.gtf            = "$projectDir/../data/gencode.v38.annotation.gtf.gz"
params.outdir         = "$projectDir/../data/processed"
params.tile_size      = 256
params.magnification  = 20
params.luminance_thr  = 210
params.max_tiles      = 500
params.backbones      = "dinov2,dinov3,phikon,uni,virchow2,gigapath"
params.token_hf       = ""   // HuggingFace token for gated models

// ─── Workflow ─────────────────────────────────────────────────────────────────
workflow {
    // Stage 1 — Download WSI and RNA-seq from GDC
    wsi_files   = DOWNLOAD_WSI(params.wsi_manifest)
    rnaseq_files = DOWNLOAD_RNASEQ(params.rnaseq_manifest)

    // Stage 2 — Normalize RNA-seq counts to TPM
    tpm_matrix = NORMALIZE_TPM(rnaseq_files.collect(), params.gtf)

    // Stage 3 — Tile WSIs
    tiles = TILE_WSI(
        wsi_files.flatten(),
        params.tile_size,
        params.luminance_thr,
        params.max_tiles
    )

    // Stage 4 — Precompute embeddings for each backbone
    backbones_ch = Channel.of(params.backbones.split(",")).flatten()
    PRECOMPUTE_EMBEDDINGS(backbones_ch, tiles.collect(), params.token_hf)
}

// ─── Process 1: Download WSI files from GDC ──────────────────────────────────
process DOWNLOAD_WSI {
    label 'download'
    publishDir "${params.outdir}/wsi_raw", mode: 'copy'

    input:
    path manifest

    output:
    path "*.svs"

    script:
    """
    gdc-client download \\
        --manifest ${manifest} \\
        --dir . \\
        --n-processes 4 \\
        --retry-amount 3

    # Verify MD5 checksums
    python3 -c "
import hashlib, csv, sys, pathlib
ok = True
with open('${manifest}') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        fpath = pathlib.Path(row['filename'])
        if not fpath.exists():
            print(f'MISSING: {fpath}'); ok = False; continue
        md5 = hashlib.md5(fpath.read_bytes()).hexdigest()
        if md5 != row['md5']:
            print(f'MD5 MISMATCH: {fpath}'); ok = False
if not ok:
    sys.exit(1)
print('All MD5 checksums verified.')
"
    """
}

// ─── Process 2: Download RNA-seq count files from GDC ────────────────────────
process DOWNLOAD_RNASEQ {
    label 'download'
    publishDir "${params.outdir}/rnaseq_raw", mode: 'copy'

    input:
    path manifest

    output:
    path "*.tsv"

    script:
    """
    gdc-client download \\
        --manifest ${manifest} \\
        --dir . \\
        --n-processes 4 \\
        --retry-amount 3

    python3 -c "
import hashlib, csv, sys, pathlib
ok = True
with open('${manifest}') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        fpath = pathlib.Path(row['filename'])
        if not fpath.exists():
            print(f'MISSING: {fpath}'); ok = False; continue
        md5 = hashlib.md5(fpath.read_bytes()).hexdigest()
        if md5 != row['md5']:
            print(f'MD5 MISMATCH: {fpath}'); ok = False
if not ok:
    sys.exit(1)
print('All MD5 checksums verified.')
"
    """
}

// ─── Process 3: Normalize RNA-seq counts to TPM ──────────────────────────────
process NORMALIZE_TPM {
    label 'cpu'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path rnaseq_files
    path gtf

    output:
    path "TCGA-BRCA_counts_gene_symbol.tsv", emit: counts
    path "TCGA-BRCA_tpm_matrix.tsv",         emit: tpm
    path "gene_lengths_gene_symbol.tsv",     emit: lengths

    script:
    """
    normalize_tpm.py \\
        --rnaseq-dir . \\
        --gtf ${gtf} \\
        --out-counts TCGA-BRCA_counts_gene_symbol.tsv \\
        --out-tpm    TCGA-BRCA_tpm_matrix.tsv \\
        --out-lengths gene_lengths_gene_symbol.tsv
    """
}

// ─── Process 4: Tile WSIs ─────────────────────────────────────────────────────
process TILE_WSI {
    label 'cpu'
    tag "${wsi.simpleName}"
    publishDir "${params.outdir}/tiles_256/${wsi.simpleName}", mode: 'copy'

    input:
    path wsi
    val  tile_size
    val  luminance_thr
    val  max_tiles

    output:
    path "${wsi.simpleName}/*.png"

    script:
    """
    tile_wsi.py \\
        --wsi        ${wsi} \\
        --outdir     ${wsi.simpleName} \\
        --tile-size  ${tile_size} \\
        --mag        20 \\
        --lum-thr    ${luminance_thr} \\
        --max-tiles  ${max_tiles}
    """
}

// ─── Process 5: Precompute tile embeddings ────────────────────────────────────
process PRECOMPUTE_EMBEDDINGS {
    label 'gpu'
    tag "${backbone}"
    publishDir "${params.outdir}/embeddings/${backbone}", mode: 'copy'

    input:
    val  backbone
    path tiles_dir
    val  hf_token

    output:
    path "*.pt"

    script:
    def token_env = hf_token ? "HUGGINGFACE_HUB_TOKEN=${hf_token}" : ""
    """
    ${token_env} precompute_embeddings.py \\
        --backbone   ${backbone} \\
        --tiles-dir  ${tiles_dir} \\
        --outdir     . \\
        --max-tiles  ${params.max_tiles}
    """
}
