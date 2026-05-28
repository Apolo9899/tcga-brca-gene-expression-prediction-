# Gene Expression Prediction from Breast Cancer H&E Whole Slide Images

Predicting the binary expression status of 19 clinically relevant genes directly
from hematoxylin & eosin-stained whole slide images (WSI) of breast cancer, using
pathology-specific Vision Transformer foundation models and Transformer-based
Multiple Instance Learning (TransformerMIL).

**Dataset:** TCGA-BRCA · **Best model:** Weighted ensemble (DINOv2-S + Phikon + UNI)
· **Mean AUROC:** 0.87 over 19 genes (test set)

---

## Key Results

| Gene  | AUROC | Gene  | AUROC |
|-------|-------|-------|-------|
| FOXA1 | 0.986 | GATA3 | 0.969 |
| BCL2  | 0.975 | AURKA | 0.955 |
| PGR   | 0.961 | ESR1  | 0.944 |

All pathology foundation models outperform ResNet50 by 10–28 AUROC points.
The weighted ensemble outperforms the best individual model (GigaPath, 0.846) by 2.7 points.

---

## Repository Structure

```
.
├── modelo_tfm_v2.ipynb            # Main pipeline: training, evaluation, figures, XAI
├── spatial_heatmap.py             # Spatial attention maps and XAI utilities
├── pyproject.toml                 # uv project definition (dependencies + metadata)
├── uv.lock                        # 185 packages locked to exact versions
├── requirements.txt               # Alternative: pip install -r requirements.txt
├── nextflow/
│   ├── pipeline.nf                # Nextflow workflow (4-stage preprocessing)
│   ├── nextflow.config            # Resource profiles: local / SLURM
│   └── bin/
│       ├── normalize_tpm.py       # Raw counts → TPM (GRCh38 GTF)
│       ├── tile_wsi.py            # WSI → 256×256 px tiles with tissue filter
│       └── precompute_embeddings.py  # Frozen backbone → per-slide .pt embeddings
├── preds_cache_multi/             # Cached test/val logits and labels for all models
└── ckpt_*/best_*.ckpt             # Best checkpoint per backbone (7 models)
```

---

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fully reproducible
environment management. The `uv.lock` file pins all 185 direct and transitive
dependencies to exact versions, ensuring the environment is identical across machines.

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repository
git clone https://github.com/Apolo9899/tcga-brca-gene-expression-prediction-.git
cd tcga-brca-gene-expression-prediction-

# 3. Create environment and install all dependencies (uses uv.lock)
uv sync
```

The `uv sync` command reads `uv.lock` and installs every package at its exact pinned
version — no dependency conflicts, no version drift.

**Alternative (pip):**

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.10 · PyTorch ≥ 2.0 · CUDA-capable GPU (≥ 16 GB VRAM for training)

---

## Data & Models

### 1. TCGA-BRCA WSI and RNA-seq data

Data is publicly available at the [GDC Data Portal](https://portal.gdc.cancer.gov/).
Download using the GDC client and the manifest files included in this repository:

```bash
# Install gdc-client: https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
gdc-client download -m data/gdc_manifest_wsi.txt    -d data/wsi/
gdc-client download -m data/gdc_manifest_rnaseq.txt -d data/rnaseq/
```

### 2. Preprocessing pipeline (Nextflow)

The full preprocessing pipeline is implemented as a Nextflow workflow covering
four sequential stages:

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `gdc-client` | Download WSI (.svs) and RNA-seq count files; verify MD5 checksums |
| 2 | `bin/normalize_tpm.py` | Normalize raw counts to TPM using gene lengths from GRCh38 GTF |
| 3 | `bin/tile_wsi.py` | Tile WSIs into 256×256 px fragments at 20×; discard background tiles (luminance > 210/255, ITU-R BT.601); keep up to 500 tiles per slide |
| 4 | `bin/precompute_embeddings.py` | Run each frozen backbone over the tiles; save per-slide embedding tensors as `.pt` files |

Nextflow caches intermediate results automatically — re-running the pipeline after
adding new samples only processes the new ones.

```bash
# Requires Nextflow ≥ 23.x  (https://www.nextflow.io/docs/latest/install.html)
nextflow run nextflow/pipeline.nf \
  -profile local \
  --wsi_manifest    data/gdc_manifest_wsi.txt \
  --rnaseq_manifest data/gdc_manifest_rnaseq.txt \
  --gtf             data/gencode.v38.annotation.gtf.gz \
  --outdir          data/processed

# For HPC clusters with SLURM:
nextflow run nextflow/pipeline.nf -profile slurm ...
```

The `bin/` scripts can also be run independently for debugging:

```bash
# Tile a single WSI
python nextflow/bin/tile_wsi.py \
  --wsi path/to/slide.svs --outdir tiles/ \
  --tile-size 256 --lum-thr 210 --max-tiles 500

# Normalize RNA-seq counts to TPM
python nextflow/bin/normalize_tpm.py \
  --rnaseq-dir data/rnaseq/ --gtf data/gencode.v38.annotation.gtf.gz \
  --out-tpm data/processed/tpm_matrix.tsv \
  --out-counts data/processed/counts_matrix.tsv \
  --out-lengths data/processed/gene_lengths.tsv

# Precompute embeddings for a given backbone
python nextflow/bin/precompute_embeddings.py \
  --backbone dinov2 --tiles-dir tiles/ --outdir embeddings/dinov2/
```

### 3. Foundation model access

| Model | Source | Access |
|-------|--------|--------|
| DINOv2-S / DINOv3 | `torch.hub` (facebookresearch/dinov2) | Public (Apache 2.0) |
| Phikon | HuggingFace: `owkin/phikon` | Public |
| Virchow2 | HuggingFace: `paige-ai/Virchow2` | Requires agreement |
| UNI | HuggingFace: `MahmoodLab/UNI` | Requires agreement |
| Prov-GigaPath | HuggingFace: `prov-gigapath/prov-gigapath` | Requires agreement |

For gated models, accept the terms on HuggingFace and authenticate:

```bash
huggingface-cli login
```

---

## Running the Main Notebook

Open `modelo_tfm_v2.ipynb` in Jupyter and run cells sequentially.
Computationally expensive steps are gated by boolean flags:

```python
FORCE_EXTRACT   = False   # Set True to recompute tile embeddings from scratch
FORCE_TRAIN_*   = False   # Set True to retrain a specific backbone head
```

The repository includes cached test/val predictions (`preds_cache_multi/`) and best
checkpoints (`ckpt_*/best_*.ckpt`) for all 7 models. With these, the full notebook —
all results and figures — runs in under 10 minutes on CPU without needing a GPU.

---

## Gene Panel

| Category | Genes |
|----------|-------|
| Luminal markers / hormone receptors | ESR1, PGR, FOXA1, GATA3, BCL2 |
| Proliferation | MKI67, AURKA |
| DNA repair | BRCA1, BRCA2 |
| HER2 / growth factor receptors | ERBB2, EGFR, FGFR1 |
| PI3K/AKT pathway | PIK3CA, PTEN |
| Tumour suppressor | TP53 |
| Immune checkpoint | CD274 (PD-L1) |
| EMT | CDH1, SNAI1, VIM |

---

## Architecture

```
Frozen backbone (ViT)
        │
  Tile embeddings  [N_tiles × D_backbone]
        │
  Linear projection  →  [N_tiles × 256]
        │
  TransformerMIL  (4 heads, 2 layers, D_ff = 512)
        │
  Sigmoid output  →  [19 genes]
```

Ensemble: weighted soft voting of DINOv2-S, Phikon and UNI, with per-gene weights
derived from validation AUROC.

---

## Citation

```bibtex
@mastersthesis{apolo2026gene,
  title   = {Prediction of gene expression profile in breast cancer from
             histopathological images using foundational models of pathology
             and multiple instance learning},
  author  = {Marco Apolo Pulpillo Berrocal},
  school  = {Universidad Politécnica de Madrid},
  year    = {2026}
}
```

---

## License

Code: MIT License.  
Data: subject to [TCGA data use policy](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga/using-tcga/citing-tcga).  
Foundation models: subject to their respective licenses (see [Data & Models](#data--models)).
