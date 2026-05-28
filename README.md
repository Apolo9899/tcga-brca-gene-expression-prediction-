# tcga-brca-gene-expression-prediction-
Gene expression prediction from breast cancer H&amp;E whole slide images using pathology foundation models (DINOv2, Phikon, UNI, Virchow2, GigaPath) and TransformerMIL — TCGA-BRCA,   19 genes, mean AUROC 0.87

 # Gene Expression Prediction from Breast Cancer H&E Whole Slide Images

  Predicting the binary expression status of 19 clinically relevant genes directly
  from hematoxylin & eosin-stained whole slide images (WSI) of breast cancer, using
  pathology-specific Vision Transformer foundation models and Transformer-based
  Multiple Instance Learning (TransformerMIL).

  **Dataset:** TCGA-BRCA · **Best model:** Weighted ensemble (DINOv2-S + Phikon + UNI)
  · **Mean AUROC:** 0.87 over 19 genes (test set)

  ---

  ## Key Results

  | Gene | AUROC | Gene | AUROC |
  |------|-------|------|-------|
  | FOXA1 | 0.986 | GATA3 | 0.969 |
  | BCL2  | 0.975 | AURKA | 0.955 |
  | PGR   | 0.961 | ESR1  | 0.944 |

  All pathology foundation models outperform ResNet50 by 10–28 AUROC points.
  The weighted ensemble outperforms the best individual model (GigaPath, 0.846) by 2.7 points.

  ---

  ## Repository Structure

  .
  ├── modelo_tfm_v2.ipynb       # Main pipeline: training, evaluation, figures, XAI
  ├── spatial_heatmap.py        # Spatial attention maps and XAI utilities
  ├── pyproject.toml            # Python environment specification (uv)
  ├── uv.lock                   # Locked dependency versions
  ├── nextflow/
  │   └── pipeline.nf           # Preprocessing pipeline (WSI tiling + RNA-seq TPM)
  └── README.md

  > **Large files** (checkpoints, precomputed embeddings) are hosted on Zenodo:
  > `https://doi.org/10.5281/zenodo.XXXXXXX`
  > Download instructions are in the [Data & Models](#data--models) section.

  ---

  ## Installation

  This project uses [uv](https://github.com/astral-sh/uv) for reproducible environment management.

  ```bash
  # Install uv (if not already installed)
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # Clone the repository
  git clone https://github.com/<your-username>/<repo-name>.git
  cd <repo-name>

  # Create environment and install dependencies
  uv sync

  Alternatively, with pip:

  pip install -r requirements.txt

  Requirements: Python 3.10 · PyTorch ≥ 2.0 · CUDA-capable GPU (≥ 16 GB VRAM recommended)

  ---
  Data & Models

  1. TCGA-BRCA WSI and RNA-seq data

  Data is publicly available at the GDC Data Portal (https://portal.gdc.cancer.gov/).
  Download using the GDC client and the manifest included in this repository:

  # Install gdc-client
  # https://gdc.cancer.gov/access-data/gdc-data-transfer-tool

  gdc-client download -m gdc_manifest_wsi.txt   -d data/wsi/
  gdc-client download -m gdc_manifest_rnaseq.txt -d data/rnaseq/

  2. Preprocessing pipeline (Nextflow)

  # Requires Nextflow ≥ 23.x
  nextflow run nextflow/pipeline.nf \
    --wsi_dir data/wsi/ \
    --rnaseq_dir data/rnaseq/ \
    --outdir data/processed/

  This pipeline handles: WSI tiling (256×256 px at 20×) → tissue filtering → TPM
  normalization → embedding precomputation for each backbone.

  3. Foundation model access

  ┌───────────────────┬──────────────────────────────────────────┬─────────────────────┐
  │       Model       │                  Source                  │       Access        │
  ├───────────────────┼──────────────────────────────────────────┼─────────────────────┤
  │ DINOv2-S / DINOv3 │ torch.hub (facebookresearch/dinov2)      │ Public (Apache 2.0) │
  ├───────────────────┼──────────────────────────────────────────┼─────────────────────┤
  │ Phikon            │ HuggingFace: owkin/phikon                │ Public              │
  ├───────────────────┼──────────────────────────────────────────┼─────────────────────┤
  │ Virchow2          │ HuggingFace: paige-ai/Virchow2           │ Requires agreement  │
  ├───────────────────┼──────────────────────────────────────────┼─────────────────────┤
  │ UNI               │ HuggingFace: MahmoodLab/UNI              │ Requires agreement  │
  ├───────────────────┼──────────────────────────────────────────┼─────────────────────┤
  │ Prov-GigaPath     │ HuggingFace: prov-gigapath/prov-gigapath │ Requires agreement  │
  └───────────────────┴──────────────────────────────────────────┴─────────────────────┘

  For models requiring an access agreement, accept the terms on their HuggingFace
  page and set your token:

  huggingface-cli login

  4. Pre-trained checkpoints and embeddings (Zenodo)

  Download and place in the project root:

  # Checkpoints (~3 GB total)
  zenodo_get 10.5281/zenodo.XXXXXXX -o .

  # Precomputed embeddings (~7 GB total, optional — skip to use FORCE_EXTRACT=True)
  zenodo_get 10.5281/zenodo.YYYYYYY -o .

  ---
  Running the Pipeline

  Open modelo_tfm_v2.ipynb in Jupyter and run cells sequentially.
  All computationally expensive steps are gated by boolean flags:

  FORCE_EXTRACT = False   # Set True to recompute tile embeddings
  FORCE_TRAIN_* = False   # Set True to retrain a specific backbone head

  With cached embeddings and checkpoints, the full notebook (all results + figures)
  runs in under 10 minutes on CPU.

  ---
  Gene Panel

  19 clinically relevant genes covering key breast cancer pathways:

  ┌─────────────────────────────────────┬───────────────────────────────┐
  │              Category               │             Genes             │
  ├─────────────────────────────────────┼───────────────────────────────┤
  │ Luminal markers / hormone receptors │ ESR1, PGR, FOXA1, GATA3, BCL2 │
  ├─────────────────────────────────────┼───────────────────────────────┤
  │ Proliferation                       │ MKI67, AURKA                  │
  ├─────────────────────────────────────┼───────────────────────────────┤
  │ DNA repair                          │ BRCA1, BRCA2                  │
  ├─────────────────────────────────────┼───────────────────────────────┤
  │ HER2 pathway                        │ ERBB2, EGFR, FGFR1            │
  ├─────────────────────────────────────┼───────────────────────────────┤
  │ PI3K/AKT pathway                    │ PIK3CA, PTEN                  │
  ├─────────────────────────────────────┼───────────────────────────────┤
  │ Tumour suppressor                   │ TP53                          │
  ├─────────────────────────────────────┼───────────────────────────────┤
  │ Immune checkpoint                   │ CD274 (PD-L1)                 │
  ├─────────────────────────────────────┼───────────────────────────────┤
  │ EMT                                 │ CDH1, SNAI1, VIM              │
  └─────────────────────────────────────┴───────────────────────────────┘

  ---
  Architecture

  Frozen backbone (ViT)
          │
    Tile embeddings  [N_tiles × D_backbone]
          │
     Linear projection  →  [N_tiles × 256]
          │
    TransformerMIL  (4 heads, 2 layers)
          │
    Sigmoid output  →  [19 genes]

  Ensemble: weighted soft voting of DINOv2-S, Phikon and UNI, with per-gene weights
  derived from validation AUROC.

  ---
  Citation

  If you use this code or results in your research, please cite:

  @mastersthesis{author2025gene,
    title   = {Prediction of gene expression profile in breast cancer from histopathological images using foundational models of pathology and multiple instance learning},
    author  = {<Marco Apolo Pulpillo Berrocal>},
    school  = {<Universidad Politécnica de Madrid>},
    year    = {2026}
  }

  ---
  License

  Code: MIT License.
  Data: subject to TCGA data use policy (https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga/using-tcga/citing-tcga).
  Foundation models: subject to their respective licenses (see Data & Models (#data--models)).

  ---
