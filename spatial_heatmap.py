"""
spatial_heatmap.py
Reconstruccion WSI + heatmap de prediccion por gen.

Enfoque: rejilla de tiles (cada tile = 1 celda, stride=256px).
Genera UNA figura por corte histologico (una columna, todas las filas = genes).
"""

import re, math, os, gc
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
GENES_VIZ     = ["EGFR", "ESR1", "MKI67", "ERBB2", "GATA3", "BCL2"]
CELL_PX       = 24       # px por celda de tile (>= 12; 24 es buen equilibrio)
HEATMAP_ALPHA = 0.60     # opacidad del heatmap sobre el tejido
CMAP          = "RdBu_r"
STRIDE        = 256      # paso del tiling en el WSI (px)
MAX_TILES     = 80       # tiles seleccionados para inferencia
FIG_WIDTH     = 14.0     # pulgadas de ancho de cada figura por muestra

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def xy_from_path(path):
    m = re.search(r'_x(\d+)_y(\d+)', str(path))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def tissue_score(path):
    try:
        with Image.open(path) as img:
            arr = np.frombuffer(
                img.resize((32, 32), Image.LANCZOS).convert("L").tobytes(),
                dtype=np.uint8).astype(np.float32)
        return 0.0 if arr.mean() > 210 else float(arr.std())
    except Exception:
        return 0.0


def _find_tile_dir(sample_id, tiles_dir):
    """Busca la carpeta de tiles probando distintos formatos de nombre."""
    base = Path(tiles_dir)
    for candidate in [
        sample_id,
        sample_id.replace('.', '-'),
        sample_id.replace('TCGA-', 'TCGA.').replace('-', '.'),
        sample_id.split('.')[-1],
    ]:
        d = base / candidate
        if d.exists():
            return d
    return base / sample_id   # fallback (puede no existir)


# ─────────────────────────────────────────────────────────────────────────────
# RECONSTRUCCION WSI EN REJILLA DE TILES
# ─────────────────────────────────────────────────────────────────────────────
def build_tile_grid(sample_id, tiles_dir, cell_px=CELL_PX, stride=STRIDE,
                    max_all=2000):
    """
    Coloca TODOS los tiles en su posicion de rejilla.
    Cada celda del grid = un tile = cell_px x cell_px pixeles.
    Devuelve (canvas, meta) o (None, None).
    """
    tile_dir  = _find_tile_dir(sample_id, tiles_dir)
    all_paths = sorted(tile_dir.glob("*.png")) if tile_dir.exists() else []
    if not all_paths:
        return None, None

    coords = []
    for p in all_paths[:max_all]:
        xy = xy_from_path(p)
        if xy[0] is not None:
            coords.append((xy[0], xy[1], p))
    if not coords:
        return None, None

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    x_min, y_min = min(xs), min(ys)

    cols = [(x - x_min) // stride for x, y, p in coords]
    rows = [(y - y_min) // stride for x, y, p in coords]
    n_cols = max(cols) + 1
    n_rows = max(rows) + 1

    H = n_rows * cell_px
    W = n_cols * cell_px
    canvas = np.ones((H, W, 3), dtype=np.float32)

    for (x, y, p), col, row in zip(coords, cols, rows):
        x0, y0 = col * cell_px, row * cell_px
        x1, y1 = x0 + cell_px, y0 + cell_px
        try:
            tile_img = np.array(
                Image.open(p).convert("RGB").resize(
                    (cell_px, cell_px), Image.LANCZOS),
                dtype=np.float32) / 255.
            canvas[y0:y1, x0:x1] = tile_img
        except Exception:
            pass

    meta = dict(x_min=x_min, y_min=y_min, stride=stride, cell_px=cell_px,
                n_cols=n_cols, n_rows=n_rows)
    return canvas, meta


# ─────────────────────────────────────────────────────────────────────────────
# HEATMAP EN REJILLA
# ─────────────────────────────────────────────────────────────────────────────
def build_heatmap_grid(paths, values, meta, sigma_cells=2.5):
    """Coloca predicciones en la rejilla y suaviza con Gaussian."""
    cell_px = meta['cell_px']
    x_min   = meta['x_min']
    y_min   = meta['y_min']
    stride  = meta['stride']
    n_cols  = meta['n_cols']
    n_rows  = meta['n_rows']
    H, W    = n_rows * cell_px, n_cols * cell_px
    sigma   = max(1.0, sigma_cells * cell_px)

    canvas = np.full((H, W), np.nan)
    count  = np.zeros((H, W))

    for p, v in zip(paths, values):
        xy = xy_from_path(p)
        if xy[0] is None:
            continue
        col = (xy[0] - x_min) // stride
        row = (xy[1] - y_min) // stride
        x0, y0 = col * cell_px, row * cell_px
        x1, y1 = x0 + cell_px, y0 + cell_px
        if x1 > W or y1 > H:
            continue
        prev = np.where(np.isnan(canvas[y0:y1, x0:x1]), 0.0,
                        canvas[y0:y1, x0:x1])
        cnt  = count[y0:y1, x0:x1]
        canvas[y0:y1, x0:x1] = (prev * cnt + v) / (cnt + 1)
        count [y0:y1, x0:x1] += 1

    data_mask = (count > 0).astype(float)
    filled    = np.where(np.isnan(canvas), 0.0, canvas)
    denom     = gaussian_filter(data_mask, sigma=sigma) + 1e-8
    smooth    = gaussian_filter(filled * data_mask, sigma=sigma) / denom

    tissue_mask = gaussian_filter(data_mask, sigma=sigma * 1.5) > 0.015
    smooth[~tissue_mask] = np.nan
    return smooth


# ─────────────────────────────────────────────────────────────────────────────
# RENDER DE UN PANEL (tejido + heatmap)
# ─────────────────────────────────────────────────────────────────────────────
def render_panel(ax, canvas, heatmap, vmin, vmax,
                 cmap=CMAP, alpha=HEATMAP_ALPHA,
                 scale_bar_cells=None, title=None, ylabel=None):
    ax.imshow(canvas, interpolation='nearest', aspect='auto')

    norm_    = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)
    hmap_rgba = cmap_obj(norm_(np.nan_to_num(heatmap, nan=(vmin + vmax) / 2)))
    hmap_rgba[..., 3] = np.where(np.isnan(heatmap), 0.0, alpha)
    ax.imshow(hmap_rgba, interpolation='bilinear', aspect='auto')

    # Fade en bordes blancos
    gray = canvas.mean(axis=2)
    bg   = (gray > 0.93).astype(float)
    fade = gaussian_filter(bg, sigma=3)
    fade_rgba          = np.zeros((*fade.shape, 4), dtype=float)
    fade_rgba[..., :3] = 1.0
    fade_rgba[..., 3]  = np.clip(fade * 2.0, 0, 1)
    ax.imshow(fade_rgba, interpolation='bilinear', aspect='auto')

    # Barra de escala
    if scale_bar_cells is not None:
        H, W = canvas.shape[:2]
        bar_w = scale_bar_cells
        ax.plot([W * 0.02, W * 0.02 + bar_w], [H * 0.90, H * 0.90],
                color='black', lw=2.5, solid_capstyle='butt')
        ax.text(W * 0.02, H * 0.95, '1 mm', fontsize=8,
                va='top', ha='left', color='black', fontweight='bold')

    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=4)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold',
                      rotation=0, ha='right', va='center', labelpad=48)
    ax.axis('off')


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCIA POR TILE
# ─────────────────────────────────────────────────────────────────────────────
def _select_tiles(sample_id, tiles_dir, max_tiles=MAX_TILES):
    """Selecciona los top-max_tiles tiles por tissue score."""
    tile_dir  = _find_tile_dir(sample_id, tiles_dir)
    all_paths = sorted(tile_dir.glob("*.png")) if tile_dir.exists() else []
    if not all_paths:
        return None
    scored = [(tissue_score(p), p) for p in all_paths]
    scored.sort(reverse=True)
    tissue = [(s, p) for s, p in scored if s > 0] or scored
    return [p for _, p in tissue[:max_tiles]]


def _load_precomp_emb_and_paths(sample_id, tiles_dir, emb_dir, max_tiles,
                                 max_tiles_precomp):
    """
    Carga embeddings precomputados desde disco y selecciona tiles por tissue score.

    Los embeddings en disco están ordenados igual que
    sorted(tile_dir.glob("*.png"))[:max_tiles_precomp].
    Cuando se usan embeddings precomputados se usan TODOS (max_tiles_precomp)
    para aprovechar los 500 tiles disponibles; max_tiles sólo aplica al path raw.

    Devuelve (sel_paths, sel_embs_tensor) o (None, None).
    """
    emb_path = Path(emb_dir) / f"{sample_id}.pt"
    if not emb_path.exists():
        return None, None

    tile_dir = _find_tile_dir(sample_id, tiles_dir)
    sorted_paths = sorted(tile_dir.glob("*.png"))[:max_tiles_precomp] \
        if tile_dir.exists() else []
    if not sorted_paths:
        return None, None

    path_to_idx = {p: i for i, p in enumerate(sorted_paths)}

    # Usar TODOS los tiles precomputados (ordenados por tissue score)
    scored = [(tissue_score(p), p) for p in sorted_paths]
    scored.sort(reverse=True)
    tissue = [(s, p) for s, p in scored if s > 0] or scored
    sel_paths = [p for _, p in tissue]   # todos los max_tiles_precomp tiles

    all_embs = torch.load(emb_path, map_location="cpu", weights_only=False)
    sel_indices = [path_to_idx[p] for p in sel_paths]
    sel_embs = all_embs[sel_indices]           # [T, emb_dim]
    return sel_paths, sel_embs


def _run_inference(lm, tile_fn, sample_id, tiles_dir, device,
                   max_tiles=MAX_TILES, emb_dir=None, max_tiles_precomp=500):
    """Devuelve (sel_paths, tile_probs [T x G]) o (None, None).

    Si emb_dir es especificado (para PrecomputedEmbMIL), carga los embeddings
    precomputados desde disco en lugar de procesar imagenes raw.
    """
    if emb_dir is not None:
        sel_paths, sel_embs = _load_precomp_emb_and_paths(
            sample_id, tiles_dir, emb_dir, max_tiles, max_tiles_precomp)
        if sel_paths is None:
            return None, None
        emb  = sel_embs.unsqueeze(0).to(device)   # [1, T, emb_dim]
        mask = torch.zeros(1, len(sel_paths), dtype=torch.bool, device=device)
        lm.eval().to(device)
        with torch.no_grad():
            tile_logits, _, _ = lm.model.forward_tiles(emb, mask)
        tile_probs = torch.sigmoid(tile_logits[0]).cpu().numpy()
        return sel_paths, tile_probs

    sel_paths = _select_tiles(sample_id, tiles_dir, max_tiles)
    if sel_paths is None:
        return None, None

    tiles = torch.stack([tile_fn(p, training=False) for p in sel_paths])
    x_img = tiles.unsqueeze(0).to(device)
    mask  = torch.zeros(1, len(sel_paths), dtype=torch.bool, device=device)

    lm.eval().to(device)
    with torch.no_grad():
        tile_logits, _, _ = lm.model.forward_tiles(x_img, mask)
    tile_probs = torch.sigmoid(tile_logits[0]).cpu().numpy()   # [T, G]
    return sel_paths, tile_probs


def _run_inference_with_attention(lm, tile_fn, sample_id, tiles_dir, device,
                                   max_tiles=MAX_TILES, emb_dir=None,
                                   max_tiles_precomp=500):
    """Devuelve (sel_paths, tile_probs [T,G], attn_w [T]) o (None, None, None).

    Si emb_dir es especificado (para PrecomputedEmbMIL), carga los embeddings
    precomputados desde disco en lugar de procesar imagenes raw.
    """
    if emb_dir is not None:
        sel_paths, sel_embs = _load_precomp_emb_and_paths(
            sample_id, tiles_dir, emb_dir, max_tiles, max_tiles_precomp)
        if sel_paths is None:
            return None, None, None
        emb  = sel_embs.unsqueeze(0).to(device)
        mask = torch.zeros(1, len(sel_paths), dtype=torch.bool, device=device)
        lm.eval().to(device)
        with torch.no_grad():
            tile_logits, attn_w, _ = lm.model.forward_tiles(emb, mask)
        tile_probs = torch.sigmoid(tile_logits[0]).cpu().numpy()
        attn_w = attn_w[0].cpu().numpy()
        attn_w = attn_w / (attn_w.max() + 1e-8)
        return sel_paths, tile_probs, attn_w

    sel_paths = _select_tiles(sample_id, tiles_dir, max_tiles)
    if sel_paths is None:
        return None, None, None

    tiles = torch.stack([tile_fn(p, training=False) for p in sel_paths])
    x_img = tiles.unsqueeze(0).to(device)
    mask  = torch.zeros(1, len(sel_paths), dtype=torch.bool, device=device)

    lm.eval().to(device)
    with torch.no_grad():
        tile_logits, attn_w, _ = lm.model.forward_tiles(x_img, mask)
    tile_probs = torch.sigmoid(tile_logits[0]).cpu().numpy()   # [T, G]
    attn_w     = attn_w[0].cpu().numpy()                       # [T]
    # Normalizar attn_w a [0,1] para que sea comparable entre muestras
    attn_w = attn_w / (attn_w.max() + 1e-8)
    return sel_paths, tile_probs, attn_w


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL — UNA FIGURA POR MUESTRA
# ─────────────────────────────────────────────────────────────────────────────
def generate_spatial_figures(lm, tile_fn, sample_ids, tiles_dir,
                              selected_genes, device,
                              genes_viz=None, n_samples=4,
                              cell_px=CELL_PX, alpha=HEATMAP_ALPHA,
                              out_dir="graficas_v1", model_name="Phikon+LoRA",
                              emb_dir=None, max_tiles_precomp=500):
    """
    Genera UNA figura PNG por corte histologico.

    Layout de cada figura:
      - Fila 0     : H&E puro (sin heatmap)
      - Filas 1..G : tejido + heatmap de prediccion para cada gen
      - Columna de colorbar a la derecha

    Parametros
    ----------
    lm            : LightningModule con lm.model.forward_tiles()
    tile_fn       : funcion de carga de tiles (load_tile_phikon / load_tile_dinov2)
    sample_ids    : lista de IDs de muestra
    tiles_dir     : ruta a carpeta de tiles
    selected_genes: lista de genes del modelo (en orden)
    device        : 'cuda' o 'cpu'
    genes_viz     : genes a mostrar (None = GENES_VIZ global)
    n_samples     : max numero de muestras
    cell_px       : px por celda de tile (mayor = mas detalle, mas RAM)
    alpha         : opacidad del heatmap (0=solo tejido, 1=solo heatmap)
    out_dir       : carpeta de salida
    model_name    : nombre para los titulos
    """
    if genes_viz is None:
        genes_viz = [g for g in GENES_VIZ if g in selected_genes]
    if not genes_viz:
        genes_viz = selected_genes[:6]

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    samples = [s for s in sample_ids if s][:n_samples]
    N_G = len(genes_viz)

    # 1 mm en px del canvas (a 20x: 1mm ≈ 3968px en WSI)
    def scale_bar(meta):
        return int(3968 / meta['stride'] * meta['cell_px'])

    for sid in samples:
        print(f"\n── Sample: {sid} ──────────────────────────")

        # Construir rejilla
        canvas, meta = build_tile_grid(sid, tiles_dir, cell_px=cell_px)
        if canvas is None:
            print("  No tiles found. Skipping.")
            continue

        # Inferencia
        sel_paths, tile_probs = _run_inference(
            lm, tile_fn, sid, tiles_dir, device,
            emb_dir=emb_dir, max_tiles_precomp=max_tiles_precomp)
        if sel_paths is None:
            print("  No predictions. Skipping.")
            continue

        print(f"  Grid: {meta['n_cols']}×{meta['n_rows']} celdas  "
              f"({canvas.shape[1]}×{canvas.shape[0]} px)")
        print(f"  Tiles inferencia: {len(sel_paths)}  "
              f"probs=[{tile_probs.min():.3f}, {tile_probs.max():.3f}]")

        # ── FIGURA: una columna, N_G+1 filas ────────────────────────────────
        ar = meta['n_cols'] / max(1, meta['n_rows'])   # aspect ratio WSI

        # Altura de cada fila en pulgadas
        row_h = FIG_WIDTH / ar
        row_h = max(row_h, 1.2)   # minimo 1.2"

        fig_h  = (N_G + 1) * (row_h + 0.15) + 0.7
        fig_w  = FIG_WIDTH + 0.6   # +0.6 para la colorbar

        fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
        gs  = gridspec.GridSpec(
            N_G + 1, 2,
            width_ratios=[FIG_WIDTH, 0.25],
            height_ratios=[row_h * 1.1] + [row_h] * N_G,
            hspace=0.08, wspace=0.03,
            left=0.08, right=0.97, top=0.96, bottom=0.02)

        sb = scale_bar(meta)
        H_c, W_c = canvas.shape[:2]

        # Fila 0: H&E puro
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(canvas, interpolation='nearest', aspect='auto')
        ax0.plot([W_c * 0.02, W_c * 0.02 + sb], [H_c * 0.88, H_c * 0.88],
                 color='k', lw=2.5, solid_capstyle='butt')
        ax0.text(W_c * 0.02, H_c * 0.93, '1 mm', fontsize=8,
                 va='top', ha='left', color='k', fontweight='bold')
        ax0.set_ylabel('H&E', fontsize=12, fontweight='bold',
                       rotation=0, ha='right', va='center', labelpad=48)
        ax0.axis('off')

        # Colorbar vacia en la fila 0 (para alinear)
        ax_cb0 = fig.add_subplot(gs[0, 1])
        ax_cb0.axis('off')

        # Filas 1..N_G: heatmaps por gen
        for i, gene in enumerate(genes_viz):
            if gene not in selected_genes:
                continue
            gene_idx = selected_genes.index(gene)
            vals = tile_probs[:, gene_idx]

            # Rango percentil 5-95 de las predicciones de este gen/muestra
            vmin_g = float(np.percentile(vals, 5))
            vmax_g = float(np.percentile(vals, 95))
            if vmax_g - vmin_g < 0.05:
                mid = (vmin_g + vmax_g) / 2
                vmin_g, vmax_g = mid - 0.15, mid + 0.15

            hmap = build_heatmap_grid(sel_paths, vals, meta, sigma_cells=2.5)

            ax = fig.add_subplot(gs[i + 1, 0])
            render_panel(ax, canvas, hmap, vmin_g, vmax_g,
                         alpha=alpha, scale_bar_cells=sb, ylabel=gene)

            # Colorbar
            ax_cb = fig.add_subplot(gs[i + 1, 1])
            sm = plt.cm.ScalarMappable(
                cmap=CMAP, norm=Normalize(vmin_g, vmax_g))
            sm.set_array([])
            cb = plt.colorbar(sm, cax=ax_cb)
            cb.set_ticks([vmin_g, vmax_g])
            cb.set_ticklabels(['low', 'high'], fontsize=9)
            cb.ax.tick_params(labelsize=9)

        short = sid.replace('TCGA.', '').replace('.', '-')
        fig.suptitle(
            f"{short}  —  Spatial gene expression  ({model_name})",
            fontsize=13, fontweight='bold')

        out_path = out_dir / f"spatial_{short}.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight',
                    facecolor='white')
        plt.show()
        print(f"  Saved: {out_path}")

        plt.close(fig)
        gc.collect()
        torch.cuda.empty_cache()

    print("\nSpatial figures complete.")


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZACIÓN BASADA EN ATENCIÓN — UNA FIGURA POR GEN
# ─────────────────────────────────────────────────────────────────────────────
# Colormap divergente: azul=baja expresion, blanco=neutro, rojo=alta expresion
CMAP_ATTN = "RdBu_r"

# Tamaño fijo de cada panel (pulgadas) — aspect='auto' comprime el WSI para que quepa
PANEL_W = 3.2   # ancho por muestra
PANEL_H = 2.0   # alto de cada fila


def _signed_attended(attn_w, tile_probs_gene):
    """
    Activación atendida con signo: attn_w[t] * (prob[t,g] - 0.5) * 2

    Valores en [-1, +1]:
      +1  (rojo)  → tile muy atendido Y modelo predice expresión ALTA
      -1  (azul)  → tile muy atendido Y modelo predice expresión BAJA
       0  (blanco)→ tile ignorado  O  modelo indeciso (prob ≈ 0.5)
    """
    return attn_w * (tile_probs_gene - 0.5) * 2.0


def _draw_he_panel(ax, canvas, meta, sid_label):
    """Dibuja un panel H&E con barra de escala y etiqueta de muestra."""
    H_c, W_c = canvas.shape[:2]
    sb = int(3968 / meta['stride'] * meta['cell_px'])
    ax.imshow(canvas, interpolation='nearest', aspect='auto')
    ax.plot([W_c*0.03, W_c*0.03+sb], [H_c*0.88, H_c*0.88],
            color='k', lw=2.0, solid_capstyle='butt')
    ax.text(W_c*0.03, H_c*0.93, '1 mm', fontsize=7,
            va='top', ha='left', color='k', fontweight='bold')
    ax.set_title(sid_label, fontsize=8, fontweight='bold', pad=3)
    ax.axis('off')


def generate_attention_figures(lm, tile_fn, sample_ids, tiles_dir,
                                selected_genes, device,
                                genes_viz=None, n_samples=5,
                                cell_px=CELL_PX, alpha=0.70,
                                out_dir="graficas_v1", model_name="Phikon+LoRA",
                                emb_dir=None, max_tiles_precomp=500):
    """
    Genera UNA figura PNG por (muestra, gen).

    Estructura de carpetas:
        out_dir/spatial_<sample>/
            ESR1.png
            PGR.png
            ...

    Layout de cada figura (1 muestra, 1 gen):
      - Fila 0 : H&E puro
      - Fila 1 : p(alta expresion) - 0.5 para cada tile
                   Rojo  → tile predice expresion ALTA
                   Blanco→ tile indeciso (prob~0.5)
                   Azul  → tile predice expresion BAJA
                   Cubre TODOS los tiles precomputados (500)

    Omite figuras ya existentes — si el kernel muere, reanuda donde se quedo.
    Libera memoria tras cada muestra (no acumula canvases en RAM).
    """
    if genes_viz is None:
        genes_viz = [g for g in GENES_VIZ if g in selected_genes]
    if not genes_viz:
        genes_viz = selected_genes[:6]

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    samples = [s for s in sample_ids if s][:n_samples]

    for sid in samples:
        short = sid.replace('TCGA.', '').replace('.', '-')
        sample_dir = out_dir / f"spatial_{short}"
        sample_dir.mkdir(exist_ok=True)

        # Comprobar si todos los genes de esta muestra ya están generados
        pending = [g for g in genes_viz
                   if g in selected_genes and not (sample_dir / f"{g}.png").exists()]
        if not pending:
            print(f"  {short}: all figures already exist, skipping.")
            continue

        print(f"\n── {short}  ({len(pending)} genes pending) ──────────────")

        # Construir rejilla
        canvas, meta = build_tile_grid(sid, tiles_dir, cell_px=cell_px)
        if canvas is None:
            print("  No tiles found. Skipping.")
            continue

        # Inferencia (solo si hay genes pendientes)
        sel_paths, tile_probs, attn_w = _run_inference_with_attention(
            lm, tile_fn, sid, tiles_dir, device,
            emb_dir=emb_dir, max_tiles_precomp=max_tiles_precomp)
        if sel_paths is None:
            print("  No predictions. Skipping.")
            continue

        print(f"  Grid: {meta['n_cols']}x{meta['n_rows']}  "
              f"tiles={len(sel_paths)}  attn=[{attn_w.min():.2f},{attn_w.max():.2f}]")

        # Escala de la figura: adaptar al aspect ratio del WSI
        ar    = meta['n_cols'] / max(1, meta['n_rows'])
        row_h = max(FIG_WIDTH / ar, 1.5)
        fig_w = FIG_WIDTH + 0.55
        fig_h = 2 * row_h + 0.7

        sb = int(3968 / meta['stride'] * meta['cell_px'])
        H_c, W_c = canvas.shape[:2]

        for gene in pending:
            out_path = sample_dir / f"{gene}.png"

            gene_idx  = selected_genes.index(gene)
            # Usar prediccion directa (no ponderada por atencion) para colorear todos los tiles
            pred_vals = tile_probs[:, gene_idx] - 0.5   # centrado en 0: >0=alta, <0=baja
            hmap      = build_heatmap_grid(sel_paths, pred_vals, meta, sigma_cells=2.5)

            valid_v = hmap[~np.isnan(hmap)]
            pabs    = float(np.percentile(np.abs(valid_v), 97)) if len(valid_v) else 0.15
            pabs    = max(pabs, 0.05)

            fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
            gs  = gridspec.GridSpec(
                2, 2,
                width_ratios=[FIG_WIDTH, 0.25],
                height_ratios=[row_h, row_h],
                hspace=0.06, wspace=0.03,
                left=0.06, right=0.97, top=0.93, bottom=0.03)

            # Fila 0: H&E
            ax0 = fig.add_subplot(gs[0, 0])
            ax0.imshow(canvas, interpolation='nearest', aspect='auto')
            ax0.plot([W_c*0.03, W_c*0.03+sb], [H_c*0.88, H_c*0.88],
                     color='k', lw=2.0, solid_capstyle='butt')
            ax0.text(W_c*0.03, H_c*0.93, '1 mm', fontsize=8,
                     va='top', ha='left', color='k', fontweight='bold')
            ax0.set_ylabel('H&E', fontsize=10, fontweight='bold',
                           rotation=0, ha='right', va='center', labelpad=36)
            ax0.axis('off')
            fig.add_subplot(gs[0, 1]).axis('off')

            # Fila 1: prediccion por tile (todos los tiles coloreados)
            ax1 = fig.add_subplot(gs[1, 0])
            render_panel(ax1, canvas, hmap, vmin=-pabs, vmax=pabs,
                         cmap=CMAP_ATTN, alpha=alpha,
                         scale_bar_cells=sb, ylabel=gene)

            ax_cb = fig.add_subplot(gs[1, 1])
            sm = plt.cm.ScalarMappable(cmap=CMAP_ATTN, norm=Normalize(-pabs, pabs))
            sm.set_array([])
            cb = plt.colorbar(sm, cax=ax_cb)
            cb.set_ticks([-pabs, 0, pabs])
            cb.set_ticklabels(['low', 'neutral', 'high'], fontsize=9)
            cb.ax.tick_params(labelsize=9)

            fig.suptitle(f"{gene}  ·  {short}  ({model_name})",
                         fontsize=12, fontweight='bold')

            plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"  {gene}: saved → {out_path}")

        # Liberar memoria de esta muestra antes de pasar a la siguiente
        del canvas, tile_probs, attn_w, sel_paths
        gc.collect()
        torch.cuda.empty_cache()

    print("\nAttention figures complete.")


# ─────────────────────────────────────────────────────────────────────────────
# USO DESDE EL NOTEBOOK:
# ─────────────────────────────────────────────────────────────────────────────
# import importlib, sys
# if 'spatial_heatmap' in sys.modules:
#     importlib.reload(sys.modules['spatial_heatmap'])
# from spatial_heatmap import generate_spatial_figures
#
# sample_ids = [str(r["sample_id"]) for _, r in ds_test_dino.manifest.iterrows()]
# generate_spatial_figures(
#     lm             = lm_phikon,
#     tile_fn        = load_tile_phikon,
#     sample_ids     = sample_ids,
#     tiles_dir      = TILES_DIR,
#     selected_genes = selected_genes,
#     device         = DEVICE,
#     genes_viz      = ["PGR","ESR1","MKI67","ERBB2","GATA3","BCL2"],
#     n_samples      = 4,
#     cell_px        = 24,
#     alpha          = 0.60,
#     out_dir        = "graficas_v1",
#     model_name     = "Phikon+LoRA",
# )
