"""
Data Augmentation Script
========================
Replica locale del notebook Colab "Data_augmentation-3.ipynb".

Cosa fa questo script:
1. Carica tutte le immagini e le maschere dalla cartella locale.
2. Applica data augmentation (5 versioni per immagine):
   - Originale, Saturazione 0.5 (maschera invariata), Rotazione 90°, Flip H, Flip V.
   Totale: 88 × 5 = 440 immagini augmentate.
3. Genera patch 256×256 solo se la maschera ha almeno MIN_POSITIVE_PIXELS pixel non neri.
4. Genera lo stesso numero di patch negative (maschera tutta zero) per bilanciamento.

Uso:
    python data_augmentation.py
"""

import os
import glob
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt


def apply_saturation(img_rgb, factor=0.5):
    """Riduce la saturazione dell'immagine (maschera invariata). img in RGB."""
    hsv = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

# ──────────────────────────────────────────────
# CONFIGURAZIONE - modifica questi percorsi
# ──────────────────────────────────────────────
BASE_DIR = r"C:\Users\gdf01\Documents\Segmentation\data\images\newimages_1"

IMAGES_DIR = os.path.join(BASE_DIR, "png")       # cartella immagini PNG
MASKS_DIR  = os.path.join(BASE_DIR, "masks")      # cartella maschere

# Cartelle di output (verranno create se non esistono)
OUT_IMAGES_DIR  = os.path.join(BASE_DIR, "augmented", "images")
OUT_MASKS_DIR   = os.path.join(BASE_DIR, "augmented", "masks")
OUT_PATCHES_IMG = os.path.join(BASE_DIR, "augmented", "patches_images")
OUT_PATCHES_MSK = os.path.join(BASE_DIR, "augmented", "patches_masks")

# Estensione dei file sorgente
IMAGE_EXT = "*.jpg"   # immagini in newimages_1/png sono JPG
MASK_EXT  = "*.tiff"  # maschere in newimages_1/masks sono OME-TIFF

CROP_ROWS = None  # Imposta a 1280 se vuoi tagliare la scale bar (come nel notebook)
                   # None = nessun crop

PATCH_SIZE = 256   # Dimensione delle patch quadrate
PATCH_STEP = 256   # Step (non-overlapping se uguale a PATCH_SIZE)
MIN_POSITIVE_PIXELS = 400   # Salva patch solo se la maschera ha almeno N pixel non neri

SHOW_PLOTS = False  # True per visualizzare i plot come nel notebook

# ──────────────────────────────────────────────
# CREAZIONE CARTELLE OUTPUT
# ──────────────────────────────────────────────
for d in [OUT_IMAGES_DIR, OUT_MASKS_DIR, OUT_PATCHES_IMG, OUT_PATCHES_MSK]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────
# 1. CARICAMENTO IMMAGINI E MASCHERE
# ──────────────────────────────────────────────
print("Caricamento immagini...")
image_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, IMAGE_EXT)))
mask_paths  = sorted(glob.glob(os.path.join(MASKS_DIR, MASK_EXT)))

assert len(image_paths) > 0, f"Nessuna immagine trovata in {IMAGES_DIR}"
assert len(mask_paths) > 0,  f"Nessuna maschera trovata in {MASKS_DIR}"
assert len(image_paths) == len(mask_paths), (
    f"Numero immagini ({len(image_paths)}) != numero maschere ({len(mask_paths)})"
)

images = [cv2.imread(p, cv2.IMREAD_COLOR) for p in image_paths]   # BGR, 3 canali
masks  = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in mask_paths]  # grayscale

# Converti immagini BGR → RGB (per coerenza con il notebook che usava skimage)
images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]

image_dataset = np.array(images)
mask_dataset  = np.array(masks)

# Crop opzionale (nel notebook: image_dataset[:, :1280, :, :])
if CROP_ROWS is not None:
    image_dataset = image_dataset[:, :CROP_ROWS, :, :]
    mask_dataset  = mask_dataset[:, :CROP_ROWS, :]

print(f"Image data shape: {image_dataset.shape}")
print(f"Mask data shape:  {mask_dataset.shape}")
print(f"Max pixel value:  {image_dataset.max()}")
print(f"Classi nella mask: {np.unique(mask_dataset)}")

# ──────────────────────────────────────────────
# 2. DATA AUGMENTATION — 5 versioni per immagine
#    Originale, Saturazione 0.5 (maschera invariata), Rotazione 90°, Flip H, Flip V
# ──────────────────────────────────────────────
print("\nApplicazione augmentation (orig, sat 0.5, rot90, flipH, flipV)...")

num_images = len(image_dataset)

for i in range(num_images):
    img = image_dataset[i]
    msk = mask_dataset[i]
    base_name = os.path.splitext(os.path.basename(image_paths[i]))[0]

    # 1) Originale
    cv2.imwrite(
        os.path.join(OUT_IMAGES_DIR, f"{base_name}_orig.png"),
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(os.path.join(OUT_MASKS_DIR, f"{base_name}_orig.png"), msk)

    # 2) Saturazione 0.5 — solo immagine, maschera invariata
    img_sat = apply_saturation(img, factor=0.5)
    cv2.imwrite(
        os.path.join(OUT_IMAGES_DIR, f"{base_name}_sat.png"),
        cv2.cvtColor(img_sat, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(os.path.join(OUT_MASKS_DIR, f"{base_name}_sat.png"), msk)

    # 3) Rotazione 90° (orario) — immagine e maschera
    img_rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    msk_rot = cv2.rotate(msk, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(
        os.path.join(OUT_IMAGES_DIR, f"{base_name}_rot90.png"),
        cv2.cvtColor(img_rot, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(os.path.join(OUT_MASKS_DIR, f"{base_name}_rot90.png"), msk_rot)

    # 4) Flip orizzontale
    img_flipH = cv2.flip(img, 1)
    msk_flipH = cv2.flip(msk, 1)
    cv2.imwrite(
        os.path.join(OUT_IMAGES_DIR, f"{base_name}_flipH.png"),
        cv2.cvtColor(img_flipH, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(os.path.join(OUT_MASKS_DIR, f"{base_name}_flipH.png"), msk_flipH)

    # 5) Flip verticale
    img_flipV = cv2.flip(img, 0)
    msk_flipV = cv2.flip(msk, 0)
    cv2.imwrite(
        os.path.join(OUT_IMAGES_DIR, f"{base_name}_flipV.png"),
        cv2.cvtColor(img_flipV, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(os.path.join(OUT_MASKS_DIR, f"{base_name}_flipV.png"), msk_flipV)

    if (i + 1) % 10 == 0:
        print(f"  Processate {i + 1}/{num_images}")

expected_total = num_images * 5
print(f"Augmentation completata: {num_images} immagini × 5 = {expected_total} file.")
print(f"Salvati in:\n  {OUT_IMAGES_DIR}\n  {OUT_MASKS_DIR}")

# ──────────────────────────────────────────────
# 3. PATCHIFY — Patch 256×256 solo se ≥ MIN_POSITIVE_PIXELS non neri; + bilanciamento negative
# ──────────────────────────────────────────────
try:
    from patchify import patchify
except ImportError:
    print("\nInstallo patchify...")
    os.system("pip install patchify")
    from patchify import patchify

print(f"\nCreazione patch {PATCH_SIZE}x{PATCH_SIZE} (solo se maschera ha ≥ {MIN_POSITIVE_PIXELS} pixel non neri)...")

aug_img_paths = sorted(glob.glob(os.path.join(OUT_IMAGES_DIR, "*.png")))
aug_msk_paths = sorted(glob.glob(os.path.join(OUT_MASKS_DIR, "*.png")))

positive_count = 0
negative_candidates = []   # lista di (patch_img, name_prefix) per patch con maschera tutta zero

for img_path, msk_path in zip(aug_img_paths, aug_msk_paths):
    name = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)

    h, w = msk.shape[:2]
    new_h = (h // PATCH_SIZE) * PATCH_SIZE
    new_w = (w // PATCH_SIZE) * PATCH_SIZE
    img = img[:new_h, :new_w]
    msk = msk[:new_h, :new_w]

    img_patches = patchify(img, (PATCH_SIZE, PATCH_SIZE, 3), step=PATCH_STEP)
    msk_patches = patchify(msk, (PATCH_SIZE, PATCH_SIZE), step=PATCH_STEP)

    for pi in range(img_patches.shape[0]):
        for pj in range(img_patches.shape[1]):
            patch_img = img_patches[pi, pj, 0]
            patch_msk = msk_patches[pi, pj]
            non_zero = np.count_nonzero(patch_msk)

            if non_zero >= MIN_POSITIVE_PIXELS:
                # Patch positiva: salva immagine e maschera
                cv2.imwrite(
                    os.path.join(OUT_PATCHES_IMG, f"{name}_p{pi}_{pj}.png"),
                    patch_img,
                )
                cv2.imwrite(
                    os.path.join(OUT_PATCHES_MSK, f"{name}_p{pi}_{pj}.png"),
                    patch_msk,
                )
                positive_count += 1
            elif non_zero == 0:
                # Candidata negativa: maschera tutta zero
                negative_candidates.append((patch_img.copy(), f"{name}_p{pi}_{pj}"))

print(f"Patch positive (≥ {MIN_POSITIVE_PIXELS} pixel non neri): {positive_count}")

# Genera lo stesso numero di patch negative (maschera tutta zero)
mask_zero = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
n_negative = positive_count
if len(negative_candidates) < n_negative:
    # Se non ci sono abbastanza candidate, campiona con ripetizione
    indices = random.choices(range(len(negative_candidates)), k=n_negative)
else:
    indices = random.sample(range(len(negative_candidates)), n_negative)

for idx, save_idx in enumerate(indices):
    patch_img, _ = negative_candidates[save_idx]
    neg_name = f"neg_{idx:05d}"
    cv2.imwrite(os.path.join(OUT_PATCHES_IMG, f"{neg_name}.png"), patch_img)
    cv2.imwrite(os.path.join(OUT_PATCHES_MSK, f"{neg_name}.png"), mask_zero)

print(f"Patch negative (maschera tutta zero): {n_negative}")
print(f"Totale patch: {positive_count} positive + {n_negative} negative = {positive_count + n_negative}")
print(f"Salvate in:\n  {OUT_PATCHES_IMG}\n  {OUT_PATCHES_MSK}")
print("\nDone!")
