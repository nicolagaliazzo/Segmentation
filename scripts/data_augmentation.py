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


def mask_for_save(mask_patch):
    """Scala la maschera (es. 0,1,2,3) in 0-255 per visibilità nel PNG."""
    m = np.asarray(mask_patch, dtype=np.float32)
    mx = float(max(1, m.max()))
    return (m * (255.0 / mx)).astype(np.uint8)

# ──────────────────────────────────────────────
# CONFIGURAZIONE - modifica questi percorsi
# ──────────────────────────────────────────────
BASE_DIR = r"C:\Users\gdf01\Documents\Segmentation\data\images\newimages_1"

IMAGES_DIR = os.path.join(BASE_DIR, "jpg")       # cartella immagini
MASKS_DIR  = os.path.join(BASE_DIR, "masks")      # cartella maschere

# Augmentation: non salvata su disco (solo in memoria per generare le patch)

# Patch: salvate in patch_1_256, divise in train (80%) e validation (20%)
PATCHES_BASE_DIR = r"C:\Users\gdf01\Documents\Segmentation\patch_1_256"
TRAIN_FRACTION = 0.80   # 80% train, 20% validation
RANDOM_SEED_SPLIT = 42  # per riproducibilità train/val

# Estensione dei file sorgente
IMAGE_EXT = "*.jpg"   # immagini in newimages_1/jpg
MASK_EXT  = "*.tif"   # maschere in newimages_1/masks (estensione .tif)

CROP_ROWS = None  # Imposta a 1280 se vuoi tagliare la scale bar (come nel notebook)
                   # None = nessun crop

PATCH_SIZE = 256   # Dimensione delle patch quadrate
PATCH_STEP = 256   # Step (non-overlapping se uguale a PATCH_SIZE)
MIN_POSITIVE_PIXELS = 400   # Salva patch solo se la maschera ha almeno N pixel non neri
EXCLUDE_TI_IN_NAME = True  # Se True, ignora immagini con "TI" nel nome file

SHOW_PLOTS = False  # True per visualizzare i plot come nel notebook

# ──────────────────────────────────────────────
# CREAZIONE CARTELLE OUTPUT (solo per patch)
# ──────────────────────────────────────────────
for split in ("train", "validation"):
    for sub in ("images", "masks"):
        os.makedirs(os.path.join(PATCHES_BASE_DIR, split, sub), exist_ok=True)

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
if EXCLUDE_TI_IN_NAME:
    pairs = [(i, m) for i, m in zip(image_paths, mask_paths) if "TI" not in os.path.basename(i)]
    image_paths = [p[0] for p in pairs]
    mask_paths = [p[1] for p in pairs]
    print(f"Escluse immagini con 'TI' nel nome. Restano {len(image_paths)} coppie immagine/maschera.")
assert len(image_paths) > 0, "Nessuna immagine rimasta dopo il filtro (es. TI)."

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
# 2. DATA AUGMENTATION — 5 versioni per immagine (solo in memoria, non salvate su disco)
#    Originale, Saturazione 0.5 (maschera invariata), Rotazione 90°, Flip H, Flip V
# ──────────────────────────────────────────────
print("\nApplicazione augmentation (orig, sat 0.5, rot90, flipH, flipV) in memoria...")

augmented_list = []   # lista di (img_bgr, mask, name) per generazione patch
num_images = len(image_dataset)

for i in range(num_images):
    img = image_dataset[i]
    msk = mask_dataset[i]
    base_name = os.path.splitext(os.path.basename(image_paths[i]))[0]

    # 1) Originale
    augmented_list.append((cv2.cvtColor(img, cv2.COLOR_RGB2BGR), msk.copy(), f"{base_name}_orig"))
    # 2) Saturazione 0.5
    img_sat = apply_saturation(img, factor=0.5)
    augmented_list.append((cv2.cvtColor(img_sat, cv2.COLOR_RGB2BGR), msk.copy(), f"{base_name}_sat"))
    # 3) Rotazione 90°
    img_rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    msk_rot = cv2.rotate(msk, cv2.ROTATE_90_CLOCKWISE)
    augmented_list.append((cv2.cvtColor(img_rot, cv2.COLOR_RGB2BGR), msk_rot, f"{base_name}_rot90"))
    # 4) Flip orizzontale
    img_flipH = cv2.flip(img, 1)
    msk_flipH = cv2.flip(msk, 1)
    augmented_list.append((cv2.cvtColor(img_flipH, cv2.COLOR_RGB2BGR), msk_flipH, f"{base_name}_flipH"))
    # 5) Flip verticale
    img_flipV = cv2.flip(img, 0)
    msk_flipV = cv2.flip(msk, 0)
    augmented_list.append((cv2.cvtColor(img_flipV, cv2.COLOR_RGB2BGR), msk_flipV, f"{base_name}_flipV"))

    if (i + 1) % 10 == 0:
        print(f"  Processate {i + 1}/{num_images}")

print(f"Augmentation completata: {num_images} immagini x 5 = {len(augmented_list)} versioni (non salvate su disco).")

# ──────────────────────────────────────────────
# 3. PATCHIFY — Patch 256x256 solo se >= MIN_POSITIVE_PIXELS non neri; + bilanciamento negative
# ──────────────────────────────────────────────
try:
    from patchify import patchify
except ImportError:
    print("\nInstallo patchify...")
    os.system("pip install patchify")
    from patchify import patchify

print(f"\nCreazione patch {PATCH_SIZE}x{PATCH_SIZE} (solo se maschera ha >= {MIN_POSITIVE_PIXELS} pixel non neri)...")
print(f"Salvataggio in {PATCHES_BASE_DIR} con split train ({TRAIN_FRACTION*100:.0f}%) / validation ({(1-TRAIN_FRACTION)*100:.0f}%)")

# Fase 1: raccogli patch positive e candidate negative (da dati augmentation in memoria)
positive_patches = []   # lista di (patch_img, patch_msk, nome_file)
negative_candidates = []   # lista di (patch_img,) per patch con maschera tutta zero

for img, msk, name in augmented_list:

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
                positive_patches.append((patch_img.copy(), patch_msk.copy(), f"{name}_p{pi}_{pj}"))
            elif non_zero == 0:
                negative_candidates.append((patch_img.copy(),))

n_positive = len(positive_patches)
print(f"Patch positive trovate: {n_positive}")

# Fase 2: split deterministico 80/20 per le positive (stesso numero train/val per positive e negative)
random.seed(RANDOM_SEED_SPLIT)
order = list(range(n_positive))
random.shuffle(order)
n_train = int(round(n_positive * TRAIN_FRACTION))
n_val = n_positive - n_train

# Salva patch positive: prime n_train -> train, rest -> validation
for idx, pos_idx in enumerate(order):
    patch_img, patch_msk, base_name = positive_patches[pos_idx]
    split = "train" if idx < n_train else "validation"
    out_img_dir = os.path.join(PATCHES_BASE_DIR, split, "images")
    out_msk_dir = os.path.join(PATCHES_BASE_DIR, split, "masks")
    cv2.imwrite(os.path.join(out_img_dir, f"{base_name}.png"), patch_img)
    cv2.imwrite(os.path.join(out_msk_dir, f"{base_name}.png"), mask_for_save(patch_msk))

print(f"Patch positive (>= {MIN_POSITIVE_PIXELS} pixel non neri): {n_positive} (train: {n_train}, validation: {n_val})")

# Fase 3: stesso numero di patch negative, stessa distribuzione (n_train in train, n_val in validation)
mask_zero = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
if len(negative_candidates) < n_positive:
    neg_indices = random.choices(range(len(negative_candidates)), k=n_positive)
else:
    neg_indices = random.sample(range(len(negative_candidates)), n_positive)
random.seed(RANDOM_SEED_SPLIT + 1)
random.shuffle(neg_indices)
for idx in range(n_positive):
    patch_img = negative_candidates[neg_indices[idx]][0]
    split = "train" if idx < n_train else "validation"
    neg_name = f"neg_{idx:05d}"
    out_img_dir = os.path.join(PATCHES_BASE_DIR, split, "images")
    out_msk_dir = os.path.join(PATCHES_BASE_DIR, split, "masks")
    cv2.imwrite(os.path.join(out_img_dir, f"{neg_name}.png"), patch_img)
    cv2.imwrite(os.path.join(out_msk_dir, f"{neg_name}.png"), mask_zero)

print(f"Patch negative (maschera tutta zero): {n_positive} (train: {n_train}, validation: {n_val})")
print(f"Totale: {n_positive * 2} patch. In train: {n_train * 2} (pos+neg), in validation: {n_val * 2} (pos+neg).")
print(f"Salvate in:\n  {PATCHES_BASE_DIR}\n    train/images, train/masks\n    validation/images, validation/masks")
print("\nDone!")
