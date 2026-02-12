"""
Mask levels to PNG (2 immagini)
================================
Legge la maschera OME-TIFF e genera esattamente 2 immagini PNG:
  1. background (0) + primo livello di colore  → nero + bianco dove c’è il livello 1
  2. background (0) + secondo livello di colore → nero + bianco dove c’è il livello 2

Uso:
    python mask_levels_to_png.py
"""

import os
import numpy as np

# ──────────────────────────────────────────────
# CONFIGURAZIONE
# ──────────────────────────────────────────────
MASKS_DIR = r"C:\Users\laboratorio\Documents\Segmentation\data\images\newimages_1\masks"
MASK_FILENAME = "L_M_CG(1)_1HT_2NT_3CG_background.ome.tif"
MASK_PATH = os.path.join(MASKS_DIR, MASK_FILENAME)
OUTPUT_DIR = r"C:\Users\laboratorio\Documents\Segmentation\mask_levels_output"


def load_mask(path):
    """Carica la maschera da OME-TIFF o TIFF."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File non trovato: {path}")
    try:
        import tifffile
        data = tifffile.imread(path)
    except ImportError:
        import cv2
        data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if data is None:
            raise RuntimeError(f"Impossibile leggere il file: {path}")
    return np.asarray(data)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.basename(MASK_PATH)
    for ext in (".ome.tif", ".ome.tiff", ".tif", ".tiff"):
        if base_name.lower().endswith(ext):
            base_name = base_name[: -len(ext)]
            break

    print(f"Caricamento: {MASK_PATH}")
    data = load_mask(MASK_PATH)
    print(f"Shape: {data.shape}, dtype: {data.dtype}")

    try:
        import cv2
        def save_png(path, img):
            cv2.imwrite(path, np.asarray(img, dtype=np.uint8))
    except ImportError:
        from PIL import Image
        def save_png(path, img):
            Image.fromarray(np.asarray(img, dtype=np.uint8)).save(path)

    # Maschera 2D con valori 0, 1, 2 (un valore per pixel)
    if data.ndim == 2:
        mask = np.asarray(data, dtype=np.intp)
        levels = np.unique(mask)
        print(f"Livelli nella maschera 2D: {levels.tolist()}")
        # Immagine 1: background (0) + livello 1
        img1 = np.where(mask == 1, 255, 0).astype(np.uint8)
        # Immagine 2: background (0) + livello 2
        img2 = np.where(mask == 2, 255, 0).astype(np.uint8)
        name1 = f"{base_name}_background_plus_level_1.png"
        name2 = f"{base_name}_background_plus_level_2.png"
        save_png(os.path.join(OUTPUT_DIR, name1), img1)
        save_png(os.path.join(OUTPUT_DIR, name2), img2)
        print(f"  Salvato: {name1}")
        print(f"  Salvato: {name2}")
        if len(levels) > 3:
            print(f"  Nota: presenti altri livelli {levels.tolist()}; usati solo livello 1 e 2.")
    # Multi-canale (H, W, C) o (C, H, W)
    elif data.ndim == 3:
        if data.shape[-1] <= 16:
            single = np.asarray(data[:, :, 0], dtype=np.intp)  # un canale per eventuale uso
            ch0 = np.asarray(data[:, :, 0], dtype=np.uint8)
            ch1 = np.asarray(data[:, :, 1], dtype=np.uint8)
        else:
            single = np.asarray(data[0], dtype=np.intp)
            ch0 = np.asarray(data[0], dtype=np.uint8)
            ch1 = np.asarray(data[1], dtype=np.uint8)

        # Se i due canali sono identici, interpreta un solo canale: 0=background, altri valori=livelli
        uniq = np.unique(single)
        non_zero = uniq[uniq > 0]
        if len(uniq) >= 2 and np.array_equal(ch0, ch1):
            # Stesso canale ripetuto: primo e secondo livello (valori non zero)
            if len(non_zero) >= 2:
                l1, l2 = int(non_zero[0]), int(non_zero[1])
                print(f"Canali identici; livelli nel canale: {uniq.tolist()}. Immagine 1 = valore {l1}, immagine 2 = valore {l2}.")
                img1 = (single == l1).astype(np.uint8) * 255
                img2 = (single == l2).astype(np.uint8) * 255
            else:
                print(f"Canali identici; un solo livello non-zero: {non_zero.tolist()}. Stesso contenuto in entrambe le immagini.")
                img1 = (single > 0).astype(np.uint8) * 255
                img2 = img1.copy()
        else:
            # Due canali diversi: background + canale 0 e background + canale 1
            img1 = (ch0 > 0).astype(np.uint8) * 255
            img2 = (ch1 > 0).astype(np.uint8) * 255

        name1 = f"{base_name}_background_plus_level_1.png"
        name2 = f"{base_name}_background_plus_level_2.png"
        save_png(os.path.join(OUTPUT_DIR, name1), img1)
        save_png(os.path.join(OUTPUT_DIR, name2), img2)
        print(f"  Salvato: {name1}")
        print(f"  Salvato: {name2}")
    else:
        raise RuntimeError(f"Formato non supportato: shape {data.shape}")

    print(f"\nOutput in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
